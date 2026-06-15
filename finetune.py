import logging, json, time, joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────
MODEL_NAME   = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN      = 512
BATCH_SIZE   = 16        # safe for MI300X, increase to 32 if no OOM
EPOCHS       = 5
LR           = 2e-4      # higher LR works well with LoRA
DEVICE       = torch.device("cuda")   # ROCm maps this to MI300X
SAVE_PATH    = "artifacts/lora_bert"

# ── LoRA config ──────────────────────────────────────────────
# Only trains ~0.5% of parameters — fast, stable, no full fine-tune needed
LORA_CONFIG = LoraConfig(
    task_type    = TaskType.SEQ_CLS,
    r            = 16,          # rank — higher = more capacity, more memory
    lora_alpha   = 32,
    lora_dropout = 0.1,
    target_modules = ["query", "value"],   # only attention Q and V matrices
    bias         = "none",
)

# ── Dataset ──────────────────────────────────────────────────
class TrialDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.texts  = df["full_text"].fillna("").tolist()
        self.labels = df["terminated"].astype(int).tolist()
        self.tok    = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length    = self.max_len,
            padding       = "max_length",
            truncation    = True,
            return_tensors = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── Training loop ────────────────────────────────────────────
def train():
    log.info(f"Device: {torch.cuda.get_device_name(0)}")
    log.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Load data
    df_train = pd.read_parquet("data/trials_raw.parquet")
    df_train = df_train[df_train["split"] == "train"].reset_index(drop=True)
    df_test  = pd.read_parquet("data/trials_raw.parquet")
    df_test  = df_test[df_test["split"] == "test"].reset_index(drop=True)

    log.info(f"Train: {len(df_train)}, Test: {len(df_test)}")
    log.info(f"Positive rate: {df_train['terminated'].mean():.1%}")

    # Tokenizer + model
    log.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = 2,
        ignore_mismatched_sizes = True,
        revision = "refs/pr/16",    # Fix for torch.load CVE: pull safetensors from HF PR
        use_safetensors = True
    )

    # Apply LoRA — only trains ~800K params instead of 110M
    model = get_peft_model(base_model, LORA_CONFIG)
    model.print_trainable_parameters()
    model = model.to(DEVICE)

    # Datasets
    train_ds = TrialDataset(df_train, tokenizer)
    test_ds  = TrialDataset(df_test,  tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

    # Class weights for imbalance
    pos_rate = df_train["terminated"].mean()
    weight   = torch.tensor([pos_rate, 1 - pos_rate]).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = total_steps // 10,
        num_training_steps = total_steps,
    )

    best_auroc = 0.0

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        t0 = time.time()
        total_loss = 0

        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            loss    = criterion(outputs.logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 20 == 0:
                mem = torch.cuda.memory_allocated() / 1e9
                log.info(f"Epoch {epoch+1} step {step}/{len(train_dl)} "
                         f"loss={loss.item():.4f} gpu_mem={mem:.1f}GB")

        avg_loss = total_loss / len(train_dl)
        train_time = time.time() - t0

        # ── Evaluate ──
        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in test_dl:
                input_ids = batch["input_ids"].to(DEVICE)
                attn_mask = batch["attention_mask"].to(DEVICE)
                labels    = batch["labels"].cpu().numpy()

                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                probs   = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

                all_probs.extend(probs)
                all_labels.extend(labels)

        auroc = roc_auc_score(all_labels, all_probs)
        log.info(f"Epoch {epoch+1}/{EPOCHS} | loss={avg_loss:.4f} | "
                 f"AUROC={auroc:.4f} | time={train_time:.1f}s")

        # Save best
        if auroc > best_auroc:
            best_auroc = auroc
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            log.info(f"  ✅ New best AUROC={auroc:.4f} — saved to {SAVE_PATH}")

        torch.cuda.empty_cache()

    log.info(f"\nTraining complete. Best AUROC: {best_auroc:.4f}")
    log.info(f"Model saved to: {SAVE_PATH}")

    # Save inference metadata
    json.dump({
        "model_path": SAVE_PATH,
        "model_name": MODEL_NAME,
        "best_auroc": best_auroc,
        "max_len": MAX_LEN,
        "epochs": EPOCHS,
        "lora_rank": LORA_CONFIG.r,
    }, open("artifacts/lora_meta.json", "w"), indent=2)

if __name__ == "__main__":
    train()
