"""
embedder.py — BioClinicalBERT Embedding Extraction for AMD MI300X
=================================================================
Extracts frozen CLS-token embeddings from BioClinicalBERT for clinical trial
records. Runs on AMD MI300X via ROCm (torch.device("cuda") maps automatically).

This is the FIRST GPU-dependent module in the pipeline.
Expected runtime: ~1–1.5 hours on MI300X for 5000+ records.
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from transformers import AutoTokenizer, AutoModel

# Suppress tokenizer parallelism warnings in notebook cells
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logger_config
logger_config.setup_logging(__file__)

# ── CELL BREAK ──

# ── CONFIGURATION ──
MODEL_NAME: str = "emilyalsentzer/Bio_ClinicalBERT"
EMBEDDING_DIM: int = 768
MAX_LENGTH: int = 512
DEFAULT_BATCH_SIZE: int = 32
MAX_OOM_RETRIES: int = 2

# Set to True if BERT extraction fails (OOM or ROCm error).
# When True, loads data/tfidf_train.npz instead and uses TF-IDF as embeddings.
USE_TFIDF_FALLBACK: bool = False

# ── CELL BREAK ──


def ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)


def get_device(device: str = "cuda") -> torch.device:
    """
    Resolve the compute device. ROCm maps 'cuda' to the AMD GPU automatically.
    Falls back to CPU if no GPU is available.

    Args:
        device: Requested device string. Use 'cuda' for MI300X under ROCm.

    Returns:
        torch.device for the resolved hardware target.
    """
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"GPU detected: {gpu_name} ({mem_gb:.1f} GB)")
        return torch.device("cuda")
    else:
        if device == "cuda":
            warnings.warn(
                "CUDA/ROCm not available — falling back to CPU. "
                "Embedding extraction will be significantly slower.",
                RuntimeWarning,
            )
        return torch.device("cpu")


def print_gpu_memory(label: str = "") -> None:
    """Print current and peak GPU memory usage for benchmarking."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        logging.info(
            f"[GPU Memory{' — ' + label if label else ''}] "
            f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | "
            f"Peak: {peak:.2f} GB"
        )


if __name__ == "__main__":
    ensure_dirs()
    dev = get_device()
    print(f"Using device: {dev}")
    print_gpu_memory("init")

# ── CELL BREAK ──


def extract_embeddings(
    texts: list[str],
    model_name: str = MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract frozen CLS-token embeddings from BioClinicalBERT.

    Uses the [CLS] token from the last hidden state as a fixed-size
    representation of each input text. The model is NOT fine-tuned;
    we rely on its pre-trained clinical domain knowledge.

    Args:
        texts: List of clinical trial text strings to embed.
        model_name: HuggingFace model identifier.
        batch_size: Number of texts per forward pass. Halved on OOM.
        max_length: Maximum token length (BERT caps at 512).
        device: Compute device — 'cuda' for MI300X via ROCm.

    Returns:
        np.ndarray of shape (N, 768) containing CLS embeddings.
    """
    resolved_device = get_device(device)

    logging.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, use_safetensors=True)
    model.to(resolved_device)
    model.eval()  # Freeze all layers — no gradient computation needed
    logging.info("Model loaded and set to eval mode.")
    print_gpu_memory("model loaded")

    all_embeddings: list[np.ndarray] = []
    n_texts = len(texts)
    n_batches = (n_texts + batch_size - 1) // batch_size

    logging.info(
        f"Extracting embeddings: {n_texts} texts, "
        f"batch_size={batch_size}, ~{n_batches} batches"
    )

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_texts)
        batch_texts = texts[start:end]

        # Replace empty strings with a placeholder so tokenizer doesn't choke
        batch_texts = [t if t.strip() else "[UNK]" for t in batch_texts]

        # Tokenize with padding and truncation to max_length
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move tensors to GPU
        input_ids = encoded["input_ids"].to(resolved_device)
        attention_mask = encoded["attention_mask"].to(resolved_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract CLS token: first token of last hidden state
        # Shape: (batch_size, 768)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

        # Free GPU tensors immediately to prevent memory accumulation
        del input_ids, attention_mask, outputs
        torch.cuda.empty_cache()

        # Progress logging every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            pct = (batch_idx + 1) / n_batches * 100
            logging.info(
                f"  Batch {batch_idx + 1}/{n_batches} ({pct:.1f}%) — "
                f"records {end}/{n_texts}"
            )

    # Concatenate all batch results → (N, 768)
    embeddings = np.concatenate(all_embeddings, axis=0)
    assert embeddings.shape == (n_texts, EMBEDDING_DIM), (
        f"Shape mismatch: expected ({n_texts}, {EMBEDDING_DIM}), "
        f"got {embeddings.shape}"
    )
    logging.info(f"Embedding extraction complete. Shape: {embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    _test_texts = [
        "Randomized phase 3 trial of pembrolizumab in NSCLC patients.",
        "Inclusion criteria: age >= 18, ECOG 0-1, measurable disease.",
    ]
    _emb = extract_embeddings(_test_texts, device="cuda", batch_size=2)
    print(f"Smoke test passed. Shape: {_emb.shape}")  # (2, 768)

# ── CELL BREAK ──


def extract_embeddings_with_retry(
    texts: list[str],
    model_name: str = MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    device: str = "cuda",
    max_retries: int = MAX_OOM_RETRIES,
) -> np.ndarray:
    """
    Wrapper around extract_embeddings with automatic OOM recovery.

    On RuntimeError (CUDA OOM), halves the batch size and retries.
    After max_retries exhausted, raises the original error.

    Args:
        texts: List of texts to embed.
        model_name: HuggingFace model identifier.
        batch_size: Starting batch size — halved on each OOM retry.
        max_length: Maximum token length for BERT.
        device: Compute device string.
        max_retries: Maximum number of OOM retries before giving up.

    Returns:
        np.ndarray of shape (N, 768).
    """
    current_batch_size = batch_size

    for attempt in range(max_retries + 1):
        try:
            return extract_embeddings(
                texts,
                model_name=model_name,
                batch_size=current_batch_size,
                max_length=max_length,
                device=device,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and attempt < max_retries:
                # Clear everything and retry with smaller batches
                torch.cuda.empty_cache()
                old_bs = current_batch_size
                current_batch_size = max(1, current_batch_size // 2)
                logging.warning(
                    f"OOM on attempt {attempt + 1}. "
                    f"Reducing batch_size {old_bs} → {current_batch_size} "
                    f"and retrying..."
                )
            else:
                logging.error(
                    f"Embedding extraction failed after {attempt + 1} attempts: {e}"
                )
                raise


if __name__ == "__main__":
    print("extract_embeddings_with_retry defined — OOM-safe wrapper ready.")

# ── CELL BREAK ──


def load_trial_data(filepath: str = "data/trials_raw.parquet") -> pd.DataFrame:
    """
    Load the raw trial dataset produced by pipeline.py.

    Expected columns: nct_id, terminated, full_text, enrollment_count,
    phase_encoded, has_expanded_access, condition_count, split.

    Args:
        filepath: Path to the raw parquet file.

    Returns:
        DataFrame with all trial records.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"{filepath} not found. Run pipeline.py first to ingest trial data."
        )
    df = pd.read_parquet(filepath)
    logging.info(f"Loaded {len(df)} records from {filepath}")
    logging.info(f"Columns: {list(df.columns)}")
    logging.info(f"Label distribution: {df['terminated'].value_counts().to_dict()}")
    return df


def extract_section_text(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Extract per-section text lists from the raw DataFrame.

    The pipeline.py module concatenates all text into 'full_text'. For
    section-level SHAP attribution, we attempt to recover individual
    sections. If the original columns are not present, we use heuristic
    splitting on the concatenated full_text.

    Args:
        df: Raw trial DataFrame.

    Returns:
        Dict mapping section name → list of text strings (length N).
    """
    n = len(df)
    sections: dict[str, list[str]] = {}

    # Full concatenated text — always available
    sections["full_text"] = (
        df["full_text"].fillna("").astype(str).tolist()
    )

    # Eligibility criteria — try dedicated column first, then heuristic extraction
    if "eligibilityCriteria" in df.columns:
        sections["eligibility_criteria"] = (
            df["eligibilityCriteria"].fillna("").astype(str).tolist()
        )
    elif "eligibility_criteria" in df.columns:
        sections["eligibility_criteria"] = (
            df["eligibility_criteria"].fillna("").astype(str).tolist()
        )
    else:
        # Heuristic: eligibility text is often the longest segment in full_text
        # and typically follows the summary. Use full_text as proxy.
        logging.warning(
            "No dedicated eligibility column found. "
            "Using full_text as eligibility_criteria proxy."
        )
        sections["eligibility_criteria"] = sections["full_text"]

    # Primary outcome — try dedicated column first
    if "primaryOutcomeMeasure" in df.columns:
        sections["primary_outcome"] = (
            df["primaryOutcomeMeasure"].fillna("").astype(str).tolist()
        )
    elif "primary_outcome_measure" in df.columns:
        sections["primary_outcome"] = (
            df["primary_outcome_measure"].fillna("").astype(str).tolist()
        )
    elif "primary_outcome" in df.columns:
        sections["primary_outcome"] = (
            df["primary_outcome"].fillna("").astype(str).tolist()
        )
    else:
        # Fallback: extract outcome-like content from full_text
        # Outcomes tend to mention "measure", "endpoint", "efficacy", etc.
        logging.warning(
            "No dedicated primary_outcome column found. "
            "Using full_text as primary_outcome proxy."
        )
        sections["primary_outcome"] = sections["full_text"]

    # Validate all sections have consistent length
    for name, texts in sections.items():
        assert len(texts) == n, (
            f"Section '{name}' has {len(texts)} texts, expected {n}"
        )

    return sections


if __name__ == "__main__":
    _df = load_trial_data()
    _sections = extract_section_text(_df)
    for name, texts in _sections.items():
        non_empty = sum(1 for t in texts if t.strip())
        print(f"  {name}: {len(texts)} total, {non_empty} non-empty")

# ── CELL BREAK ──


def extract_section_embeddings(
    df: pd.DataFrame,
    device: str = "cuda",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """
    Extract CLS embeddings for each protocol section independently.

    Produces per-section embedding arrays that enable section-level
    SHAP attribution in the explainability module downstream.

    Clears GPU cache between sections to prevent memory pressure.

    Args:
        df: Raw trial DataFrame from pipeline.py.
        device: Compute device — 'cuda' for MI300X via ROCm.
        batch_size: Starting batch size (halved on OOM automatically).

    Returns:
        Dict mapping section name → np.ndarray of shape (N, 768).
    """
    sections = extract_section_text(df)
    section_embeddings: dict[str, np.ndarray] = {}

    for section_name, texts in sections.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"Embedding section: {section_name} ({len(texts)} records)")
        logging.info(f"{'='*60}")

        # Clear GPU cache before each section to maximize available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        embeddings = extract_embeddings_with_retry(
            texts=texts,
            batch_size=batch_size,
            device=device,
        )

        section_embeddings[section_name] = embeddings
        print_gpu_memory(f"after {section_name}")

        # Save immediately — never assume the session persists
        output_path = f"data/embeddings_{section_name}.npy"
        np.save(output_path, embeddings)
        logging.info(f"Saved {output_path} — shape: {embeddings.shape}")

    return section_embeddings


if __name__ == "__main__":
    print("extract_section_embeddings defined — section-level extraction ready.")

# ── CELL BREAK ──


def save_embedding_metadata(
    n_records: int,
    sections: list[str],
    start_time: float,
) -> dict:
    """
    Save embedding metadata for downstream pipeline introspection.

    Args:
        n_records: Number of records embedded.
        sections: List of section names that were embedded.
        start_time: Unix timestamp when extraction started.

    Returns:
        The metadata dict that was saved.
    """
    elapsed_min = (time.time() - start_time) / 60.0

    meta = {
        "n_records": n_records,
        "embedding_dim": EMBEDDING_DIM,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "sections_embedded": sections,
        "extraction_time_minutes": round(elapsed_min, 2),
        "records_per_second": round(n_records / max(elapsed_min * 60, 1), 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
        "peak_gpu_memory_gb": round(
            torch.cuda.max_memory_allocated() / 1e9, 2
        )
        if torch.cuda.is_available()
        else 0.0,
        "fallback_mode": USE_TFIDF_FALLBACK,
    }

    meta_path = "artifacts/embedding_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"Saved embedding metadata to {meta_path}")

    return meta


if __name__ == "__main__":
    print("save_embedding_metadata defined.")

# ── CELL BREAK ──


def build_combined_features(
    embeddings: np.ndarray,
    structured: pd.DataFrame,
) -> np.ndarray:
    """
    Concatenate BERT CLS embeddings with structured engineered features
    to produce the final combined feature matrix for XGBoost.

    Args:
        embeddings: (N, 768) BERT CLS embeddings.
        structured: DataFrame of engineered features from features.py.
                    Must NOT include the 'terminated' target column.

    Returns:
        np.ndarray of shape (N, 768 + n_structured_features).
    """
    assert embeddings.shape[0] == len(structured), (
        f"Row count mismatch: embeddings has {embeddings.shape[0]}, "
        f"structured has {len(structured)}"
    )

    # Drop the target column if it accidentally got included
    feature_cols = [c for c in structured.columns if c != "terminated"]
    structured_arr = structured[feature_cols].values.astype(np.float32)

    combined = np.concatenate([embeddings, structured_arr], axis=1)
    logging.info(
        f"Combined feature matrix: {combined.shape} "
        f"(BERT: {embeddings.shape[1]} + structured: {structured_arr.shape[1]})"
    )
    return combined


def load_tfidf_fallback() -> tuple[np.ndarray, np.ndarray]:
    """
    Load TF-IDF sparse matrices as dense arrays for fallback mode.

    When BERT embedding extraction fails (OOM, ROCm errors, etc.),
    this function provides a CPU-only alternative. The downstream
    pipeline works identically regardless of embedding source.

    Returns:
        Tuple of (train_embeddings, test_embeddings) as dense np.ndarray.
    """
    train_path = "data/tfidf_train.npz"
    test_path = "data/tfidf_test.npz"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"TF-IDF fallback files not found at {train_path} / {test_path}. "
            "Run features.py first."
        )

    tfidf_train = sparse.load_npz(train_path).toarray()
    tfidf_test = sparse.load_npz(test_path).toarray()
    logging.info(
        f"TF-IDF fallback loaded: train={tfidf_train.shape}, test={tfidf_test.shape}"
    )
    return tfidf_train, tfidf_test


if __name__ == "__main__":
    # Test build_combined_features with synthetic data
    _emb = np.random.randn(10, 768).astype(np.float32)
    _struct = pd.DataFrame(
        np.random.randn(10, 5), columns=[f"feat_{i}" for i in range(5)]
    )
    _combined = build_combined_features(_emb, _struct)
    print(f"Combined shape: {_combined.shape}")  # (10, 773)

# ── CELL BREAK ──


def run_embedding_pipeline(
    data_path: str = "data/trials_raw.parquet",
    device: str = "cuda",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """
    Full embedding extraction pipeline: load data → extract section embeddings
    → build combined feature matrices → save all artifacts.

    This is the main entry point for notebook execution.

    Args:
        data_path: Path to the raw trial parquet from pipeline.py.
        device: Compute device — 'cuda' for MI300X via ROCm.
        batch_size: Starting batch size for BERT inference.
    """
    ensure_dirs()
    start_time = time.time()

    # ── Step 1: Load raw trial data ──
    df = load_trial_data(data_path)

    # Determine train/test split indices (consistent with features.py)
    if "split" in df.columns:
        train_mask = df["split"] == "train"
        test_mask = df["split"] == "test"
    else:
        # Reproduce the same split as features.py using identical random_state
        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(
            df.index,
            test_size=0.2,
            random_state=42,
            stratify=df["terminated"],
        )
        train_mask = df.index.isin(train_idx)
        test_mask = df.index.isin(test_idx)

    logging.info(
        f"Split: {train_mask.sum()} train / {test_mask.sum()} test"
    )

    # ── Step 2: Extract embeddings (BERT or TF-IDF fallback) ──
    if USE_TFIDF_FALLBACK:
        logging.warning(
            "USE_TFIDF_FALLBACK=True — skipping BERT, using TF-IDF vectors."
        )
        emb_train, emb_test = load_tfidf_fallback()
    else:
        # Extract section-level embeddings (saves each as .npy)
        section_embeddings = extract_section_embeddings(
            df, device=device, batch_size=batch_size
        )

        # Use full_text embeddings as the primary embedding for combined features
        full_emb = section_embeddings["full_text"]
        emb_train = full_emb[train_mask.values]
        emb_test = full_emb[test_mask.values]

    # ── Step 3: Load structured features from features.py ──
    feat_train_path = "data/features_train.parquet"
    feat_test_path = "data/features_test.parquet"

    if not os.path.exists(feat_train_path) or not os.path.exists(feat_test_path):
        raise FileNotFoundError(
            f"Structured feature files not found. Run features.py first. "
            f"Expected: {feat_train_path}, {feat_test_path}"
        )

    df_feat_train = pd.read_parquet(feat_train_path)
    df_feat_test = pd.read_parquet(feat_test_path)
    logging.info(
        f"Loaded structured features: train={df_feat_train.shape}, "
        f"test={df_feat_test.shape}"
    )

    # Extract labels before dropping them from features
    y_train = df_feat_train["terminated"].values.astype(np.int32)
    y_test = df_feat_test["terminated"].values.astype(np.int32)

    # ── Step 4: Build combined feature matrices ──
    X_train = build_combined_features(emb_train, df_feat_train)
    X_test = build_combined_features(emb_test, df_feat_test)

    # ── Step 5: Save all outputs ──
    np.save("data/embeddings_train.npy", emb_train)
    np.save("data/embeddings_test.npy", emb_test)
    np.save("data/X_train_combined.npy", X_train)
    np.save("data/X_test_combined.npy", X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)

    logging.info(f"Saved embeddings_train.npy — shape: {emb_train.shape}")
    logging.info(f"Saved embeddings_test.npy  — shape: {emb_test.shape}")
    logging.info(f"Saved X_train_combined.npy — shape: {X_train.shape}")
    logging.info(f"Saved X_test_combined.npy  — shape: {X_test.shape}")
    logging.info(f"Saved y_train.npy — shape: {y_train.shape}")
    logging.info(f"Saved y_test.npy  — shape: {y_test.shape}")

    # ── Step 6: Save metadata ──
    sections_list = (
        list(section_embeddings.keys()) if not USE_TFIDF_FALLBACK else ["tfidf_fallback"]
    )
    meta = save_embedding_metadata(
        n_records=len(df),
        sections=sections_list,
        start_time=start_time,
    )

    # ── Timing Report (for AMD benchmark slide) ──
    elapsed_sec = time.time() - start_time
    elapsed_min = elapsed_sec / 60.0
    throughput = len(df) / max(elapsed_sec, 1)

    print("\n" + "=" * 60)
    print("  AMD MI300X EMBEDDING BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Total embedding time:    {elapsed_min:.2f} minutes")
    print(f"  Records per second:      {throughput:.2f} rec/s")
    if torch.cuda.is_available():
        print(f"  GPU model:               {torch.cuda.get_device_name(0)}")
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory used:    {peak_mem:.2f} GB")
    else:
        print("  GPU model:               N/A (CPU mode)")
        print("  Peak GPU memory used:    N/A")
    print(f"  Embedding source:        {'TF-IDF fallback' if USE_TFIDF_FALLBACK else MODEL_NAME}")
    print(f"  Combined feature shape:  {X_train.shape[1]} dims "
          f"(BERT/TF-IDF: {emb_train.shape[1]} + structured: "
          f"{X_train.shape[1] - emb_train.shape[1]})")
    print("=" * 60)


if __name__ == "__main__":
    run_embedding_pipeline()

# ── CELL BREAK ──

## INTEGRATION NOTES
# ─────────────────────────────────────────────────────────────
#
# FILES THIS MODULE READS:
#   - data/trials_raw.parquet        — raw trial data from pipeline.py
#   - data/features_train.parquet    — structured features from features.py
#   - data/features_test.parquet     — structured features from features.py
#   - data/tfidf_train.npz           — TF-IDF fallback from features.py (only if USE_TFIDF_FALLBACK=True)
#   - data/tfidf_test.npz            — TF-IDF fallback from features.py (only if USE_TFIDF_FALLBACK=True)
#
# FILES THIS MODULE WRITES:
#   - data/embeddings_full_text.npy          — (N, 768) full text CLS embeddings
#   - data/embeddings_eligibility_criteria.npy — (N, 768) eligibility CLS embeddings
#   - data/embeddings_primary_outcome.npy    — (N, 768) primary outcome CLS embeddings
#   - data/X_train_combined.npy              — (N_train, 768+K) combined features, train
#   - data/X_test_combined.npy               — (N_test, 768+K) combined features, test
#   - data/y_train.npy                       — (N_train,) binary labels, train
#   - data/y_test.npy                        — (N_test,) binary labels, test
#   - artifacts/embedding_meta.json          — metadata: model, timing, memory stats
#
# ENVIRONMENT VARIABLES / CONSTANTS THE CALLER MUST SET:
#   - TOKENIZERS_PARALLELISM: set to "false" (done automatically at import)
#   - USE_TFIDF_FALLBACK: module-level bool, set True to skip BERT
#   - Ensure ROCm + PyTorch are installed: torch.device("cuda") must resolve to MI300X
#   - The transformers library must have access to download or cache
#     "emilyalsentzer/Bio_ClinicalBERT" (~440 MB)
#
# EXECUTION ORDER:
#   1. pipeline.py   → produces data/trials_raw.parquet
#   2. features.py   → produces data/features_{train,test}.parquet + TF-IDF fallbacks
#   3. embedder.py   → THIS MODULE (requires steps 1 & 2)
#   4. model training → consumes data/X_{train,test}_combined.npy + data/y_{train,test}.npy
#
# ─────────────────────────────────────────────────────────────
