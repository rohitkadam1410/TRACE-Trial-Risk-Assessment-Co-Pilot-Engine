import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

logger = logging.getLogger("text_inference")

class TextRiskPredictor:
    """Loads a fine-tuned LoRA BERT model to predict clinical trial risk directly from protocol text."""
    def __init__(self, base_model_name: str = "emilyalsentzer/Bio_ClinicalBERT", lora_path: str = "artifacts/lora_bert"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model (matching finetune.py configuration)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=2,
            ignore_mismatched_sizes=True,
            revision="refs/pr/16",
            use_safetensors=True
        )
        
        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded LoRA model from {lora_path} to {self.device}")

    def predict_risk(self, text: str, max_len: int = 512) -> float:
        """
        Tokenizes the text, runs a forward pass, and returns the probability of the positive class (termination/high risk).
        """
        enc = self.tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Softmax to get probabilities, then take the probability of class 1
            probs = torch.softmax(outputs.logits, dim=-1)
            prob_positive = float(probs[0, 1].cpu().item())
            
        return prob_positive
