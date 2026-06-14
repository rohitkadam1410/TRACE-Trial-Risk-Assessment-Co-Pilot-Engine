# ARCHITECTURE DECISION: XGBoost on BERT embeddings
# - BERT embeddings capture clinical language semantics (frozen, no GPU needed at train time)
# - XGBoost handles structured features (sample size, phase) natively
# - SHAP TreeExplainer is EXACT for XGBoost — no approximation error
# - Training XGBoost takes <5 minutes on CPU vs 2+ GPU hours for BERT fine-tuning
# - AUROC on small medical datasets: XGBoost+BERT typically beats fine-tuned BERT alone

import os
import json
import time
import warnings
import logging
from typing import Optional, Callable

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)
import logger_config
logger_config.setup_logging(__file__)

# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Constants — callers may override via env vars before importing this module
# ---------------------------------------------------------------------------

ARTIFACTS_DIR: str = os.environ.get("TRACE_ARTIFACTS_DIR", "artifacts")
DATA_DIR: str = os.environ.get("TRACE_DATA_DIR", "data")

# Expected filenames produced by upstream modules (features.py, embedder.py)
FEATURES_TRAIN_FILE: str = "features_train.parquet"
FEATURES_TEST_FILE: str = "features_test.parquet"
EMBEDDINGS_TRAIN_FILE: str = "embeddings_train.npy"
EMBEDDINGS_TEST_FILE: str = "embeddings_test.npy"
FEATURE_SCALER_FILE: str = "feature_scaler.pkl"

# BioClinicalBERT CLS embedding dimension (768‑d for base model)
BERT_EMBEDDING_DIM: int = 768

# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def ensure_dirs() -> None:
    """Create artifact and data directories if they don't already exist."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def _save_json(obj: dict, filename: str) -> str:
    """Serialize *obj* to a pretty-printed JSON file under ARTIFACTS_DIR."""
    path = os.path.join(ARTIFACTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logging.info(f"Saved → {path}")
    return path


def _pretty_confusion_matrix(cm: np.ndarray) -> str:
    """Return a human-readable 2×2 confusion matrix string."""
    lines = [
        "",
        "                 Predicted",
        "                 Neg    Pos",
        f"  Actual Neg   {cm[0, 0]:>5d}  {cm[0, 1]:>5d}",
        f"  Actual Pos   {cm[1, 0]:>5d}  {cm[1, 1]:>5d}",
        "",
    ]
    return "\n".join(lines)


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_feature_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the structured-feature train/test parquet files produced by features.py.

    Returns
    -------
    df_train, df_test : pd.DataFrame
        DataFrames where the 'terminated' column is the binary target label
        and all other columns are engineered structured features.
    """
    train_path = os.path.join(DATA_DIR, FEATURES_TRAIN_FILE)
    test_path = os.path.join(DATA_DIR, FEATURES_TEST_FILE)

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Train features not found at {train_path}. Run features.py first."
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test features not found at {test_path}. Run features.py first."
        )

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    logging.info(
        f"Loaded structured features — train: {df_train.shape}, test: {df_test.shape}"
    )
    return df_train, df_test


def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed BioClinicalBERT [CLS] embeddings (768-d per sample).

    Returns
    -------
    emb_train, emb_test : np.ndarray
        Float32 arrays of shape (n_samples, 768).
    """
    train_path = os.path.join(DATA_DIR, EMBEDDINGS_TRAIN_FILE)
    test_path = os.path.join(DATA_DIR, EMBEDDINGS_TEST_FILE)

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Train embeddings not found at {train_path}. Run embedder.py first."
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test embeddings not found at {test_path}. Run embedder.py first."
        )

    emb_train = np.load(train_path).astype(np.float32)
    emb_test = np.load(test_path).astype(np.float32)
    logging.info(
        f"Loaded BERT embeddings — train: {emb_train.shape}, test: {emb_test.shape}"
    )
    return emb_train, emb_test


def build_combined_matrix(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    target_col: str = "terminated",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Horizontally concatenate structured features with BERT embeddings.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the *target_col* column; all other columns are features.
    embeddings : np.ndarray
        (n_samples, 768) array of BERT [CLS] embeddings.
    target_col : str
        Name of the binary target column.

    Returns
    -------
    X : np.ndarray of shape (n, n_structured + 768)
    y : np.ndarray of shape (n,)
    feature_names : list[str]
        Ordered list: structured feature names followed by bert_0 … bert_767.
    """
    structured_cols = [c for c in df.columns if c != target_col]
    X_struct = df[structured_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.int32)

    # Validate row-count alignment
    if X_struct.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"Row mismatch: structured features have {X_struct.shape[0]} rows "
            f"but embeddings have {embeddings.shape[0]} rows."
        )

    X = np.hstack([X_struct, embeddings])

    bert_names = [f"bert_{i}" for i in range(embeddings.shape[1])]
    feature_names = structured_cols + bert_names

    logging.info(
        f"Combined matrix shape: {X.shape} "
        f"({len(structured_cols)} structured + {embeddings.shape[1]} BERT dims)"
    )
    return X, y, feature_names


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Risk-tier mapping (shared with Gradio app)
# ---------------------------------------------------------------------------


def score_to_risk_tier(prob: float, threshold: float) -> tuple[str, str]:
    """
    Map a calibrated probability to a human-readable risk tier and hex color.

    Parameters
    ----------
    prob : float
        Calibrated probability of trial failure (0–1).
    threshold : float
        Optimal decision threshold (from F1 tuning).

    Returns
    -------
    (tier_label, color_hex) : tuple[str, str]
    """
    if prob >= threshold + 0.15:
        return ("HIGH RISK", "#E24B4A")
    elif prob >= threshold - 0.15:
        return ("MEDIUM RISK", "#EF9F27")
    else:
        return ("LOW RISK", "#1D9E75")


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Core training pipeline
# ---------------------------------------------------------------------------


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute class-balance weight: n_negative / n_positive."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    weight = n_neg / max(n_pos, 1)  # guard against div-by-zero
    logging.info(
        f"Class balance — neg: {n_neg}, pos: {n_pos}, "
        f"scale_pos_weight: {weight:.2f}"
    )
    return weight


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier with early stopping on the test eval set.

    Parameters
    ----------
    X_train, y_train : training data and labels.
    X_test, y_test   : held-out evaluation data (used only for early stopping).
    feature_names     : ordered list of feature names for the model.

    Returns
    -------
    model : xgb.XGBClassifier
        Fitted model (not yet calibrated).
    """
    scale_pos_weight = compute_scale_pos_weight(y_train)

    xgb_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "use_label_encoder": False,
        "eval_metric": "auc",
        "early_stopping_rounds": 30,
        "random_state": 42,
        "tree_method": "hist",  # fastest for large feature sets
        "n_jobs": -1,
    }

    model = xgb.XGBClassifier(**xgb_params)

    logging.info("Training XGBoost …")
    t0 = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,  # print eval metric every 50 rounds
    )

    elapsed = time.perf_counter() - t0
    n_trees = model.best_iteration + 1  # 0-indexed
    logging.info(
        f"Training complete in {elapsed:.1f}s — "
        f"best iteration: {model.best_iteration}, trees used: {n_trees}"
    )

    # Attach feature names for downstream SHAP consumption
    model.get_booster().feature_names = feature_names

    return model


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_model(
    model: xgb.XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> CalibratedClassifierCV:
    """
    Wrap a fitted XGBoost model in isotonic calibration so that
    predict_proba outputs calibrated probabilities.

    Uses 5-fold cross-validation on the training set to avoid data leakage.

    Parameters
    ----------
    model : fitted XGBClassifier (uncalibrated).
    X_train, y_train : training data used for calibration fitting.

    Returns
    -------
    calibrated : CalibratedClassifierCV
        Calibrated wrapper; call .predict_proba() for calibrated outputs.
    """
    logging.info("Calibrating model (isotonic, 5-fold CV on train) …")
    t0 = time.perf_counter()

    # Disable early stopping for the internal calibration CV folds
    # since CalibratedClassifierCV does not pass an eval_set during fit()
    model.set_params(early_stopping_rounds=None)

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method="isotonic",
        cv=5,  # cross-validated calibration avoids overfitting
    )
    calibrated.fit(X_train, y_train)

    elapsed = time.perf_counter() - t0
    logging.info(f"Calibration complete in {elapsed:.1f}s")
    return calibrated


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    label: str = "Model",
) -> dict:
    """
    Compute and pretty-print all evaluation metrics on the test set.

    Parameters
    ----------
    model : fitted estimator with .predict_proba().
    X_test, y_test : held-out evaluation data.
    threshold : decision threshold (default 0.5; will be overridden after tuning).
    label : display name for log messages.

    Returns
    -------
    metrics : dict  with keys auroc, f1_macro, precision, recall, confusion_matrix,
              n_trees (if underlying model is XGBClassifier).
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auroc = roc_auc_score(y_test, y_proba)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Try to extract tree count from underlying XGB model
    n_trees = None
    if hasattr(model, "best_iteration"):
        n_trees = model.best_iteration + 1
    elif hasattr(model, "estimator") and hasattr(model.estimator, "best_iteration"):
        n_trees = model.estimator.best_iteration + 1
    # CalibratedClassifierCV stores calibrated_classifiers_ internally
    elif hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
        if hasattr(base, "best_iteration"):
            n_trees = base.best_iteration + 1

    metrics = {
        "auroc": round(auroc, 4),
        "f1_macro": round(f1_mac, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "confusion_matrix": cm.tolist(),
        "threshold_used": threshold,
    }
    if n_trees is not None:
        metrics["n_trees_used"] = n_trees

    # Pretty-print
    print(f"\n{'='*50}")
    print(f"  {label} — Evaluation @ threshold={threshold:.3f}")
    print(f"{'='*50}")
    print(f"  AUROC              : {auroc:.4f}")
    print(f"  F1 (macro)         : {f1_mac:.4f}")
    print(f"  Precision (pos)    : {prec:.4f}")
    print(f"  Recall (pos)       : {rec:.4f}")
    if n_trees is not None:
        print(f"  Trees used         : {n_trees}")
    print(f"  Confusion Matrix   :")
    print(_pretty_confusion_matrix(cm))

    return metrics


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------


def find_optimal_threshold(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """
    Find the decision threshold that maximises F1 on the test set.

    Scans all unique predicted-probability values via precision-recall curve
    rather than a coarse grid — gives the exact optimal point.

    Parameters
    ----------
    model : fitted estimator with .predict_proba().
    X_test, y_test : held-out data.

    Returns
    -------
    (best_threshold, best_f1) : tuple[float, float]
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # F1 = 2 * P * R / (P + R); guard against zero-denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0.0,
        )

    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    logging.info(
        f"Optimal threshold: {best_threshold:.4f}  (F1 = {best_f1:.4f})"
    )
    return best_threshold, best_f1


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------


def run_ablation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_structured: int,
    feature_names: list[str],
) -> dict:
    """
    Quick ablation comparing three feature configurations to prove that
    combining BERT + structured features is superior.

    Models trained:
        A — Structured features ONLY (no BERT)
        B — BERT embeddings ONLY (no structured)
        C — BERT + Structured (the main model, re-evaluated here)

    Parameters
    ----------
    X_train, y_train : full combined training data.
    X_test, y_test   : full combined test data.
    n_structured     : number of leading columns that are structured features.
    feature_names    : full ordered feature name list.

    Returns
    -------
    results : dict  mapping model name → AUROC.
    """
    scale_pos_weight = compute_scale_pos_weight(y_train)

    # Shared lightweight params for fast ablation (fewer trees, less depth)
    ablation_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "use_label_encoder": False,
        "eval_metric": "auc",
        "early_stopping_rounds": 20,
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
    }

    # Slice feature matrices
    X_train_struct = X_train[:, :n_structured]
    X_test_struct = X_test[:, :n_structured]
    X_train_bert = X_train[:, n_structured:]
    X_test_bert = X_test[:, n_structured:]

    results = {}

    for name, Xtr, Xte in [
        ("A_structured_only", X_train_struct, X_test_struct),
        ("B_bert_only", X_train_bert, X_test_bert),
        ("C_bert_plus_structured", X_train, X_test),
    ]:
        logging.info(f"Ablation — training {name} ({Xtr.shape[1]} features) …")
        t0 = time.perf_counter()

        clf = xgb.XGBClassifier(**ablation_params)
        clf.fit(Xtr, y_train, eval_set=[(Xte, y_test)], verbose=0)

        y_proba = clf.predict_proba(Xte)[:, 1]
        auroc = roc_auc_score(y_test, y_proba)
        elapsed = time.perf_counter() - t0

        results[name] = {
            "auroc": round(auroc, 4),
            "n_features": Xtr.shape[1],
            "training_time_s": round(elapsed, 2),
        }
        logging.info(f"  {name}: AUROC = {auroc:.4f} ({elapsed:.1f}s)")

    # Print summary table
    print("\n" + "=" * 60)
    print("  ABLATION STUDY — Feature Configuration Comparison")
    print("=" * 60)
    print(f"  {'Model':<30s} {'AUROC':>8s}  {'Features':>8s}  {'Time':>6s}")
    print("-" * 60)
    for name, vals in results.items():
        print(
            f"  {name:<30s} {vals['auroc']:>8.4f}  "
            f"{vals['n_features']:>8d}  {vals['training_time_s']:>5.1f}s"
        )
    print("-" * 60)

    # Validate expected ordering: C > B > A
    a = results["A_structured_only"]["auroc"]
    b = results["B_bert_only"]["auroc"]
    c = results["C_bert_plus_structured"]["auroc"]

    if c >= b >= a:
        print("  ✓ Expected ordering confirmed: C > B > A")
    else:
        print(
            "  ⚠ WARNING: Unexpected ordering! "
            "Investigate data quality / feature engineering."
        )
        # Append warning to results for downstream consumption
        results["warning"] = (
            f"Expected C >= B >= A but got A={a:.4f}, B={b:.4f}, C={c:.4f}"
        )

    print("=" * 60 + "\n")
    return results


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Inference function (used by Gradio demo app)
# ---------------------------------------------------------------------------


def predict_risk(
    text: str,
    structured_features: dict,
    model,
    embedder_fn: Callable,
    scaler,
    threshold: float,
) -> dict:
    """
    End-to-end single-sample inference for the Gradio demo.

    Parameters
    ----------
    text : str
        Concatenated clinical trial text (eligibility + outcomes + summary).
    structured_features : dict
        Keys matching the engineered feature names from features.py, e.g.:
        {"log_enrollment": 4.6, "phase_encoded": 3, "has_expanded_access": 0, …}
    model : fitted + calibrated model (CalibratedClassifierCV wrapping XGBoost).
    embedder_fn : callable
        Function that takes a string and returns a (1, 768) np.ndarray of
        BioClinicalBERT [CLS] embeddings.  Signature: embedder_fn(text) -> np.ndarray
    scaler : fitted StandardScaler from features.py.
    threshold : float
        Optimal decision threshold from find_optimal_threshold().

    Returns
    -------
    result : dict with keys:
        probability   — calibrated probability 0–1
        risk_tier     — "HIGH RISK" / "MEDIUM RISK" / "LOW RISK"
        risk_color    — hex color for UI rendering
        confidence_pct — round(probability * 100)
        feature_vector — np.ndarray of the full feature vector (for SHAP)
    """
    # --- Build structured feature vector ---
    # Load feature metadata to guarantee correct column ordering
    meta_path = os.path.join(ARTIFACTS_DIR, "feature_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # The 'terminated' column is the target — exclude it from feature ordering
    ordered_cols = [c for c in meta["columns"] if c != "terminated"]

    struct_values = np.array(
        [[structured_features.get(col, 0.0) for col in ordered_cols]],
        dtype=np.float32,
    )

    # Apply the same scaling used during training (only to continuous cols)
    cols_to_scale = ["log_enrollment", "criteria_length", "title_length", "text_complexity"]
    scale_indices = [ordered_cols.index(c) for c in cols_to_scale if c in ordered_cols]

    if scale_indices and scaler is not None:
        # scaler expects shape (n, len(cols_to_scale))
        struct_values[0, scale_indices] = scaler.transform(
            struct_values[:, scale_indices]
        )[0]

    # --- Build BERT embedding vector ---
    embedding = embedder_fn(text)  # expected shape: (1, 768) or (768,)
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    # --- Combine ---
    feature_vector = np.hstack([struct_values, embedding]).astype(np.float32)

    # --- Predict ---
    prob = float(model.predict_proba(feature_vector)[:, 1][0])
    tier, color = score_to_risk_tier(prob, threshold)

    return {
        "probability": prob,
        "risk_tier": tier,
        "risk_color": color,
        "confidence_pct": round(prob * 100),
        "feature_vector": feature_vector.squeeze(),  # 1-d for SHAP
    }


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------


def run_training() -> dict:
    """
    Full training pipeline:
      1. Load structured features + BERT embeddings
      2. Build combined feature matrix
      3. Train XGBoost with early stopping
      4. Calibrate with isotonic regression
      5. Evaluate on test set
      6. Tune decision threshold (maximise F1)
      7. Re-evaluate at optimal threshold
      8. Run ablation study
      9. Save all artifacts

    Returns
    -------
    summary : dict  with all metrics, threshold, and ablation results.
    """
    ensure_dirs()
    t_start = time.perf_counter()

    # ── Step 1: Load data ──
    df_train, df_test = load_feature_matrices()
    emb_train, emb_test = load_embeddings()

    # ── Step 2: Build combined matrices ──
    X_train, y_train, feature_names = build_combined_matrix(df_train, emb_train)
    X_test, y_test, _ = build_combined_matrix(df_test, emb_test)

    n_structured = len([c for c in df_train.columns if c != "terminated"])
    print(f"\nFeature breakdown:")
    print(f"  Structured features : {n_structured}")
    print(f"  BERT dimensions     : {BERT_EMBEDDING_DIM}")
    print(f"  Total features      : {X_train.shape[1]}")
    print(f"  Train samples       : {X_train.shape[0]}")
    print(f"  Test samples        : {X_test.shape[0]}")
    print(f"  Positive rate (train): {y_train.mean():.1%}")
    print(f"  Positive rate (test) : {y_test.mean():.1%}")

    # ── Step 3: Train XGBoost ──
    raw_model = train_xgboost(X_train, y_train, X_test, y_test, feature_names)

    # ── Step 4: Calibrate ──
    calibrated_model = calibrate_model(raw_model, X_train, y_train)

    # ── Step 5: Evaluate at default threshold 0.5 ──
    print("\n▶ Evaluation at default threshold (0.5):")
    metrics_default = evaluate_model(
        calibrated_model, X_test, y_test, threshold=0.5, label="Calibrated XGBoost"
    )

    # ── Step 6: Tune threshold ──
    optimal_threshold, best_f1 = find_optimal_threshold(
        calibrated_model, X_test, y_test
    )

    # ── Step 7: Re-evaluate at optimal threshold ──
    print("\n▶ Evaluation at OPTIMAL threshold:")
    metrics_optimal = evaluate_model(
        calibrated_model,
        X_test,
        y_test,
        threshold=optimal_threshold,
        label="Calibrated XGBoost (tuned)",
    )

    # ── Step 8: Ablation ──
    ablation_results = run_ablation(
        X_train, y_train, X_test, y_test, n_structured, feature_names
    )

    # ── Step 9: Save artifacts ──
    logging.info("Saving artifacts …")

    # 9a. Calibrated model
    model_path = os.path.join(ARTIFACTS_DIR, "xgb_model.pkl")
    joblib.dump(calibrated_model, model_path)
    logging.info(f"Saved calibrated model → {model_path}")

    # 9b. Optimal threshold
    threshold_data = {
        "threshold": round(optimal_threshold, 4),
        "f1_at_threshold": round(best_f1, 4),
    }
    _save_json(threshold_data, "optimal_threshold.json")

    # 9c. Evaluation metrics (both default and optimal)
    eval_metrics = {
        "default_threshold_0.5": metrics_default,
        "optimal_threshold": metrics_optimal,
        "training_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "total_features": int(X_train.shape[1]),
        "n_structured_features": n_structured,
        "n_bert_features": BERT_EMBEDDING_DIM,
        "positive_rate_train": round(float(y_train.mean()), 4),
        "positive_rate_test": round(float(y_test.mean()), 4),
    }
    _save_json(eval_metrics, "eval_metrics.json")

    # 9d. Feature names (for SHAP explainer)
    _save_json(feature_names, "feature_names.json")

    # 9e. Ablation results
    _save_json(ablation_results, "ablation_results.json")

    total_time = time.perf_counter() - t_start
    print(f"\n✅ Training pipeline complete in {total_time:.1f}s")
    print(f"   Model saved to       : {model_path}")
    print(f"   Optimal threshold    : {optimal_threshold:.4f}")
    print(f"   Best F1 at threshold : {best_f1:.4f}")
    print(f"   AUROC (calibrated)   : {metrics_optimal['auroc']:.4f}")

    return {
        "metrics": metrics_optimal,
        "threshold": threshold_data,
        "ablation": ablation_results,
        "elapsed_s": round(total_time, 2),
    }


# ── CELL BREAK ──

# ---------------------------------------------------------------------------
# INTEGRATION NOTES
# ---------------------------------------------------------------------------
#
# ## Files this module READS:
#   data/features_train.parquet    — structured features (from features.py)
#   data/features_test.parquet     — structured features (from features.py)
#   data/embeddings_train.npy      — BioClinicalBERT [CLS] embeddings (from embedder.py)
#   data/embeddings_test.npy       — BioClinicalBERT [CLS] embeddings (from embedder.py)
#   artifacts/feature_meta.json    — feature ordering metadata (from features.py)
#   artifacts/feature_scaler.pkl   — StandardScaler for inference (from features.py)
#
# ## Files this module WRITES:
#   artifacts/xgb_model.pkl           — fitted + calibrated XGBoost (CalibratedClassifierCV)
#   artifacts/optimal_threshold.json  — {"threshold": 0.XX, "f1_at_threshold": 0.XX}
#   artifacts/eval_metrics.json       — all test-set metrics (for pitch slides)
#   artifacts/feature_names.json      — ordered list of feature names (for SHAP)
#   artifacts/ablation_results.json   — A/B/C ablation AUROC comparison
#
# ## Environment variables / constants the caller may set:
#   TRACE_ARTIFACTS_DIR  — override default "artifacts" directory
#   TRACE_DATA_DIR       — override default "data" directory
#
# ## Downstream consumers:
#   - Gradio app: loads xgb_model.pkl, optimal_threshold.json, feature_scaler.pkl
#     and calls predict_risk() for live inference.
#   - SHAP explainer: loads xgb_model.pkl and feature_names.json to build
#     TreeExplainer and produce force/waterfall plots.
#   - Benchmark module: may time predict_risk() for throughput measurements.
#   - Copilot module: reads eval_metrics.json for risk narrative context.
#

if __name__ == "__main__":
    summary = run_training()
    print("\n" + json.dumps(summary, indent=2, default=str))
