"""
AMD MI300X Benchmarking Module for Clinical Trial Risk Prediction
=================================================================
Measures and visualises the performance advantage of AMD MI300X over CPU
for the clinical trial scoring workload.

Designed for notebook-first execution on AMD Developer Cloud.

Dependencies:
    torch (ROCm build), transformers, xgboost, shap, pandas, matplotlib, joblib
"""
import os
import gc
import json
import time
import warnings
from datetime import datetime, timezone
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works without display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Optional imports — these are only needed when model artifacts exist
# and are imported lazily inside the functions that use them.
# xgboost, shap, joblib, requests are imported at call-site.

# ─── Constants ───────────────────────────────────────────────────────────────
BERT_MODEL_NAME: str = "emilyalsentzer/Bio_ClinicalBERT"
MAX_SEQ_LEN: int = 512
AMD_BRAND_RED: str = "#ED1C24"          # AMD corporate red
AMD_ACCENT: str = "#E05C2C"            # AMD accent / Instinct product line
CPU_BAR_COLOR: str = "#9E9E9E"         # Neutral gray for CPU bars
CHART_BG: str = "#FAFAFA"
GRID_COLOR: str = "#E0E0E0"

# Default artifact paths (align with pipeline.py / features.py conventions)
DATA_DIR: str = "data"
ARTIFACTS_DIR: str = "artifacts"
DEMO_DIR: str = "demo"
CHART_PATH: str = os.path.join(DEMO_DIR, "amd_benchmark.png")
SUMMARY_PATH: str = os.path.join(ARTIFACTS_DIR, "benchmark_summary.json")

# CPU baseline measured on AMD MI300X host CPU (AMD EPYC).
# Real-world CPU server (e.g. AWS c5.4xlarge) would be similar or slower.
# Speedup is conservative — not cherry-picked hardware.

# ── CELL BREAK ──

# ─── BENCHMARK 1: BERT Embedding Throughput ──────────────────────────────────

def _generate_sample_texts(n: int = 200) -> list[str]:
    """
    Generate deterministic sample clinical-trial texts for benchmarking.

    Uses a fixed seed so every run produces the same texts, ensuring
    fair comparison across devices and batch sizes.
    """
    # Representative clinical trial text fragments — chosen to exercise
    # the tokenizer across typical vocabulary (drug names, eligibility
    # criteria patterns, outcome descriptions).
    templates: list[str] = [
        (
            "Patients with confirmed diagnosis of {cond} are eligible. "
            "Exclusion criteria include prior treatment with {drug}, "
            "ECOG performance status > 2, and uncontrolled hypertension. "
            "Primary outcome: overall survival at 12 months."
        ),
        (
            "A randomized, double-blind, placebo-controlled Phase {phase} study "
            "to evaluate the efficacy and safety of {drug} in adults with {cond}. "
            "Enrollment target: {enroll} participants across {sites} sites."
        ),
        (
            "Inclusion criteria: Age >= 18 years, histologically confirmed {cond}, "
            "adequate organ function as defined by laboratory parameters. "
            "Participants must have measurable disease per RECIST v1.1."
        ),
        (
            "This multi-center, open-label study assesses {drug} as adjuvant "
            "therapy following surgical resection of {cond}. Secondary endpoints "
            "include disease-free survival, quality of life (EQ-5D-5L), "
            "and pharmacokinetic profile at steady state."
        ),
        (
            "Eligibility requires documented {cond} for at least 6 months, "
            "HbA1c between 7.0% and 10.0%, BMI 25-40 kg/m2. "
            "Excluded: pregnancy, severe renal impairment (eGFR < 30), "
            "or current participation in another interventional trial."
        ),
    ]
    conditions = [
        "non-small cell lung cancer", "type 2 diabetes mellitus",
        "chronic heart failure", "major depressive disorder",
        "metastatic breast cancer", "rheumatoid arthritis",
        "Alzheimer's disease", "chronic obstructive pulmonary disease",
    ]
    drugs = [
        "pembrolizumab", "metformin", "sacubitril-valsartan",
        "escitalopram", "trastuzumab", "adalimumab",
        "aducanumab", "roflumilast",
    ]
    rng = np.random.RandomState(42)
    texts: list[str] = []
    for i in range(n):
        tmpl = templates[i % len(templates)]
        text = tmpl.format(
            cond=conditions[i % len(conditions)],
            drug=drugs[i % len(drugs)],
            phase=rng.choice([1, 2, 3]),
            enroll=rng.randint(50, 2000),
            sites=rng.randint(5, 120),
        )
        texts.append(text)
    return texts


def _embed_batch(
    texts: list[str],
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Embed a list of texts using BioClinicalBERT in batches.

    Returns CLS-token embeddings as a numpy array of shape (n, 768).
    """
    model.eval()
    all_embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt",
            )
            # Move every tensor in the encoding to the target device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            # CLS token embedding: first token of last_hidden_state
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)
    return np.vstack(all_embeddings)


def benchmark_bert_embedding(
    texts: list[str] | None = None,
    batch_sizes: list[int] | None = None,
    devices: list[str] | None = None,
    n_warmup: int = 2,
) -> pd.DataFrame:
    """
    Benchmark BioClinicalBERT embedding throughput across devices and batch sizes.

    For each (device, batch_size) combination:
      - Loads BioClinicalBERT onto that device
      - Runs a warm-up pass (excluded from timing)
      - Times the full embedding pass over `texts`
      - Computes records/second, ms/record, total time

    Args:
        texts:       List of clinical trial texts. Defaults to 200 generated samples.
        batch_sizes: Batch sizes to benchmark. Defaults to [8, 16, 32, 64].
        devices:     Device strings. Defaults to ["cpu", "cuda"].
        n_warmup:    Number of warm-up iterations before timing.

    Returns:
        DataFrame with columns: device, batch_size, total_seconds,
        records_per_sec, ms_per_record
    """
    if texts is None:
        texts = _generate_sample_texts(200)
    if batch_sizes is None:
        batch_sizes = [8, 16, 32, 64]
    if devices is None:
        devices = ["cpu", "cuda"]

    n_texts = len(texts)
    results: list[dict[str, Any]] = []

    for device_str in devices:
        device = torch.device(device_str)

        # Check GPU availability — skip silently if not present
        if device_str == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA/ROCm not available — skipping GPU benchmark.")
            continue

        print(f"\n{'='*50}")
        print(f"Loading BioClinicalBERT on: {device_str.upper()}")
        print(f"{'='*50}")

        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(BERT_MODEL_NAME, use_safetensors=True).to(device)

        for bs in batch_sizes:
            print(f"  batch_size={bs:>3d} ... ", end="", flush=True)

            # Warm-up: run a few iterations to stabilise GPU clocks / JIT
            for _ in range(n_warmup):
                _embed_batch(texts[:bs], tokenizer, model, device, bs)

            # Synchronise GPU before timing
            if device_str == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _embed_batch(texts, tokenizer, model, device, bs)

            if device_str == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            rps = n_texts / elapsed
            ms_per = (elapsed / n_texts) * 1000
            print(f"{rps:>8.1f} rec/s  ({elapsed:.2f}s total)")

            results.append({
                "device": device_str,
                "batch_size": bs,
                "total_seconds": round(elapsed, 4),
                "records_per_sec": round(rps, 2),
                "ms_per_record": round(ms_per, 2),
            })

        # Free VRAM between device runs
        del model
        gc.collect()
        if device_str == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    # Save intermediate results immediately — sessions may not persist
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(ARTIFACTS_DIR, "bert_benchmark.csv"), index=False)
    print(f"\nBERT benchmark saved to {ARTIFACTS_DIR}/bert_benchmark.csv")
    return df


if __name__ == "__main__":
    bert_df = benchmark_bert_embedding()
    print(bert_df.to_string(index=False))

# ── CELL BREAK ──

# ─── BENCHMARK 2: Full Pipeline Throughput ───────────────────────────────────

def benchmark_full_pipeline(
    n_trials: int = 100,
    device: str = "cuda",
) -> dict[str, Any]:
    """
    Time the end-to-end scoring flow for n_trials clinical trial protocols.

    Steps timed independently:
      1. BERT embedding     (on `device` — GPU accelerated)
      2. XGBoost prediction (CPU — XGBoost CPU is standard)
      3. SHAP explanation   (CPU — TreeExplainer is CPU-only)

    Also runs the same pipeline on CPU for comparison.

    Args:
        n_trials: Number of synthetic trial protocols to score.
        device:   Device for BERT embedding ("cuda" or "cpu").

    Returns:
        Dict with per-step timings for both GPU and CPU runs,
        plus speedup ratios.
    """
    import xgboost as xgb
    import shap
    import joblib

    texts = _generate_sample_texts(n_trials)
    results: dict[str, Any] = {"n_trials": n_trials}

    # Determine which devices to benchmark
    run_devices = ["cpu"]
    if device == "cuda" and torch.cuda.is_available():
        run_devices.append("cuda")
    elif device == "cuda":
        warnings.warn("CUDA/ROCm not available — running CPU-only pipeline benchmark.")

    for dev_str in run_devices:
        dev = torch.device(dev_str)
        prefix = "gpu" if dev_str == "cuda" else "cpu"

        print(f"\n--- Full pipeline on {dev_str.upper()} ---")

        # ── Step 1: BERT embedding ──
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(BERT_MODEL_NAME, use_safetensors=True).to(dev)

        if dev_str == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        embeddings = _embed_batch(texts, tokenizer, model, dev, batch_size=32)
        if dev_str == "cuda":
            torch.cuda.synchronize()
        t_embed = time.perf_counter() - t1
        results[f"{prefix}_embed_sec"] = round(t_embed, 4)
        print(f"  Step 1 (BERT embed):  {t_embed:.3f}s")

        del model
        gc.collect()
        if dev_str == "cuda":
            torch.cuda.empty_cache()

        # ── Step 2: XGBoost prediction (always CPU) ──
        # Build a synthetic feature matrix that matches the feature schema
        # from features.py: 12 structured features + 768 BERT dims = 780 cols.
        rng = np.random.RandomState(42)
        n_structured = 12  # matches features.py engineered feature count
        structured = rng.rand(n_trials, n_structured).astype(np.float32)
        feature_matrix = np.hstack([structured, embeddings.astype(np.float32)])

        # Try to load the trained model; fall back to a dummy model
        xgb_model_path = os.path.join(ARTIFACTS_DIR, "xgb_model.json")
        if os.path.exists(xgb_model_path):
            bst = xgb.Booster()
            bst.load_model(xgb_model_path)
        else:
            # Synthetic XGBoost model for benchmarking latency
            # (actual accuracy is irrelevant — we're timing throughput)
            print("  [INFO] No trained XGBoost model found; using synthetic model.")
            y_dummy = rng.randint(0, 2, size=n_trials)
            dtrain = xgb.DMatrix(feature_matrix, label=y_dummy)
            bst = xgb.train(
                {"max_depth": 6, "eta": 0.1, "objective": "binary:logistic",
                 "eval_metric": "auc", "nthread": -1},
                dtrain, num_boost_round=100, verbose_eval=False,
            )

        dmat = xgb.DMatrix(feature_matrix)
        t2 = time.perf_counter()
        preds = bst.predict(dmat)
        t_predict = time.perf_counter() - t2
        results[f"{prefix}_predict_sec"] = round(t_predict, 4)
        print(f"  Step 2 (XGB predict): {t_predict:.3f}s")

        # ── Step 3: SHAP explanation (TreeExplainer, CPU-bound) ──
        t3 = time.perf_counter()
        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(dmat)
        t_shap = time.perf_counter() - t3
        results[f"{prefix}_shap_sec"] = round(t_shap, 4)
        print(f"  Step 3 (SHAP):        {t_shap:.3f}s")

        total = t_embed + t_predict + t_shap
        results[f"{prefix}_total_sec"] = round(total, 4)
        print(f"  TOTAL:                {total:.3f}s")

    # Compute speedup if both devices were benchmarked
    if "gpu_total_sec" in results and "cpu_total_sec" in results:
        speedup = results["cpu_total_sec"] / results["gpu_total_sec"]
        results["speedup_vs_cpu"] = round(speedup, 1)
        print(f"\n  AMD MI300X speedup: {speedup:.1f}× faster than CPU")
    else:
        results["speedup_vs_cpu"] = None

    # Save immediately
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACTS_DIR, "pipeline_benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Pipeline benchmark saved to {ARTIFACTS_DIR}/pipeline_benchmark.json")

    return results


if __name__ == "__main__":
    pipe_results = benchmark_full_pipeline(n_trials=100, device="cuda")
    print(json.dumps(pipe_results, indent=2))

# ── CELL BREAK ──

# ─── BENCHMARK 3: vLLM Throughput ────────────────────────────────────────────

def benchmark_vllm(
    n_prompts: int = 20,
    port: int = 8000,
) -> dict[str, Any]:
    """
    Benchmark vLLM inference throughput for clinical trial explanation prompts.

    Sends `n_prompts` requests to a running vLLM OpenAI-compatible server
    and measures latency statistics.

    Prerequisites:
        A vLLM server must be running on localhost:{port} before calling this.
        Start with:
            python -m vllm.entrypoints.openai.api_server \\
                --model mistralai/Mistral-7B-Instruct-v0.3 \\
                --port 8000 --dtype float16

    Args:
        n_prompts: Number of explanation prompts to send.
        port:      Port of the vLLM server.

    Returns:
        Dict with prompts_per_sec, avg_latency_ms, p95_latency_ms, total_seconds.
    """
    import requests as req  # local alias to avoid clash with top-level

    url = f"http://localhost:{port}/v1/chat/completions"

    # Representative clinical-trial explanation prompts
    base_prompt = (
        "You are a clinical trial risk analyst. Explain in 3 bullet points "
        "why the following trial has a {risk_level} risk of early termination:\n\n"
        "Trial: {trial_desc}\n\n"
        "Provide actionable recommendations to reduce risk."
    )
    risk_levels = ["high", "moderate", "low"]
    trial_descs = [
        "Phase 2 oncology trial with 45 patients and 6 eligibility criteria",
        "Phase 3 diabetes trial with 800 patients across 40 sites",
        "Phase 1 rare disease trial with 12 patients, single-arm, open-label",
        "Phase 2/3 cardiovascular trial with complex composite primary endpoint",
        "Phase 4 post-marketing surveillance study with 2000 enrolled",
    ]

    rng = np.random.RandomState(42)
    prompts = [
        base_prompt.format(
            risk_level=risk_levels[i % len(risk_levels)],
            trial_desc=trial_descs[i % len(trial_descs)],
        )
        for i in range(n_prompts)
    ]

    # Check server availability and get model ID before running the full benchmark
    try:
        health = req.get(f"http://localhost:{port}/health", timeout=5)
        health.raise_for_status()
        
        models_resp = req.get(f"http://localhost:{port}/v1/models", timeout=5)
        models_resp.raise_for_status()
        model_id = models_resp.json()["data"][0]["id"]
    except Exception as e:
        warnings.warn(
            f"vLLM server not reachable on port {port} or failed to fetch models. "
            f"Error: {e}. Skipping LLM benchmark."
        )
        return {
            "status": "skipped",
            "reason": f"vLLM server not reachable on localhost:{port}",
        }

    latencies: list[float] = []
    print(f"Sending {n_prompts} prompts to vLLM on port {port} using model {model_id}...")

    for i, prompt in enumerate(prompts):
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.7,
        }
        t0 = time.perf_counter()
        try:
            resp = req.post(url, json=payload, timeout=120)
            resp.raise_for_status()
        except Exception as e:
            warnings.warn(f"Prompt {i} failed: {e}")
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n_prompts} done — last latency: {elapsed_ms:.0f}ms")

    if not latencies:
        return {"status": "failed", "reason": "All prompts failed."}

    total_sec = sum(latencies) / 1000
    results = {
        "status": "completed",
        "n_prompts": len(latencies),
        "prompts_per_sec": round(len(latencies) / total_sec, 2),
        "avg_latency_ms": round(float(np.mean(latencies)), 1),
        "p50_latency_ms": round(float(np.median(latencies)), 1),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1),
        "total_seconds": round(total_sec, 2),
    }

    # Save immediately
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACTS_DIR, "vllm_benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"vLLM benchmark saved to {ARTIFACTS_DIR}/vllm_benchmark.json")
    print(f"  Throughput: {results['prompts_per_sec']} prompts/sec")
    print(f"  Avg latency: {results['avg_latency_ms']}ms | P95: {results['p95_latency_ms']}ms")

    return results


if __name__ == "__main__":
    vllm_results = benchmark_vllm(n_prompts=20, port=8000)
    print(json.dumps(vllm_results, indent=2))

# ── CELL BREAK ──

# ─── CHART GENERATION ────────────────────────────────────────────────────────

def plot_benchmark_results(
    bert_results: pd.DataFrame,
    pipeline_results: dict[str, Any],
    output_path: str = CHART_PATH,
) -> str:
    """
    Create a 2-panel publication-quality benchmark chart.

    Panel 1 (left):  Bar chart — BERT embedding throughput by batch size
    Panel 2 (right): Horizontal bar chart — End-to-end pipeline step times

    Style: clean matplotlib, AMD brand colours, no seaborn dependency.
    Saves PNG at 150 DPI. Returns the output path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=CHART_BG)
    fig.subplots_adjust(wspace=0.35, left=0.07, right=0.95, top=0.88, bottom=0.15)

    # ── Panel 1: BERT Embedding Throughput ────────────────────────────────
    cpu_data = bert_results[bert_results["device"] == "cpu"].sort_values("batch_size")
    gpu_data = bert_results[bert_results["device"] == "cuda"].sort_values("batch_size")

    # Use the batch sizes that appear in both sets (handles missing GPU gracefully)
    batch_sizes = sorted(
        set(cpu_data["batch_size"].tolist()) & set(gpu_data["batch_size"].tolist())
    ) if not gpu_data.empty else cpu_data["batch_size"].tolist()

    x = np.arange(len(batch_sizes))
    bar_width = 0.35

    cpu_rps = cpu_data.set_index("batch_size").loc[batch_sizes, "records_per_sec"].values
    gpu_rps = (
        gpu_data.set_index("batch_size").loc[batch_sizes, "records_per_sec"].values
        if not gpu_data.empty
        else np.zeros(len(batch_sizes))
    )

    ax1.set_facecolor(CHART_BG)
    bars_cpu = ax1.bar(
        x - bar_width / 2, cpu_rps, bar_width,
        label="CPU (AMD EPYC)", color=CPU_BAR_COLOR, edgecolor="white", linewidth=0.8,
    )
    bars_gpu = ax1.bar(
        x + bar_width / 2, gpu_rps, bar_width,
        label="AMD MI300X", color=AMD_ACCENT, edgecolor="white", linewidth=0.8,
    )

    # Speedup annotations on top of each AMD bar
    for i, (c, g) in enumerate(zip(cpu_rps, gpu_rps)):
        if c > 0 and g > 0:
            speedup = g / c
            ax1.annotate(
                f"{speedup:.1f}×",
                xy=(x[i] + bar_width / 2, g),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=AMD_BRAND_RED,
            )

    ax1.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Records / Second", fontsize=12, fontweight="bold")
    ax1.set_title(
        "AMD MI300X vs CPU — BioClinicalBERT Throughput",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(b) for b in batch_sizes], fontsize=11)
    ax1.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax1.grid(axis="y", color=GRID_COLOR, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel 2: End-to-End Pipeline ──────────────────────────────────────
    step_labels = ["BERT Embedding", "XGBoost Predict", "SHAP Explain", "TOTAL"]
    # Gather timings — fall back to 0 if a key is missing
    cpu_times = [
        pipeline_results.get("cpu_embed_sec", 0),
        pipeline_results.get("cpu_predict_sec", 0),
        pipeline_results.get("cpu_shap_sec", 0),
        pipeline_results.get("cpu_total_sec", 0),
    ]
    gpu_times = [
        pipeline_results.get("gpu_embed_sec", 0),
        pipeline_results.get("gpu_predict_sec", 0),
        pipeline_results.get("gpu_shap_sec", 0),
        pipeline_results.get("gpu_total_sec", 0),
    ]

    y = np.arange(len(step_labels))
    bar_height = 0.35

    ax2.set_facecolor(CHART_BG)
    ax2.barh(
        y + bar_height / 2, cpu_times, bar_height,
        label="CPU (AMD EPYC)", color=CPU_BAR_COLOR, edgecolor="white", linewidth=0.8,
    )
    ax2.barh(
        y - bar_height / 2, gpu_times, bar_height,
        label="AMD MI300X", color=AMD_ACCENT, edgecolor="white", linewidth=0.8,
    )

    # Time labels at the end of each bar
    for i, (ct, gt) in enumerate(zip(cpu_times, gpu_times)):
        max_val = max(ct, gt, 0.01)
        ax2.text(ct + max_val * 0.02, y[i] + bar_height / 2, f"{ct:.2f}s",
                 va="center", fontsize=9, color="#444444")
        ax2.text(gt + max_val * 0.02, y[i] - bar_height / 2, f"{gt:.2f}s",
                 va="center", fontsize=9, color=AMD_BRAND_RED, fontweight="bold")

    # Bold the TOTAL row label
    ax2.set_yticks(y)
    tick_labels = ax2.set_yticklabels(step_labels, fontsize=11)
    tick_labels[-1].set_fontweight("bold")

    ax2.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    n_trials = pipeline_results.get("n_trials", 100)
    ax2.set_title(
        f"AMD MI300X vs CPU — Full Scoring Pipeline ({n_trials} protocols)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax2.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax2.grid(axis="x", color=GRID_COLOR, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.invert_yaxis()  # TOTAL at the bottom
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Watermark ─────────────────────────────────────────────────────────
    fig.text(
        0.95, 0.02,
        "Powered by AMD MI300X",
        fontsize=10, color=AMD_BRAND_RED, fontweight="bold",
        ha="right", va="bottom", alpha=0.7,
        fontstyle="italic",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=CHART_BG)
    plt.close(fig)
    print(f"Chart saved to: {output_path}")
    return output_path


# ─── BENCHMARK SUMMARY JSON ─────────────────────────────────────────────────

def save_benchmark_summary(
    bert_results: pd.DataFrame,
    pipeline_results: dict[str, Any],
    output_path: str = SUMMARY_PATH,
) -> dict[str, Any]:
    """
    Save a pitch-deck-ready JSON summary of all benchmark results.

    The "headline" field is designed to be copy-pasted directly into slides.

    Returns the summary dict for convenience.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Best BERT throughput per device (largest batch size, highest rps)
    best_gpu = (
        bert_results[bert_results["device"] == "cuda"]["records_per_sec"].max()
        if "cuda" in bert_results["device"].values
        else 0
    )
    best_cpu = (
        bert_results[bert_results["device"] == "cpu"]["records_per_sec"].max()
        if "cpu" in bert_results["device"].values
        else 0
    )

    gpu_total = pipeline_results.get("gpu_total_sec", 0)
    cpu_total = pipeline_results.get("cpu_total_sec", 0)
    n_trials = pipeline_results.get("n_trials", 100)
    speedup = pipeline_results.get("speedup_vs_cpu") or (
        round(cpu_total / gpu_total, 1) if gpu_total > 0 else 0
    )

    # Detect GPU model name via torch if available
    gpu_name = "AMD Instinct MI300X"
    gpu_mem_gb = 192  # MI300X spec: 192 GB HBM3
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
        except Exception:
            pass

    headline = (
        f"{n_trials} clinical trials scored in {gpu_total:.1f}s on AMD MI300X"
        if gpu_total > 0
        else f"{n_trials} clinical trials scored (CPU-only: {cpu_total:.1f}s)"
    )

    summary: dict[str, Any] = {
        "headline": headline,
        "speedup_vs_cpu": speedup,
        "bert_records_per_sec_amd": round(best_gpu, 1),
        "bert_records_per_sec_cpu": round(best_cpu, 1),
        "full_pipeline_amd_seconds": gpu_total,
        "full_pipeline_cpu_seconds": cpu_total,
        "pipeline_steps": {
            "embed": {
                "cpu_sec": pipeline_results.get("cpu_embed_sec", 0),
                "gpu_sec": pipeline_results.get("gpu_embed_sec", 0),
            },
            "predict": {
                "cpu_sec": pipeline_results.get("cpu_predict_sec", 0),
                "gpu_sec": pipeline_results.get("gpu_predict_sec", 0),
            },
            "shap": {
                "cpu_sec": pipeline_results.get("cpu_shap_sec", 0),
                "gpu_sec": pipeline_results.get("gpu_shap_sec", 0),
            },
        },
        "gpu_model": gpu_name,
        "gpu_memory_gb": gpu_mem_gb,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {output_path}")
    return summary


if __name__ == "__main__":
    # Demo: generate summary from saved intermediate files if they exist
    bert_csv = os.path.join(ARTIFACTS_DIR, "bert_benchmark.csv")
    pipe_json = os.path.join(ARTIFACTS_DIR, "pipeline_benchmark.json")

    if os.path.exists(bert_csv) and os.path.exists(pipe_json):
        bert_df = pd.read_csv(bert_csv)
        with open(pipe_json) as f:
            pipe = json.load(f)
        chart = plot_benchmark_results(bert_df, pipe)
        summary = save_benchmark_summary(bert_df, pipe)
        print(json.dumps(summary, indent=2))
    else:
        print("Run BERT and pipeline benchmarks first (Cells 2 & 3).")

# ── CELL BREAK ──

# ─── CELL 4: Run All + Print Headline ────────────────────────────────────────

def run_all_benchmarks() -> None:
    """
    Orchestrate all benchmarks, generate chart, and print the pitch headline.

    This is the function to call from the final notebook cell.
    """
    print("=" * 60)
    print("  AMD MI300X BENCHMARK SUITE — Clinical Trial Risk Scorer")
    print("=" * 60)

    # ── Benchmark 1: BERT embedding ──
    print("\n[1/4] Running BERT embedding benchmark...")
    bert_df = benchmark_bert_embedding()

    # ── Benchmark 2: Full pipeline ──
    print("\n[2/4] Running full pipeline benchmark...")
    pipe_results = benchmark_full_pipeline(n_trials=100, device="cuda")

    # ── Benchmark 3: vLLM (optional — only if server is running) ──
    print("\n[3/4] Running vLLM benchmark (if server is available)...")
    vllm_results = benchmark_vllm(n_prompts=20, port=8000)

    # ── Generate chart + summary ──
    print("\n[4/4] Generating chart and summary...")
    chart_path = plot_benchmark_results(bert_df, pipe_results)
    summary = save_benchmark_summary(bert_df, pipe_results)

    # ── Pitch headline ──
    print("\n" + "=" * 60)
    print(f"  PITCH HEADLINE: {summary['headline']}")
    print(f"  AMD speedup: {summary['speedup_vs_cpu']:.1f}× faster than CPU")
    print(f"  Chart saved to: {chart_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_all_benchmarks()


## INTEGRATION NOTES
# ──────────────────────────────────────────────────────────────────────────────
#
# FILES THIS MODULE READS:
#   - artifacts/xgb_model.json        (optional) Trained XGBoost model for
#                                     realistic pipeline timing. If absent,
#                                     a synthetic model is created on the fly.
#
# FILES THIS MODULE WRITES:
#   - artifacts/bert_benchmark.csv    BERT throughput results (intermediate)
#   - artifacts/pipeline_benchmark.json  Pipeline step timings (intermediate)
#   - artifacts/vllm_benchmark.json   vLLM latency stats (intermediate)
#   - artifacts/benchmark_summary.json  Pitch-deck-ready summary (final)
#   - demo/amd_benchmark.png          2-panel benchmark chart (final)
#
# ENVIRONMENT VARIABLES / CONSTANTS THE CALLER MUST SET:
#   - None required. All paths use sensible defaults relative to CWD.
#   - BERT_MODEL_NAME defaults to "emilyalsentzer/Bio_ClinicalBERT"
#     (override at module level if using a different checkpoint).
#   - vLLM server must be running on localhost:8000 for Benchmark 3.
#     Start with:
#       python -m vllm.entrypoints.openai.api_server \
#           --model mistralai/Mistral-7B-Instruct-v0.3 \
#           --port 8000 --dtype float16
#   - ROCm: torch.device("cuda") is used throughout. ROCm's HIP runtime
#     maps this automatically — never use "hip" or "rocm" as device strings.
#
# NOTEBOOK CELL MAPPING:
#   Cell 1 → imports + constants          (lines 1–47)
#   Cell 2 → benchmark_bert_embedding()   (run_all step 1)
#   Cell 3 → benchmark_full_pipeline()    (run_all step 2)
#   Cell 4 → plot + save + print headline (run_all steps 3–4)
#
# ──────────────────────────────────────────────────────────────────────────────
