"""
TRACE – Trial Risk Assessment & Co-Pilot Engine — Full Pipeline Runner
====================================
Executes all pipeline steps in order with timing and progress tracking.

Usage:
    python run_pipeline.py              # Run all steps
    python run_pipeline.py --from 3     # Resume from step 3 (embedder)
    python run_pipeline.py --only 8     # Run only step 8 (app)
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path


STEPS = [
    ("pipeline.py",   "Data Ingestion",        "Fetching clinical trials from ClinicalTrials.gov"),
    ("features.py",   "Feature Engineering",   "Engineering 13 structured features"),
    ("embedder.py",   "BERT Embeddings",       "Extracting BioClinicalBERT embeddings on GPU"),
    ("trainer.py",    "Model Training",        "Training XGBoost + calibration + ablation"),
    ("explainer.py",  "SHAP Explainability",   "Generating SHAP analysis and demo cache"),
    ("benchmark.py",  "AMD Benchmarking",      "Benchmarking GPU vs CPU performance"),
    ("copilot.py",    "Co-Pilot Cache",        "Pre-generating LLM explanations"),
    ("app.py",        "Gradio App",            "Launching the web interface"),
]


def run_step(step_num: int, script: str, name: str, desc: str) -> bool:
    """Run a single pipeline step with timing."""
    print(f"\n{'='*60}")
    print(f"  Step {step_num}/8 — {name}")
    print(f"  {desc}")
    print(f"{'='*60}\n")

    start = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script],
            cwd=str(Path(__file__).parent),
            check=True,
        )
        elapsed = time.time() - start
        print(f"\n  ✅ {name} completed in {elapsed:.1f}s")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"\n  ❌ {name} FAILED after {elapsed:.1f}s (exit code {e.returncode})")
        return False

    except KeyboardInterrupt:
        print(f"\n  ⚠️  {name} interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(description="TRACE – Trial Risk Assessment & Co-Pilot Engine — Full Pipeline Runner")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from step N (1-8)")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only step N (1-8)")
    args = parser.parse_args()

    # Create required directories
    for d in ["data", "artifacts", "demo"]:
        os.makedirs(d, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        TRACE – Trial Risk Assessment & Co-Pilot Engine — Clinical Trial Predictor          ║")
    print("║              AMD MI300X Pipeline Runner                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    total_start = time.time()
    results = []

    for i, (script, name, desc) in enumerate(STEPS, 1):
        # Skip steps based on --from / --only flags
        if args.only is not None and i != args.only:
            continue
        if i < args.from_step:
            print(f"  ⏭️  Skipping Step {i}: {name}")
            continue

        success = run_step(i, script, name, desc)
        results.append((i, name, success))

        if not success and i < 8:
            print(f"\n  ⚠️  Pipeline stopped at Step {i}. Fix the error and re-run with:")
            print(f"      python run_pipeline.py --from {i}")
            break

    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline Summary ({total_elapsed:.1f}s total)")
    print(f"{'='*60}")
    for step_num, name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status}  Step {step_num}: {name}")
    print()


if __name__ == "__main__":
    main()
