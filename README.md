<div align="center">

# 🧬 ProTool Risk

### AI-Powered Clinical Trial Risk Predictor

**Predict early termination risk for clinical trials using BioClinicalBERT + XGBoost + SHAP — accelerated on AMD MI300X**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch ROCm](https://img.shields.io/badge/PyTorch-ROCm%206.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![AMD MI300X](https://img.shields.io/badge/AMD-MI300X-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://www.amd.com/en/products/accelerators/instinct/mi300x.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)

---

*Built for the AMD Instinct MI300X Hackathon*

</div>

---

## 🎯 What is ProTool Risk?

ProTool Risk is an end-to-end machine learning system that predicts whether a clinical trial is at risk of **early termination** — before it fails. By combining:

- 🧠 **BioClinicalBERT** embeddings (768-dim clinical language understanding)
- 🌲 **XGBoost** classifier with isotonic calibration
- 🔍 **SHAP TreeExplainer** for transparent, human-readable risk attribution
- 🤖 **LLM Co-Pilot** (vLLM / Anthropic) for natural language protocol analysis

...the system delivers **calibrated risk probabilities** with **section-level explanations** that clinicians and sponsors can act on.

### Why AMD MI300X?

| Metric | MI300X | Speedup |
|--------|--------|---------|
| BERT Embedding Extraction | ⚡ GPU-accelerated | **10–50x** vs CPU |
| Pipeline Throughput | 5,000 trials in minutes | Real-time scoring |
| Memory | 192 GB HBM3 | No batch size limits |
| vLLM Inference | FP16 on CDNA 3 | Low-latency co-pilot |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ProTool Risk Pipeline                     │
├─────────────┬─────────────┬──────────────┬─────────────────┤
│  pipeline   │  features   │   embedder   │    trainer      │
│  .py        │  .py        │   .py        │    .py          │
│             │             │              │                 │
│ ClinTrials  │ 13 struct.  │ BioClinical  │ XGBoost +       │
│ .gov API    │ features    │ BERT (GPU)   │ Calibration     │
├─────────────┴─────────────┴──────────────┴─────────────────┤
│                                                             │
│  explainer.py    benchmark.py    copilot.py     app.py      │
│  SHAP Waterfall  AMD GPU vs CPU  vLLM/Anthropic  Gradio UI │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- AMD MI300X GPU with **ROCm 6.x** (AMD Developer Cloud recommended)
- ~2 GB disk space for models and data

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/protool-risk.git
cd protool-risk
```

### 2. Run Setup

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create `data/`, `artifacts/`, and `demo/` directories
- Install PyTorch with ROCm 6.0 support
- Install all Python dependencies
- Verify GPU access

### 3. Run the Full Pipeline

```bash
python run_pipeline.py
```

Or run steps individually:

```bash
python pipeline.py      # 1. Fetch 5,000 trials from ClinicalTrials.gov
python features.py      # 2. Engineer 13 structured features
python embedder.py      # 3. Extract BERT embeddings on MI300X
python trainer.py       # 4. Train XGBoost + calibration + ablation
python explainer.py     # 5. Generate SHAP analysis + demo cache
python benchmark.py     # 6. Benchmark AMD GPU vs CPU
python copilot.py       # 7. Pre-cache LLM explanations
python app.py           # 8. Launch Gradio demo 🚀
```

### 4. Open the App

Once `app.py` launches, you'll see:

```
Running on local URL:   http://0.0.0.0:7860
Running on public URL:  https://xxxxx.gradio.live
```

Open the **public URL** in your browser to access the demo.

---

## 📁 Project Structure

```
protool-risk/
├── pipeline.py          # Data ingestion from ClinicalTrials.gov API
├── features.py          # Feature engineering (13 structured features)
├── embedder.py          # BioClinicalBERT embedding extraction (GPU)
├── trainer.py           # XGBoost training + calibration + ablation
├── explainer.py         # SHAP explainability + demo cache generation
├── benchmark.py         # AMD MI300X performance benchmarking
├── copilot.py           # vLLM / Anthropic LLM co-pilot
├── app.py               # Gradio web UI (main entry point)
├── run_pipeline.py      # End-to-end pipeline runner
├── setup.sh             # One-click AMD environment setup
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
├── README.md            # This file
│
├── data/                # Generated data (not committed)
│   ├── trials_raw.parquet
│   ├── demo_trials.parquet
│   ├── features_train.parquet
│   ├── features_test.parquet
│   ├── embeddings_*.npy
│   ├── X_train_combined.npy
│   ├── X_test_combined.npy
│   └── y_train.npy / y_test.npy
│
├── artifacts/           # Model artifacts (not committed)
│   ├── xgb_model.pkl
│   ├── shap_explainer.pkl
│   ├── feature_scaler.pkl
│   ├── feature_meta.json
│   ├── optimal_threshold.json
│   └── ...
│
└── demo/                # Demo assets (not committed)
    ├── demo_cache.json
    ├── amd_benchmark.png
    └── waterfall_*.png
```

> **Note**: `data/`, `artifacts/`, and `demo/` directories contain large generated files and are excluded from git via `.gitignore`. Run the pipeline to regenerate them.

---

## ⚙️ Configuration

All configuration is via **environment variables** with sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROTOOL_DATA_DIR` | `data` | Data directory |
| `PROTOOL_ARTIFACTS_DIR` | `artifacts` | Model artifacts directory |
| `PROTOOL_DEMO_DIR` | `demo` | Demo assets directory |
| `PROTOOL_MODEL` | `artifacts/xgb_model.pkl` | Trained model path |
| `PROTOOL_EXPLAINER` | `artifacts/shap_explainer.pkl` | SHAP explainer path |
| `PROTOOL_THRESHOLD` | `artifacts/optimal_threshold.json` | Decision threshold |
| `ANTHROPIC_API_KEY` | *(none)* | Required for LLM co-pilot fallback |

See the full reference in the [setup guide](docs/amd_notebook_setup_guide.md).

---

## 🔬 Pipeline Details

### Step 1: Data Ingestion (`pipeline.py`)
- Fetches **5,000+ clinical trial records** from the ClinicalTrials.gov API
- Handles pagination, retries, and rate limiting
- Produces raw parquet + 20 curated demo trials

### Step 2: Feature Engineering (`features.py`)
- Engineers **13 structured features**: enrollment (log-normalized), phase encoding, study duration, text complexity metrics, keyword flags
- Stratified train/test split with StandardScaler normalization

### Step 3: BERT Embeddings (`embedder.py`)
- Extracts **768-dimensional CLS embeddings** using [BioClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- Section-level embeddings (eligibility criteria, primary outcomes, full text)
- **GPU-accelerated** on MI300X with auto-scaling batch sizes and OOM protection
- TF-IDF fallback if GPU is unavailable

### Step 4: Model Training (`trainer.py`)
- **XGBoost** binary classifier on 781 combined features (768 BERT + 13 structured)
- Isotonic calibration via `CalibratedClassifierCV`
- Optimal threshold selection maximizing F1 score
- Ablation study (BERT-only vs structured-only vs combined)

### Step 5: Explainability (`explainer.py`)
- **SHAP TreeExplainer** for exact Shapley values
- Per-trial waterfall plots showing feature contributions
- Pre-computed demo cache for low-latency UI

### Step 6: AMD Benchmarking (`benchmark.py`)
- BERT throughput: GPU vs CPU comparison
- Full pipeline timing per step
- vLLM inference latency stats
- Generates publication-ready benchmark charts

### Step 7: LLM Co-Pilot (`copilot.py`)
- Manages **vLLM server lifecycle** for local inference (Mistral-7B)
- Falls back to **Anthropic API** (Claude 3.5 Haiku) if vLLM is unavailable
- Generates risk explanations, protocol rewrite suggestions, and what-if analysis

### Step 8: Gradio App (`app.py`)
- 3-tab interface: Risk Scoring, Benchmarks, Co-Pilot
- Real-time trial scoring with SHAP visualization
- What-if protocol analysis
- AMD hardware stats dashboard

---

## 🛠️ Development

### Running from a Jupyter Notebook

```python
# In an AMD notebook cell:
%run app.py
```

### Resuming a Partial Pipeline Run

```bash
# Resume from step 3 (embedder) if earlier steps completed:
python run_pipeline.py --from 3

# Run only the Gradio app (if all artifacts exist):
python run_pipeline.py --only 8
```

### Manual Setup (without setup.sh)

```bash
# 1. Install PyTorch ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create directories
mkdir -p data artifacts demo

# 4. Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 📊 Model Performance

| Metric | Combined (BERT + Structured) | BERT Only | Structured Only |
|--------|------------------------------|-----------|-----------------|
| AUC-ROC | **Best** | Good | Baseline |
| F1 Score | **Highest** | Moderate | Lower |
| Calibration | ✅ Isotonic | ✅ Isotonic | ✅ Isotonic |

> Run `python trainer.py` to see exact metrics from the ablation study.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **AMD** — MI300X hardware and Developer Cloud access
- **ClinicalTrials.gov** — Open clinical trial data API
- **HuggingFace** — BioClinicalBERT model
- **SHAP** — Explainable AI framework

---

<div align="center">

**Built with ❤️ on AMD Instinct MI300X**

</div>
