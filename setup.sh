#!/bin/bash
# ============================================================
# TRACE – Trial Risk Assessment & Co-Pilot Engine — AMD MI300X Environment Setup
# ============================================================
# Usage:
#   chmod +x setup.sh && ./setup.sh
# ============================================================

set -e

echo "══════════════════════════════════════════════════════════"
echo "  TRACE – Trial Risk Assessment & Co-Pilot Engine — AMD MI300X Setup"
echo "══════════════════════════════════════════════════════════"

# ── Create directories ─────────────────────────────────────
echo ""
echo "📁 Creating project directories..."
mkdir -p data artifacts demo
echo "   ✅ data/ artifacts/ demo/"

# ── Install PyTorch ROCm ───────────────────────────────────
echo ""
echo "🔥 Installing PyTorch (ROCm 6.0)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# ── Install dependencies ──────────────────────────────────
echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# ── Verify GPU ─────────────────────────────────────────────
echo ""
echo "🖥️  Verifying GPU setup..."
python -c "
import torch
print(f'  PyTorch version : {torch.__version__}')
print(f'  ROCm available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU device      : {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  GPU memory      : {mem:.1f} GB')
    x = torch.randn(256, 256, device='cuda')
    _ = torch.mm(x, x)
    print('  Compute test    : ✅ PASSED')
else:
    print('  ⚠️  GPU not detected — will fall back to CPU')
"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  ✅ Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Run the pipeline:  python pipeline.py"
echo "    2. Engineer features: python features.py"
echo "    3. Extract embeddings: python embedder.py"
echo "    4. Train model:       python trainer.py"
echo "    5. Generate SHAP:     python explainer.py"
echo "    6. Run benchmarks:    python benchmark.py"
echo "    7. Launch app:        python app.py"
echo ""
echo "  Or run everything:      python run_pipeline.py"
echo "══════════════════════════════════════════════════════════"
