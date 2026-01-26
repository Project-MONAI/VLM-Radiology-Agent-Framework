#!/bin/bash
# =============================================================================
# AMD ROCm Setup Script for VILA-M3 Inference
# Supports MI300X and MI300A GPUs
# Based on: https://rocm.docs.amd.com/projects/monai/en/latest/install/installation.html
# =============================================================================

set -e

echo "=============================================="
echo "VILA-M3 AMD ROCm Inference Setup"
echo "=============================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect Python
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python not found"
    exit 1
fi
echo "Using: $PYTHON_CMD"

# Upgrade pip first
pip install --upgrade pip setuptools wheel -q

# =============================================================================
# Step 1: Install PyTorch with ROCm 6.4
# =============================================================================
echo ""
echo "[1/6] Installing PyTorch ROCm 6.4..."
$PYTHON_CMD -c "import torch; assert torch.cuda.is_available(); print(f'PyTorch {torch.__version__}, GPUs: {torch.cuda.device_count()}')" 2>/dev/null || {
    # Use --no-cache-dir to avoid memory error with large wheel files
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
}
$PYTHON_CMD -c "import torch; print(f'PyTorch {torch.__version__}, ROCm: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# =============================================================================
# Step 2: Install AMD hipCIM (accelerated image loading)
# =============================================================================
echo ""
echo "[2/6] Installing AMD hipCIM..."
pip install -q amd-hipcim --extra-index-url=https://pypi.amd.com/simple 2>/dev/null || echo "hipCIM: skipped (optional)"

# =============================================================================
# Step 3: Install AMD MONAI and dependencies
# =============================================================================
echo ""
echo "[3/6] Installing AMD MONAI and dependencies..."

# MONAI optional dependencies (from AMD docs)
pip install -q nibabel pynrrd fire pytorch-ignite einops pandas tqdm psutil
pip install -q transformers pydicom scikit-image

# AMD MONAI (ROCm-optimized) - version 1.0.0 from AMD PyPI
echo "Installing amd-monai from AMD PyPI..."
# pip uninstall -y amd-monai monai 2>/dev/null || true
pip install amd-monai --extra-index-url=https://pypi.amd.com/simple --no-deps || {
    echo "WARNING: amd-monai failed, trying standard monai..."
    pip install --no-cache-dir monai
}
# Install MONAI dependencies separately
pip install -q numpy

$PYTHON_CMD -c "import monai; print(f'MONAI: {monai.__version__}')"

# =============================================================================
# Step 4: Setup VILA submodule
# =============================================================================
echo ""
echo "[4/6] Setting up VILA..."

if [ ! -f "thirdparty/VILA/pyproject.toml" ]; then
    if command -v git &> /dev/null; then
        git submodule update --init --recursive
    else
        echo "ERROR: VILA not found. Run: git submodule update --init --recursive"
        exit 1
    fi
fi

# Install VILA (no-deps to avoid conflicts)
pip install -q -e thirdparty/VILA --no-deps
echo "VILA: OK"

# VILA dependencies
pip install -q accelerate datasets tokenizers safetensors
pip install -q sentencepiece protobuf timm pillow numpy
pip install -q "huggingface_hub>=0.34.0,<1.0" "pydantic>=2.0"
pip install -q gradio httpx colored shortuuid python-dotenv
pip install -q torchxrayvision open_clip_torch peft
pip install -q deepspeed py-cpuinfo hjson ninja pydantic  # VILA imports deepspeed

# s2wrapper (VILA dependency)
$PYTHON_CMD -c "from s2wrapper import forward" 2>/dev/null || {
    echo "Installing s2wrapper..."
    pip install -q git+https://github.com/bfshi/scaling_on_scales.git 2>/dev/null || {
        curl -sL https://github.com/bfshi/scaling_on_scales/archive/refs/heads/master.zip -o /tmp/s2.zip
        $PYTHON_CMD -c "import zipfile; zipfile.ZipFile('/tmp/s2.zip').extractall('/tmp/')"
        pip install -q /tmp/scaling_on_scales-master/
        rm -f /tmp/s2.zip
    }
}

# flash_attn stub (AMD doesn't have native flash_attn)
$PYTHON_CMD -c "import flash_attn" 2>/dev/null || {
    echo "Installing flash_attn stub..."
    SITE=$($PYTHON_CMD -c "import site; print(site.getsitepackages()[0])")
    cp -r "$SCRIPT_DIR/amd_patches/flash_attn" "$SITE/" 2>/dev/null || \
    cp -r "$SCRIPT_DIR/amd_patches/flash_attn" /usr/local/lib/python3.10/dist-packages/
}

# =============================================================================
# Step 5: Download MONAI bundles
# =============================================================================
echo ""
echo "[5/6] Downloading MONAI model bundles..."

BUNDLE_DIR="$HOME/.cache/torch/hub/bundle"
mkdir -p "$BUNDLE_DIR"

if [ ! -d "$BUNDLE_DIR/vista3d_v0.5.4/vista3d" ]; then
    $PYTHON_CMD -m monai.bundle download vista3d --version 0.5.4 --bundle_dir "$BUNDLE_DIR"
    $PYTHON_CMD -c "import zipfile; zipfile.ZipFile('$BUNDLE_DIR/vista3d_v0.5.4.zip').extractall('$BUNDLE_DIR/vista3d_v0.5.4')"
fi
echo "VISTA3D: OK"

if [ ! -d "$BUNDLE_DIR/brats_mri_segmentation" ]; then
    $PYTHON_CMD -m monai.bundle download brats_mri_segmentation --version 0.5.2 --bundle_dir "$BUNDLE_DIR"
fi
echo "BRATS: OK"

# =============================================================================
# Step 6: Verify
# =============================================================================
echo ""
echo "[6/6] Verifying installation..."

$PYTHON_CMD << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

import monai
print(f"MONAI: {monai.__version__} (AMD)")

from s2wrapper import forward
print("s2wrapper: OK")

import flash_attn
print("flash_attn: OK")

from llava.constants import IMAGE_TOKEN_INDEX
print("VILA: OK")

print("\n✓ All checks passed!")
EOF

echo ""
echo "=============================================="
echo "Setup complete! Run the demo:"
echo "  cd $SCRIPT_DIR/m3/demo && $PYTHON_CMD gradio_m3.py"
echo "=============================================="
