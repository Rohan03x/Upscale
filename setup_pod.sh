#!/usr/bin/env bash
# =============================================================================
# VideoUpscale Pipeline — Full Pod Setup
# GPU: RTX 5090 (sm_120, 32 GB VRAM), Ubuntu 22.04, PyTorch 2.8.0+cu128
# Run: bash setup_pod.sh  (inside screen for safety)
# =============================================================================
set -e

REPO=/root/Upscale
TOOLS=$REPO/tools
MODELS=$REPO/models
CACHE=/root/autodl-tmp/inductor_cache

echo "================================================================"
echo "  VideoUpscale Pod Setup"
echo "  Repo:   $REPO"
echo "  Models: $MODELS"
echo "  Cache:  $CACHE"
echo "================================================================"

# ── 1. System packages ────────────────────────────────────────────────────────
echo ""
echo "[1/7] System packages..."
apt-get update -qq
apt-get install -y -q ffmpeg screen git wget 2>&1 | tail -5
apt-get install -y -q libvid-stab-dev 2>/dev/null || true  # not in Ubuntu 22.04 minimal; ffmpeg already has vidstab

# ── 2. Python packages ────────────────────────────────────────────────────────
echo ""
echo "[2/7] Python packages..."
pip install -q \
  basicsr realesrgan facexlib gfpgan \
  einops \
  scikit-video moviepy \
  ffmpeg-python imageio imageio-ffmpeg \
  opencv-python-headless \
  Pillow numpy scipy scikit-image tqdm \
  pyyaml addict future requests lmdb yapf \
  gdown "huggingface-hub>=0.19.0" \
  transformers diffusers accelerate omegaconf safetensors \
  torchao \
  2>&1 | tail -10

# ── 3. Clone tool repositories ────────────────────────────────────────────────
echo ""
echo "[3/7] Cloning tool repos..."
mkdir -p $TOOLS

# Real-ESRGAN (inference scripts)
if [ ! -d "$TOOLS/Real-ESRGAN/.git" ]; then
  git clone --depth=1 https://github.com/xinntao/Real-ESRGAN $TOOLS/Real-ESRGAN
  cd $TOOLS/Real-ESRGAN && pip install -q -e . && cd $REPO
else
  echo "  Real-ESRGAN already cloned"
fi

# Practical-RIFE (Python RIFE, CUDA cross-platform)
if [ ! -d "$TOOLS/RIFE/.git" ]; then
  git clone --depth=1 https://github.com/hzwer/Practical-RIFE $TOOLS/RIFE
else
  echo "  RIFE already cloned"
fi

# HAT (architecture files)
if [ ! -d "$TOOLS/HAT/.git" ]; then
  git clone --depth=1 https://github.com/XPixelGroup/HAT $TOOLS/HAT
else
  echo "  HAT already cloned"
fi

# CodeFormer (face restoration — optional)
if [ ! -d "$TOOLS/CodeFormer/.git" ]; then
  git clone --depth=1 https://github.com/sczhou/CodeFormer $TOOLS/CodeFormer
  cd $TOOLS/CodeFormer && pip install -q -r requirements.txt && cd $REPO
else
  echo "  CodeFormer already cloned"
fi

# ── 4. Download models ────────────────────────────────────────────────────────
echo ""
echo "[4/7] Downloading models..."
mkdir -p $MODELS

# RealESRGAN x4+ (64 MB — GitHub release)
if [ ! -f "$MODELS/RealESRGAN_x4plus.pth" ]; then
  echo "  Downloading RealESRGAN_x4plus.pth..."
  wget -q -O $MODELS/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
else
  echo "  RealESRGAN_x4plus.pth already exists"
fi

# Real_HAT_GAN_SRx4_sharper (162 MB — HuggingFace)
if [ ! -f "$MODELS/Real_HAT_GAN_SRx4_sharper.pth" ]; then
  echo "  Downloading Real_HAT_GAN_SRx4_sharper.pth..."
  python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='XPixelGroup/HAT', filename='Real_HAT_GAN_SRx4_sharper.pth')
shutil.copy(path, '$MODELS/Real_HAT_GAN_SRx4_sharper.pth')
print('  Downloaded via HuggingFace Hub')
" || wget -q -O $MODELS/Real_HAT_GAN_SRx4_sharper.pth \
    "https://drive.usercontent.google.com/download?id=1HcLc6v03d_BiqywcL67DO_UqDBxnRHnd&confirm=t"
else
  echo "  Real_HAT_GAN_SRx4_sharper.pth already exists"
fi

# Real_HAT_GAN_SRx4 (162 MB — HuggingFace)
if [ ! -f "$MODELS/Real_HAT_GAN_SRx4.pth" ]; then
  echo "  Downloading Real_HAT_GAN_SRx4.pth..."
  python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='XPixelGroup/HAT', filename='Real_HAT_GAN_SRx4.pth')
shutil.copy(path, '$MODELS/Real_HAT_GAN_SRx4.pth')
print('  Downloaded via HuggingFace Hub')
" || echo "  WARNING: Real_HAT_GAN_SRx4.pth download failed — skipping (sharper is primary)"
else
  echo "  Real_HAT_GAN_SRx4.pth already exists"
fi

# NAFNet deblur (65 MB — Google Drive)
if [ ! -f "$MODELS/NAFNet-GoPro-width32.pth" ]; then
  echo "  Downloading NAFNet-GoPro-width32.pth..."
  python3 -m gdown 1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj -O $MODELS/NAFNet-GoPro-width32.pth
else
  echo "  NAFNet-GoPro-width32.pth already exists"
fi

# EDVR deblur (90 MB — Google Drive)
if [ ! -f "$MODELS/EDVR-deblur.pth" ]; then
  echo "  Downloading EDVR-deblur.pth..."
  python3 -m gdown 1_ma2tgHscZtkIY2tEJkVdU-UP8bnqBRE -O $MODELS/EDVR-deblur.pth
else
  echo "  EDVR-deblur.pth already exists"
fi

# RIFE v4.25 model (train_log/ — Google Drive via gdown)
RIFE_LOG=$TOOLS/RIFE/train_log
if [ ! -f "$RIFE_LOG/RIFE_HDv3.py" ]; then
  echo "  Downloading RIFE v4.25 model weights (Google Drive)..."
  mkdir -p $RIFE_LOG
  python3 -m gdown 1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg -O /tmp/rife_v425.zip \
    && unzip -o /tmp/rife_v425.zip -d /tmp/rife_v425/ \
    && cp /tmp/rife_v425/train_log/RIFE_HDv3.py \
          /tmp/rife_v425/train_log/IFNet_HDv3.py \
          /tmp/rife_v425/train_log/refine.py \
          /tmp/rife_v425/train_log/flownet.pkl \
          $RIFE_LOG/ \
    && rm -rf /tmp/rife_v425/ /tmp/rife_v425.zip \
    && echo "  RIFE v4.25 model installed" \
    || echo "  WARNING: RIFE weights download failed — install manually"
else
  echo "  RIFE train_log already exists"
fi

# ── 5. Apply patches ──────────────────────────────────────────────────────────
echo ""
echo "[5/7] Applying patches..."

# Fix basicsr torchvision compatibility (functional_tensor removed in torchvision>=0.17)
BDIR=$(pip3 show basicsr | grep Location | awk '{print $2}')/basicsr
if grep -q 'functional_tensor' $BDIR/data/degradations.py 2>/dev/null; then
  sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    $BDIR/data/degradations.py
  echo "  Applied torchvision functional_tensor fix"
fi

# hat_arch.py → basicsr installed package
echo "  basicsr at: $BDIR"
cp $REPO/patches/hat_arch.py $BDIR/archs/hat_arch.py
echo "  Patched: $BDIR/archs/hat_arch.py"

# inference_realesrgan_video.py → Real-ESRGAN tool dir
cp $REPO/patches/inference_realesrgan_video.py $TOOLS/Real-ESRGAN/inference_realesrgan_video.py
echo "  Patched: $TOOLS/Real-ESRGAN/inference_realesrgan_video.py"

# Copy models to Real-ESRGAN/weights/ (pipeline looks here too)
mkdir -p $TOOLS/Real-ESRGAN/weights
for f in $MODELS/*.pth; do
  fname=$(basename $f)
  if [ ! -f "$TOOLS/Real-ESRGAN/weights/$fname" ]; then
    ln -s $f $TOOLS/Real-ESRGAN/weights/$fname
    echo "  Linked: weights/$fname"
  fi
done

# ── 6. Inductor cache on fast disk ────────────────────────────────────────────
echo ""
echo "[6/7] Configuring Torch inductor cache..."
mkdir -p $CACHE
# Write .env that upscale.py will auto-source (or set manually before running)
cat > $REPO/.env << 'EOF'
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/inductor_cache
EOF
echo "  Cache dir: $CACHE"
echo "  Run: source /root/Upscale/.env  (or add to ~/.bashrc)"

# Add to .bashrc for persistence
if ! grep -q "TORCHINDUCTOR_CACHE_DIR" ~/.bashrc; then
  echo 'export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/inductor_cache' >> ~/.bashrc
fi

# ── 7. Verification ───────────────────────────────────────────────────────────
echo ""
echo "[7/7] Verification..."

python3 - << 'PYEOF'
import torch, os, sys
from pathlib import Path

REPO  = Path("/root/Upscale")
TOOLS = REPO / "tools"
MODELS = REPO / "models"

print(f"  PyTorch:     {torch.__version__}")
print(f"  CUDA:        {torch.version.cuda}")
print(f"  GPU:         {torch.cuda.get_device_name(0)}")
print(f"  SM:          sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]:02d}")
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  VRAM:        {vram:.1f} GB  ->  tile=0 (full frame, no tiling!)" if vram > 24 else f"  VRAM:        {vram:.1f} GB")

print()
checks = [
    ("basicsr/hat_arch.py", Path(os.path.dirname(__import__('basicsr').__file__)) / "archs/hat_arch.py"),
    ("inference_realesrgan_video.py", TOOLS / "Real-ESRGAN/inference_realesrgan_video.py"),
    ("RIFE inference_video.py", TOOLS / "RIFE/inference_video.py"),
    ("RIFE train_log/RIFE_HDv3.py", TOOLS / "RIFE/train_log/RIFE_HDv3.py"),
    ("RealESRGAN_x4plus.pth", MODELS / "RealESRGAN_x4plus.pth"),
    ("Real_HAT_GAN_SRx4_sharper.pth", MODELS / "Real_HAT_GAN_SRx4_sharper.pth"),
    ("NAFNet-GoPro-width32.pth", MODELS / "NAFNet-GoPro-width32.pth"),
    ("EDVR-deblur.pth", MODELS / "EDVR-deblur.pth"),
]
all_ok = True
for name, path in checks:
    ok = path.exists()
    status = "OK  " if ok else "MISSING"
    print(f"  [{status}] {name}")
    if not ok:
        all_ok = False

# Verify hat_arch patch is applied
try:
    import basicsr.archs.hat_arch as ha
    import inspect
    src = inspect.getsource(ha)
    has_pad  = "_pad = (-_hd) % 8" in src
    has_cache = "_mask_cache_size" in src
    print(f"\n  hat_arch patch check:")
    print(f"    head_dim pad:   {'OK' if has_pad  else 'MISSING'}")
    print(f"    mask cache:     {'OK' if has_cache else 'MISSING'}")
except Exception as e:
    print(f"\n  hat_arch import error: {e}")

# Check if GEMM autotuning is enabled on this GPU
import torch._inductor.utils as _iu
is_big = _iu.is_big_gpu(0)
import subprocess
sm_out = subprocess.check_output(
    ["python3", "-c",
     "import torch; c=torch.cuda.get_device_capability(0); print(f'sm_{c[0]}{c[1]:02d}')"],
    text=True).strip()
print(f"\n  GEMM autotuning (is_big_gpu): {'ENABLED' if is_big else 'DISABLED'}")
print(f"  Architecture: {sm_out}")

if all_ok:
    print("\n  All checks passed — pipeline ready!")
else:
    print("\n  Some components missing — check warnings above.")
PYEOF

echo ""
echo "================================================================"
echo "  Setup complete!"
echo ""
echo "  Quick test:"
echo "    cd /root/Upscale"
echo "    source .env"
echo "    python3 upscale.py input/video.mp4 --tile 0 --model Real_HAT_GAN_SRx4_sharper"
echo ""
echo "  For long runs, use screen:"
echo "    screen -S upscale"
echo "    source .env && python3 upscale.py ..."
echo "    Ctrl+A D  (detach)"
echo "    screen -r upscale  (reattach)"
echo "================================================================"
