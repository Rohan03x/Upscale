#!/usr/bin/env bash
# bench_trt.sh — torch_tensorrt backend, FP16 TRT engine build
# Requires: pip install torch-tensorrt tensorrt (done via trt_install screen)
set -euo pipefail
LOG=/root/Upscale/output/bench_trt.log
exec > >(tee "$LOG") 2>&1
echo "=== bench_trt: tensorrt backend (FP16) ==="
echo "Start: $(date)"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/inductor_cache
cd /root/Upscale/tools/Real-ESRGAN
python3 inference_realesrgan_video.py \
    -i /root/Upscale/input/test_clip.mp4 \
    -o /root/Upscale/output/ \
    -n Real_HAT_GAN_SRx4_sharper \
    --suffix bench_trt_out \
    -s 2 \
    --tile 512 \
    --tile_pad 10 \
    --compile-backend tensorrt
echo "=== bench_trt DONE: $(date) ==="
