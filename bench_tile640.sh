#!/usr/bin/env bash
# bench_tile640.sh — tile=640 with max-autotune-no-cudagraphs
# 1080x1920 frame -> ceil(1920/640)=3 x ceil(1080/640)=2 = 6 tiles vs 12 at tile=512
# Fewer but larger tiles = half the forward passes per frame = potential 2x pass reduction
# TILE_BATCH auto-probe: at 640 each tile is (660x660) padded, much larger — B=2 likely, B=3 may OOM
# Quality: IDENTICAL (same model, same upscale — tile size does not affect output quality)
set -euo pipefail
LOG=/root/Upscale/output/bench_tile640.log
exec > >(tee "$LOG") 2>&1
echo "=== bench_tile640: tile=640, max-autotune-no-cudagraphs ==="
echo "Start: $(date)"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/inductor_cache
cd /root/Upscale/tools/Real-ESRGAN
python3 inference_realesrgan_video.py \
    -i /root/Upscale/input/test_clip.mp4 \
    -o /root/Upscale/output/ \
    -n Real_HAT_GAN_SRx4_sharper \
    --suffix bench_tile640_out \
    -s 2 \
    --tile 640 \
    --tile_pad 10 \
    --compile-mode max-autotune-no-cudagraphs
echo "=== bench_tile640 DONE: $(date) ==="
