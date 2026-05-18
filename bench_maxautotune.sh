#!/usr/bin/env bash
# bench_maxautotune.sh — max-autotune WITH CUDA graphs (Triton tuning + CUDA graphs combined)
# Untested combo: reduce-overhead=CUDA graphs only (6.97 s/f), max-autotune-no-cudagraphs=Triton only (6.60 s/f)
# This combines both. Expected: potentially faster, may crash on dynamic shapes in HAT mask cache.
set -euo pipefail
LOG=/root/Upscale/output/bench_maxautotune.log
exec > >(tee "$LOG") 2>&1
echo "=== bench_maxautotune: max-autotune (Triton + CUDA graphs) ==="
echo "Start: $(date)"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/inductor_cache
cd /root/Upscale/tools/Real-ESRGAN
python3 inference_realesrgan_video.py \
    -i /root/Upscale/input/test_clip.mp4 \
    -o /root/Upscale/output/ \
    -n Real_HAT_GAN_SRx4_sharper \
    --suffix bench_maxautotune_out \
    -s 2 \
    --tile 512 \
    --tile_pad 10 \
    --compile-mode max-autotune
echo "=== bench_maxautotune DONE: $(date) ==="
