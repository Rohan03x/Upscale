#!/usr/bin/env bash
# bench_tile_pad0.sh — tile=640, tile_pad=0 → 640x640 tiles (exact mult of 16, zero rounding)
# QUALITY EXPERIMENT: pad=0 means no overlap context at tile boundaries.
# HAT window_size=16; edge windows process with no neighbouring-tile context.
# If seams are visible at 640px intervals → revert to pad=8; if clean → new best.
# Per-pass compute: 2x640^2=819k vs 2x672^2=903k = 9.3% less than B-640 -> ~4.82 s/frame expected
# Same 6 tiles, same B=2, same 3 passes
set -euo pipefail
LOG=/root/Upscale/output/bench_tile_pad0.log
exec > >(tee "$LOG") 2>&1
echo "=== bench_tile_pad0: tile=640 tile_pad=0, max-autotune-no-cudagraphs ==="
echo "Start: $(date)"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/inductor_cache
cd /root/Upscale/tools/Real-ESRGAN
python3 inference_realesrgan_video.py \
    -i /root/Upscale/input/test_clip.mp4 \
    -o /root/Upscale/output/ \
    -n Real_HAT_GAN_SRx4_sharper \
    --suffix bench_tile_pad0_out \
    -s 2 \
    --tile 640 \
    --tile_pad 0 \
    --compile-mode max-autotune-no-cudagraphs
echo "=== bench_tile_pad0 DONE: $(date) ==="
