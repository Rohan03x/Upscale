#!/usr/bin/env bash
# bench_tile_pad8.sh — tile=640, tile_pad=8 (vs current best: tile=640, tile_pad=10)
# tile_pad=8: padded tile = 640+16=656 (exact mult of 16, vs 672 at pad=10)
# Per-pass compute: 2x656^2=860k vs 2x672^2=903k = 4.7% less compute -> ~5.1 s/frame expected
# Same 6 tiles, same B=2, same 3 passes — purely smaller tiles per pass
# Quality risk: HAT window_size=16; overlap of 8px should prevent seam artifacts
set -euo pipefail
LOG=/root/Upscale/output/bench_tile_pad8.log
exec > >(tee "$LOG") 2>&1
echo "=== bench_tile_pad8: tile=640 tile_pad=8, max-autotune-no-cudagraphs ==="
echo "Start: $(date)"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/inductor_cache
cd /root/Upscale/tools/Real-ESRGAN
python3 inference_realesrgan_video.py \
    -i /root/Upscale/input/test_clip.mp4 \
    -o /root/Upscale/output/ \
    -n Real_HAT_GAN_SRx4_sharper \
    --suffix bench_tile_pad8_out \
    -s 2 \
    --tile 640 \
    --tile_pad 8 \
    --compile-mode max-autotune-no-cudagraphs
echo "=== bench_tile_pad8 DONE: $(date) ==="
