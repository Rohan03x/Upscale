#!/usr/bin/env bash
# bench_trt.sh — torch_tensorrt backend, FP16 TRT engine build
# torch-tensorrt 2.8.0+cu128 + tensorrt 10.12.0.36 installed on pod
# Does NOT use set -euo pipefail — TRT may fail certain HAT subgraphs; we want the log, not an abort
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
