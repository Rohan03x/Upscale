#!/usr/bin/env bash
# bench_trt2.sh — torch_tensorrt backend, FP16, NO torchao FP8/INT8
# Fix: torchao quantize_() adds mul_16 scale ops TRT cannot lower → full fallback + OOM
# This run skips FP8/INT8 so TRT compiles the clean FP16 model without subgraph fallback.
# Expected: TILE_BATCH recovers to 3 (less VRAM used during TRT tactic search),
#           full model compiled by TRT → genuine speedup over Inductor baseline.
LOG=/root/Upscale/output/bench_trt2.log
exec > >(tee "$LOG") 2>&1
echo "=== bench_trt2: TRT FP16, no torchao FP8/INT8 (clean compile) ==="
echo "Start: $(date)"
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/inductor_cache
cd /root/Upscale/tools/Real-ESRGAN
python3 inference_realesrgan_video.py \
    -i /root/Upscale/input/test_clip.mp4 \
    -o /root/Upscale/output/ -n Real_HAT_GAN_SRx4_sharper \
    --suffix bench_trt2_out -s 2 --tile 512 --tile_pad 10 \
    --compile-backend tensorrt
echo "=== bench_trt2 DONE: $(date) ==="
