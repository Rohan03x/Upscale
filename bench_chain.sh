#!/usr/bin/env bash
# bench_chain.sh — waits for bench_autotune, then runs reduce → fullgraph → 720p → trt
# Each sub-bench logs to its own file. Failures are caught and reported without stopping the chain.

AUTOTUNE_LOG=/root/Upscale/output/bench_autotune.log
CHAIN_LOG=/root/Upscale/output/bench_chain.log

exec > >(tee "$CHAIN_LOG") 2>&1

echo "=== bench_chain started: $(date) ==="
echo "Waiting for bench_autotune to finish..."

while ! grep -q "bench_autotune DONE" "$AUTOTUNE_LOG" 2>/dev/null; do
    sleep 30
done

echo "=== bench_autotune DONE detected: $(date) ==="

echo ""
echo "--- [B2] Starting bench_reduce ---"
bash /root/Upscale/bench_reduce.sh && echo "bench_reduce: OK" || echo "bench_reduce: FAILED"

echo ""
echo "--- [B3] Starting bench_fullgraph ---"
bash /root/Upscale/bench_fullgraph.sh && echo "bench_fullgraph: OK" || echo "bench_fullgraph: FAILED (expected if mask guard fires)"

echo ""
echo "--- [B4] Starting bench_720p ---"
bash /root/Upscale/bench_720p.sh && echo "bench_720p: OK" || echo "bench_720p: FAILED"

echo ""
echo "--- [B5] Starting bench_trt ---"
bash /root/Upscale/bench_trt.sh && echo "bench_trt: OK" || echo "bench_trt: FAILED"

echo ""
echo "=== ALL BENCHES COMPLETE: $(date) ==="
