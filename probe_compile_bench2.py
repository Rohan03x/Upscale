"""Script 3: Post-compile profiler (fixed) + SM warning investigation + EXHAUSTIVE search space test"""
import sys, os, time
sys.path.insert(0, 'C:/VideoUpscale/tools/Real-ESRGAN')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = 'C:/VideoUpscale/.inductor_cache'

import torch
import torch._inductor.config as ic
import torch._dynamo

print("Loading model...", flush=True)
from basicsr.archs.hat_arch import HAT

model = HAT(
    upscale=4, in_chans=3, img_size=64, window_size=16,
    compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5,
    img_range=1., depths=[6,6,6,6,6,6], embed_dim=180,
    num_heads=[6,6,6,6,6,6], mlp_ratio=2, upsampler='pixelshuffle',
    resi_connection='1conv'
).cuda().half().eval()

ckpt = torch.load('C:/VideoUpscale/models/Real_HAT_GAN_SRx4_sharper.pth',
                  map_location='cuda', weights_only=True)
sd = ckpt.get('params_ema', ckpt.get('params', ckpt))
model.load_state_dict(sd, strict=True)
print("Model loaded.", flush=True)

x = torch.randn(1, 3, 416, 416, device='cuda', dtype=torch.float16)
N_BENCH = 5

# ── A: SM count investigation ─────────────────────────────────────────────────
print("\n" + "="*60)
print("A: GPU SM investigation")
print("="*60, flush=True)
print(f"  Device: {torch.cuda.get_device_name()}")
print(f"  SM count: {torch.cuda.get_device_properties(0).multi_processor_count}")
print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"  Compute: sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")

# The warning "Not enough SMs to use max_autotune_gemm mode" comes from inductor
# Check what threshold it uses
try:
    from torch._inductor.utils import get_max_y_grid, use_max_autotune_gemm_heuristics_cuda
    print(f"  max_autotune_gemm heuristics available")
except ImportError:
    pass

# Check which specific max_autotune features DO apply despite the warning
import torch._inductor.config as ic
print(f"\n  coordinate_descent_tuning: {ic.coordinate_descent_tuning}")
print(f"  max_autotune: {ic.max_autotune}")
print(f"  max_autotune_pointwise: {getattr(ic, 'max_autotune_pointwise', 'N/A')}")
print(f"  max_autotune_gemm: {getattr(ic, 'max_autotune_gemm', 'N/A')}")
print(f"  search_space: {getattr(ic, 'max_autotune_gemm_search_space', 'N/A')}")

# ── B: Compile baseline from cache ────────────────────────────────────────────
print("\n" + "="*60)
print("B: Compile baseline from cache")
print("="*60, flush=True)

torch._dynamo.config.cache_size_limit = 64
ic.coordinate_descent_tuning = True
ic.freezing = False
ic.epilogue_fusion_first = False
c_base = torch.compile(model, mode='max-autotune-no-cudagraphs', dynamic=False)

print("  Warming up...", flush=True)
t0 = time.time()
with torch.no_grad():
    for _ in range(3): c_base(x)
torch.cuda.synchronize()
w_base = time.time() - t0
print(f"  Warmup: {w_base:.1f}s", flush=True)

with torch.no_grad():
    t0 = time.time()
    for _ in range(N_BENCH): c_base(x)
    torch.cuda.synchronize()
base_ms = (time.time()-t0)/N_BENCH*1000
print(f"  Baseline: {base_ms:.0f} ms/tile", flush=True)

# ── C: Post-compile profiler (fixed - needs both CPU and CUDA activities) ─────
print("\n" + "="*60)
print("C: torch.profiler on COMPILED model")
print("="*60, flush=True)

from torch.profiler import profile, record_function, ProfilerActivity

with torch.no_grad():
    for _ in range(2): c_base(x)  # extra warmup
    torch.cuda.synchronize()
    
    # Must include CPU activities for legacy CUDA profiling
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_flops=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        with record_function("hat_compiled"):
            c_base(x)
        torch.cuda.synchronize()

key = prof.key_averages()

# Show by GPU time (device_time)
cuda_rows = sorted(
    [k for k in key if k.device_time > 0 and not k.is_user_annotation],
    key=lambda k: k.device_time, reverse=True
)
total_cuda = sum(k.device_time for k in cuda_rows)

print(f"\n  TOP CUDA OPS (compiled model, 416x416 tile):")
print(f"  {'Op':<55} {'CUDA%':>7} {'CUDA_ms':>9} {'calls':>6}")
print("  " + "-"*82)
for k in cuda_rows[:25]:
    pct = 100.0 * k.device_time / total_cuda if total_cuda > 0 else 0
    print(f"  {k.key:<55} {pct:>6.1f}% {k.device_time/1000:>8.2f}ms {k.count:>6}")
print(f"\n  Total CUDA time (profiler 1 pass): {total_cuda/1000:.0f}ms")
print(f"  Benchmark avg (5 passes):          {base_ms:.0f}ms", flush=True)

# Category totals
print(f"\n  CATEGORY BREAKDOWN:")
attn_t = sum(k.device_time for k in cuda_rows if 'attention' in k.key.lower() or 'sdp' in k.key.lower())
gemm_t = sum(k.device_time for k in cuda_rows if 'mm' in k.key.lower() or 'gemm' in k.key.lower())
conv_t = sum(k.device_time for k in cuda_rows if 'conv' in k.key.lower())
copy_t = sum(k.device_time for k in cuda_rows if 'copy' in k.key.lower())
norm_t = sum(k.device_time for k in cuda_rows if 'norm' in k.key.lower())
triton_t = sum(k.device_time for k in cuda_rows if 'triton' in k.key.lower() or 'kernel' in k.key.lower())

print(f"  Attention (sdp/attn): {attn_t/1000:.1f}ms ({100*attn_t/total_cuda:.1f}%)")
print(f"  GEMM (mm/linear):     {gemm_t/1000:.1f}ms ({100*gemm_t/total_cuda:.1f}%)")
print(f"  Conv:                 {conv_t/1000:.1f}ms ({100*conv_t/total_cuda:.1f}%)")
print(f"  Copy:                 {copy_t/1000:.1f}ms ({100*copy_t/total_cuda:.1f}%)")
print(f"  LayerNorm:            {norm_t/1000:.1f}ms ({100*norm_t/total_cuda:.1f}%)")
print(f"  Triton kernels:       {triton_t/1000:.1f}ms ({100*triton_t/total_cuda:.1f}%)")

torch._dynamo.reset()

# ── D: EXHAUSTIVE search space test (with cache - just time overhead) ──────────
print("\n" + "="*60)
print("D: EXHAUSTIVE gemm_search_space test")
print("="*60, flush=True)

# First, test with EXHAUSTIVE but most kernels should already be in cache from DEFAULT
# Changing search_space does NOT invalidate the existing cache for already-compiled shapes
ic.max_autotune_gemm_search_space = "EXHAUSTIVE"
ic.coordinate_descent_tuning = True
ic.freezing = False
c_exhaustive = torch.compile(model, mode='max-autotune-no-cudagraphs', dynamic=False)

print("  Warming up with EXHAUSTIVE (may recompile some kernels)...", flush=True)
t0 = time.time()
with torch.no_grad():
    for _ in range(3): c_exhaustive(x)
torch.cuda.synchronize()
w_ex = time.time() - t0
print(f"  Warmup: {w_ex:.1f}s", flush=True)

with torch.no_grad():
    t0 = time.time()
    for _ in range(N_BENCH): c_exhaustive(x)
    torch.cuda.synchronize()
exhaustive_ms = (time.time()-t0)/N_BENCH*1000
print(f"  EXHAUSTIVE: {exhaustive_ms:.0f} ms/tile  (warmup={w_ex:.1f}s)", flush=True)
speedup_ex = base_ms / exhaustive_ms
print(f"  EXHAUSTIVE speedup: {speedup_ex:.3f}x  ({(speedup_ex-1)*100:+.1f}%)", flush=True)

ic.max_autotune_gemm_search_space = "DEFAULT"
torch._dynamo.reset()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Baseline (max-autotune-no-cudagraphs):  {base_ms:.0f} ms/tile  (1.000x)")
print(f"  EXHAUSTIVE search space:               {exhaustive_ms:.0f} ms/tile  ({base_ms/exhaustive_ms:.3f}x)")
print(f"  fullgraph=True:                         FAILS (mask cache guard is a graph break)")
print(f"  freezing=True:                          ~3410 ms/tile  (+0.6%, negligible)")
print(f"  epilogue_fusion_first=True:             ~3401 ms/tile  (+0.8%, negligible)")
print(f"\n  SM WARNING: 'Not enough SMs for max_autotune_gemm mode'")
print(f"  RTX 3070 has {torch.cuda.get_device_properties(0).multi_processor_count} SMs.")
print(f"  Inductor's max_autotune_gemm mode requires more SMs (threshold checked at runtime).")
print(f"  IMPLICATION: The GEMM autotuning portion of max-autotune is NOT active!")
print(f"  coordinate_descent_tuning still applies (different mechanism).")
