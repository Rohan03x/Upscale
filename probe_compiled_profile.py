"""
Profile the COMPILED HAT model to find true post-compile bottlenecks.
Also tests freezing=True and counts Dynamo graph breaks.
Run from C:\VideoUpscale with venv active.
"""
import sys, os, time
sys.path.insert(0, 'C:/VideoUpscale/tools/Real-ESRGAN')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = 'C:/VideoUpscale/.inductor_cache'

import torch
import torch._inductor.config as ic
import torch._dynamo

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...", flush=True)
from basicsr.archs.hat_arch import HAT
from collections import OrderedDict

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

TILE = (416, 416)
x = torch.randn(1, 3, *TILE, device='cuda', dtype=torch.float16)

# ── SECTION 1: Count Dynamo graph breaks ─────────────────────────────────────
print("\n" + "="*60)
print("SECTION 1: Dynamo graph break analysis")
print("="*60, flush=True)

torch._dynamo.reset()
explanation = torch._dynamo.explain(model)(x)
print(f"  Number of graphs compiled:   {explanation.graph_count}")
print(f"  Number of graph breaks:      {explanation.break_reasons.__len__() if hasattr(explanation, 'break_reasons') else 'N/A'}")
print(f"  Ops per graph (approx):      {explanation.ops_per_graph}", flush=True)
# Print break reasons
if hasattr(explanation, 'break_reasons') and explanation.break_reasons:
    print("  Graph break reasons:")
    seen = set()
    for r in explanation.break_reasons:
        key = str(r.reason)[:80]
        if key not in seen:
            seen.add(key)
            print(f"    - {key}")
torch._dynamo.reset()

# ── SECTION 2: Compile (use cache) and benchmark baseline ────────────────────
print("\n" + "="*60)
print("SECTION 2: Compiled benchmark (from cache)")
print("="*60, flush=True)

torch._dynamo.config.cache_size_limit = 64
ic.coordinate_descent_tuning = True
compiled = torch.compile(model, mode='max-autotune-no-cudagraphs', dynamic=False)

# Warmup
print("  Warming up compiled model (using cache)...", flush=True)
t0 = time.time()
with torch.no_grad():
    for _ in range(3):
        compiled(x)
torch.cuda.synchronize()
print(f"  Warmup done in {time.time()-t0:.1f}s", flush=True)

# Benchmark baseline
N = 5
with torch.no_grad():
    t0 = time.time()
    for _ in range(N): compiled(x)
    torch.cuda.synchronize()
baseline_ms = (time.time()-t0)/N*1000
print(f"  Compiled baseline: {baseline_ms:.0f} ms/tile", flush=True)

# ── SECTION 3: torch.profiler on compiled model ───────────────────────────────
print("\n" + "="*60)
print("SECTION 3: torch.profiler on COMPILED model")
print("="*60, flush=True)

from torch.profiler import profile, record_function, ProfilerActivity

with torch.no_grad():
    # One warmup pass outside profiler
    compiled(x)
    torch.cuda.synchronize()
    
    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        record_shapes=False,
        with_flops=False,
        with_stack=False,
    ) as prof:
        with record_function("hat_compiled_forward"):
            compiled(x)
        torch.cuda.synchronize()

# Print top CUDA ops sorted by CUDA time
print("\n  Top CUDA ops (by CUDA time self, compiled model):")
print(f"  {'Op':<55} {'CUDA%':>7} {'CUDA_ms':>9} {'Calls':>7}")
print("  " + "-"*82)

key = prof.key_averages()
total_cuda = sum(k.device_time for k in key if not k.is_user_annotation)

rows = sorted(
    [k for k in key if k.device_time > 0 and not k.is_user_annotation],
    key=lambda k: k.device_time, reverse=True
)[:25]

for k in rows:
    pct = 100.0 * k.device_time / total_cuda if total_cuda > 0 else 0
    print(f"  {k.key:<55} {pct:>6.1f}% {k.device_time/1000:>8.1f}ms {k.count:>7}")

print(f"\n  Total CUDA time (profiler): {total_cuda/1000:.0f}ms", flush=True)

# ── SECTION 4: Test freezing=True ────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 4: freezing=True benchmark")
print("="*60, flush=True)

torch._dynamo.reset()
ic.freezing = True
ic.coordinate_descent_tuning = True
compiled_frozen = torch.compile(model, mode='max-autotune-no-cudagraphs', dynamic=False)

print("  Warming up frozen compiled model...", flush=True)
t0 = time.time()
with torch.no_grad():
    for _ in range(3):
        compiled_frozen(x)
torch.cuda.synchronize()
print(f"  Warmup done in {time.time()-t0:.1f}s", flush=True)

with torch.no_grad():
    t0 = time.time()
    for _ in range(N): compiled_frozen(x)
    torch.cuda.synchronize()
frozen_ms = (time.time()-t0)/N*1000
print(f"  Frozen compiled: {frozen_ms:.0f} ms/tile", flush=True)
print(f"  vs baseline:     {baseline_ms:.0f} ms/tile", flush=True)
speedup = baseline_ms / frozen_ms
print(f"  freezing speedup: {speedup:.3f}x  ({(speedup-1)*100:+.1f}%)", flush=True)

# Reset for next tests
ic.freezing = False
torch._dynamo.reset()

# ── SECTION 5: fullgraph=True test ───────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 5: fullgraph=True test")
print("="*60, flush=True)

ic.freezing = False
ic.coordinate_descent_tuning = True
try:
    torch._dynamo.reset()
    compiled_fg = torch.compile(model, mode='max-autotune-no-cudagraphs',
                                dynamic=False, fullgraph=True)
    print("  Compiling with fullgraph=True...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(3): compiled_fg(x)
    torch.cuda.synchronize()
    print(f"  Warmup done in {time.time()-t0:.1f}s", flush=True)
    
    with torch.no_grad():
        t0 = time.time()
        for _ in range(N): compiled_fg(x)
        torch.cuda.synchronize()
    fg_ms = (time.time()-t0)/N*1000
    print(f"  fullgraph compiled: {fg_ms:.0f} ms/tile", flush=True)
    speedup_fg = baseline_ms / fg_ms
    print(f"  fullgraph speedup: {speedup_fg:.3f}x  ({(speedup_fg-1)*100:+.1f}%)", flush=True)
    torch._dynamo.reset()
except Exception as e:
    print(f"  fullgraph=True FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
    torch._dynamo.reset()

# ── SECTION 6: epilogue_fusion_first=True test ───────────────────────────────
print("\n" + "="*60)
print("SECTION 6: epilogue_fusion_first=True test")
print("="*60, flush=True)

torch._dynamo.reset()
ic.epilogue_fusion_first = True
ic.coordinate_descent_tuning = True
ic.freezing = False
compiled_eff = torch.compile(model, mode='max-autotune-no-cudagraphs', dynamic=False)

print("  Warming up epilogue_fusion_first model...", flush=True)
t0 = time.time()
with torch.no_grad():
    for _ in range(3): compiled_eff(x)
torch.cuda.synchronize()
print(f"  Warmup done in {time.time()-t0:.1f}s", flush=True)

with torch.no_grad():
    t0 = time.time()
    for _ in range(N): compiled_eff(x)
    torch.cuda.synchronize()
eff_ms = (time.time()-t0)/N*1000
print(f"  epilogue_fusion_first compiled: {eff_ms:.0f} ms/tile", flush=True)
speedup_eff = baseline_ms / eff_ms
print(f"  epilogue_fusion_first speedup: {speedup_eff:.3f}x  ({(speedup_eff-1)*100:+.1f}%)", flush=True)

ic.epilogue_fusion_first = False
torch._dynamo.reset()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Baseline (max-autotune-no-cudagraphs):  {baseline_ms:.0f} ms/tile")
print(f"  + freezing=True:                        {frozen_ms:.0f} ms/tile  ({baseline_ms/frozen_ms:.3f}x)")
try:
    print(f"  + fullgraph=True:                       {fg_ms:.0f} ms/tile  ({baseline_ms/fg_ms:.3f}x)")
except:
    print(f"  + fullgraph=True:                       FAILED")
print(f"  + epilogue_fusion_first=True:           {eff_ms:.0f} ms/tile  ({baseline_ms/eff_ms:.3f}x)")
