"""Script 2: Profile compiled model + benchmark freezing + fullgraph + epilogue_fusion_first"""
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

def bench(m, label, warmup=3):
    with torch.no_grad():
        t0 = time.time()
        for _ in range(warmup): m(x)
        torch.cuda.synchronize()
        w = time.time() - t0
        t0 = time.time()
        for _ in range(N_BENCH): m(x)
        torch.cuda.synchronize()
    ms = (time.time()-t0)/N_BENCH*1000
    print(f"  {label:<50} warmup={w:.1f}s  {ms:.0f} ms/tile", flush=True)
    return ms

# ── A: Baseline (from cache) ──────────────────────────────────────────────────
print("\n" + "="*60)
print("A: Baseline max-autotune-no-cudagraphs (from cache)")
torch._dynamo.config.cache_size_limit = 64
ic.coordinate_descent_tuning = True
ic.freezing = False
ic.epilogue_fusion_first = False
c_base = torch.compile(model, mode='max-autotune-no-cudagraphs', dynamic=False)
base_ms = bench(c_base, "baseline")
torch._dynamo.reset()

# ── B: freezing=True ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("B: freezing=True")
ic.freezing = True
ic.coordinate_descent_tuning = True
ic.epilogue_fusion_first = False
c_frozen = torch.compile(model, mode='max-autotune-no-cudagraphs', dynamic=False)
frozen_ms = bench(c_frozen, "freezing=True")
torch._dynamo.reset()
ic.freezing = False

# ── C: epilogue_fusion_first=True ─────────────────────────────────────────────
print("\n" + "="*60)
print("C: epilogue_fusion_first=True")
ic.epilogue_fusion_first = True
ic.coordinate_descent_tuning = True
c_eff = torch.compile(model, mode='max-autotune-no-cudagraphs', dynamic=False)
eff_ms = bench(c_eff, "epilogue_fusion_first=True")
torch._dynamo.reset()
ic.epilogue_fusion_first = False

# ── D: fullgraph=True ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("D: fullgraph=True")
fg_ms = None
try:
    c_fg = torch.compile(model, mode='max-autotune-no-cudagraphs',
                         dynamic=False, fullgraph=True)
    fg_ms = bench(c_fg, "fullgraph=True")
    torch._dynamo.reset()
except Exception as e:
    print(f"  fullgraph=True FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
    torch._dynamo.reset()

# ── E: torch profiler on baseline ─────────────────────────────────────────────
print("\n" + "="*60)
print("E: torch.profiler on COMPILED model (baseline)")
from torch.profiler import profile, record_function, ProfilerActivity

# Re-compile baseline for profiling
ic.freezing = False
ic.epilogue_fusion_first = False
c_prof = torch.compile(model, mode='max-autotune-no-cudagraphs', dynamic=False)

with torch.no_grad():
    for _ in range(3): c_prof(x)  # warmup
    torch.cuda.synchronize()
    
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        with record_function("hat_compiled"):
            c_prof(x)
        torch.cuda.synchronize()

key = prof.key_averages()
total_cuda = sum(k.device_time for k in key if not k.is_user_annotation and k.device_time > 0)

rows = sorted(
    [k for k in key if k.device_time > 0 and not k.is_user_annotation],
    key=lambda k: k.device_time, reverse=True
)[:30]

print(f"\n  {'Op':<55} {'CUDA%':>7} {'ms':>8} {'calls':>6}")
print("  " + "-"*80)
for k in rows:
    pct = 100.0 * k.device_time / total_cuda if total_cuda > 0 else 0
    print(f"  {k.key:<55} {pct:>6.1f}% {k.device_time/1000:>7.2f}ms {k.count:>6}")
print(f"\n  Total CUDA (profiler): {total_cuda/1000:.0f}ms", flush=True)

torch._dynamo.reset()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  baseline:                   {base_ms:.0f} ms  (1.000x)")
print(f"  freezing=True:              {frozen_ms:.0f} ms  ({base_ms/frozen_ms:.3f}x  {(base_ms/frozen_ms-1)*100:+.1f}%)")
print(f"  epilogue_fusion_first=True: {eff_ms:.0f} ms  ({base_ms/eff_ms:.3f}x  {(base_ms/eff_ms-1)*100:+.1f}%)")
if fg_ms:
    print(f"  fullgraph=True:             {fg_ms:.0f} ms  ({base_ms/fg_ms:.3f}x  {(base_ms/fg_ms-1)*100:+.1f}%)")
else:
    print(f"  fullgraph=True:             FAILED (graph breaks exist)")
