"""bench_compare.py – unpatched vs patched vs compiled on real HAT weights."""
import os, sys, time, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
sys.path.insert(0, "tools/Real-ESRGAN")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from contextlib import contextmanager

ITERS   = 4
WARMUP  = 1
TILE    = 416
FRAMES  = 1098
TILES_F = 15

def make_model():
    from basicsr.archs.hat_arch import HAT
    m = HAT(
        upscale=4, in_chans=3, img_size=64, window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, img_range=1., depths=[6,6,6,6,6,6],
        embed_dim=180, num_heads=[6,6,6,6,6,6], mlp_ratio=2,
        upsampler="pixelshuffle", resi_connection="1conv",
    ).cuda().half().eval()
    wts = "tools/Real-ESRGAN/weights/Real_HAT_GAN_SRx4_sharper.pth"
    sd = torch.load(wts, map_location="cuda", weights_only=True)
    sd = sd.get("params_ema", sd.get("params", sd))
    m.load_state_dict(sd, strict=False)
    return m

print("Loading weights …")
model = make_model()
tile  = torch.rand(1, 3, TILE, TILE, device="cuda", dtype=torch.float16)

def run(m, tile):
    with torch.no_grad():
        out = m(tile)
    torch.cuda.synchronize()
    return out

def timed_bench(m, ctx, label):
    for _ in range(WARMUP):
        with ctx():
            run(m, tile)
    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with ctx():
            run(m, tile)
        times.append((time.perf_counter() - t0) * 1000)
    med = sorted(times)[len(times)//2]
    frame_s  = med / 1000 * TILES_F
    total_hr = frame_s * FRAMES / 3600
    print(f"  {label:<46} {med:6.0f} ms/tile  "
          f"{frame_s:6.1f} s/frame  {total_hr:5.1f} hrs")
    return med

@contextmanager
def math_only():
    with sdpa_kernel(SDPBackend.MATH): yield

@contextmanager
def auto_sdpa():
    yield   # no override — uses patched auto-select (mem_efficient_sdp)

print(f"\n{'Config':<46} {'ms/tile':>8}  {'s/frame':>9}  {'hrs':>6}")
print("-" * 75)
t_unpatched = timed_bench(model, math_only, "UNPATCHED  (math_sdp, hd=30)")
t_patched   = timed_bench(model, auto_sdpa, "PATCHED    (mem_eff_sdp, hd=32, bcast)")

# compile
print()
print("Compiling model (first call = tracing, may take 2-3 min) …")
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
model_c = torch.compile(model, mode="default", dynamic=True)

t0 = time.perf_counter()
with auto_sdpa():
    run(model_c, tile)
print(f"  compile trace done in {(time.perf_counter()-t0):.1f} s")

t_compiled = timed_bench(model_c, auto_sdpa, "PATCHED+COMPILE (hd=32, bcast, dynamo)")

print()
print(f"  Attention kernel alone (bench_attn.py):  math 152 ms  ->  mem_eff 5.8 ms  (26.3x)")
print(f"  End-to-end tile speedup:")
print(f"    patched vs unpatched : {t_unpatched/t_patched:.2f}x")
print(f"    compiled vs unpatched: {t_unpatched/t_compiled:.2f}x")
print()
print(f"  Original pipeline (tile=512 FP32, observed): ~443 s/frame")
est_tile512 = 443 / TILES_F * 12 / TILES_F   # rough: 12 tiles before, 15 now
print(f"  Patched  (tile=384 FP16, real model):  {t_patched/1000*TILES_F:.1f} s/frame")
print(f"  Compiled (tile=384 FP16, real model):  {t_compiled/1000*TILES_F:.1f} s/frame")
print(f"  Speedup vs original 443 s/frame:       {443/(t_patched/1000*TILES_F):.1f}x  /  "
      f"{443/(t_compiled/1000*TILES_F):.1f}x")
print("\n✓ Done.")
