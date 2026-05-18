"""
bench_ocab_fix.py – Quantify the OCAB head_dim=30 fix (math_sdp → mem_eff_sdp).
Also benchmarks:
  - torch.compile modes ('default' vs 'max-autotune' vs 'reduce-overhead')
  - F.pad copy_ overhead in WindowAttention
  - tile_pad 16 vs 8 pixel count impact
Run: python bench_ocab_fix.py
"""
import os, sys, time, warnings, gc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
sys.path.insert(0, "tools/Real-ESRGAN")

import torch
import torch.nn.functional as F

# ─── OCAB head_dim=30 micro-benchmark (math vs mem_eff) ───────────────────────
print("=" * 70)
print("PART 1: OCAB cross-attention kernel benchmark (nq=256, nk=576, hd=30)")
print("=" * 70)

DEVICE = "cuda"
dtype  = torch.float16
B_     = 676        # nw * batch = typical tile windows
nH     = 6          # num_heads
nq     = 256        # q seq len (window_size^2 = 16^2)
nk     = 576        # k/v seq len (overlap_win_size^2 = 24^2)
hd     = 30         # head_dim = dim / num_heads = 180 / 6
scale  = hd ** -0.5

# Build bias tensor [1, nH, nq, nk]
bias = torch.randn(1, nH, nq, nk, device=DEVICE, dtype=dtype) * 0.1

def bench(fn, n_warm=10, n_iter=50):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000  # ms

# --- math_sdp (current OCAB state) ---
q_30 = torch.randn(B_, nH, nq, hd, device=DEVICE, dtype=dtype)
k_30 = torch.randn(B_, nH, nk, hd, device=DEVICE, dtype=dtype)
v_30 = torch.randn(B_, nH, nk, hd, device=DEVICE, dtype=dtype)
bias_30 = bias.expand(B_, -1, -1, -1)  # materialize full [B_, nH, nq, nk]

def sdpa_math():
    with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
        return F.scaled_dot_product_attention(q_30, k_30, v_30, attn_mask=bias_30, scale=scale)

ms_math = bench(sdpa_math)
print(f"  math_sdp   head_dim=30, bias materialized:  {ms_math:7.2f} ms/call")

# --- auto-select head_dim=30 (what current OCAB does without our patch) ---
def sdpa_auto_30():
    return F.scaled_dot_product_attention(q_30, k_30, v_30, attn_mask=bias_30, scale=scale)

ms_auto30 = bench(sdpa_auto_30)
which = "mem_eff" if ms_auto30 < ms_math * 0.8 else "math"
print(f"  auto-select head_dim=30, bias materialized: {ms_auto30:7.2f} ms/call  → backend={which}")

# --- F.pad 30→32 then mem_eff then crop (our proposed patch) ---
hd_pad = 32
q_32 = F.pad(q_30, (0, hd_pad - hd))
k_32 = F.pad(k_30, (0, hd_pad - hd))
v_32 = F.pad(v_30, (0, hd_pad - hd))
bias_32 = bias.expand(B_, -1, -1, -1)

def sdpa_padded_32():
    _q = F.pad(q_30, (0, 2))
    _k = F.pad(k_30, (0, 2))
    _v = F.pad(v_30, (0, 2))
    out = F.scaled_dot_product_attention(_q, _k, _v, attn_mask=bias_32, scale=scale)
    return out[..., :hd]   # crop back

ms_pad32 = bench(sdpa_padded_32)
print(f"  pad 30→32 + mem_eff + crop:                  {ms_pad32:7.2f} ms/call")

# --- pure mem_eff head_dim=32 (upper bound) ---
q_32_pure = torch.randn(B_, nH, nq, hd_pad, device=DEVICE, dtype=dtype)
k_32_pure = torch.randn(B_, nH, nk, hd_pad, device=DEVICE, dtype=dtype)
v_32_pure = torch.randn(B_, nH, nk, hd_pad, device=DEVICE, dtype=dtype)

def sdpa_pure32():
    return F.scaled_dot_product_attention(q_32_pure, k_32_pure, v_32_pure, attn_mask=bias_32, scale=scale)

ms_pure32 = bench(sdpa_pure32)
print(f"  pure mem_eff head_dim=32 (no pad overhead):  {ms_pure32:7.2f} ms/call")

print(f"\n  Speedup (auto30 → padded32): {ms_auto30/ms_pad32:.2f}x")
print(f"  Speedup (auto30 → pure32):   {ms_auto30/ms_pure32:.2f}x")
print(f"  pad overhead vs pure32:       +{(ms_pad32-ms_pure32)/ms_pure32*100:.1f}%")

# Scale to full model (6 OCAB calls per forward):
print(f"\n  Per tile (6 OCAB calls):")
print(f"    Current (math_sdp auto30):  {ms_auto30 * 6:7.1f} ms")
print(f"    After patch (padded32):     {ms_pad32  * 6:7.1f} ms")
print(f"    After patch (pure32):       {ms_pure32 * 6:7.1f} ms")
print(f"    EXPECTED SAVING from patch: {(ms_auto30 - ms_pad32) * 6:7.1f} ms/tile")

# ─── PART 2: F.pad copy_ overhead measurement ─────────────────────────────────
print("\n" + "=" * 70)
print("PART 2: F.pad copy_ overhead in WindowAttention (HAB)")
print("=" * 70)

# HAB window attention shapes
B_hab  = 676
nH_hab = 6
N_hab  = 256
hd_hab = 30

q_hab = torch.randn(B_hab, nH_hab, N_hab, hd_hab, device=DEVICE, dtype=dtype)
k_hab = torch.randn(B_hab, nH_hab, N_hab, hd_hab, device=DEVICE, dtype=dtype)
v_hab = torch.randn(B_hab, nH_hab, N_hab, hd_hab, device=DEVICE, dtype=dtype)
bias_hab = torch.randn(1, nH_hab, N_hab, N_hab, device=DEVICE, dtype=dtype) * 0.1

def hab_no_pad():
    return F.scaled_dot_product_attention(q_hab, k_hab, v_hab, attn_mask=bias_hab, scale=scale)

def hab_with_pad():
    _q = F.pad(q_hab, (0, 2))
    _k = F.pad(k_hab, (0, 2))
    _v = F.pad(v_hab, (0, 2))
    out = F.scaled_dot_product_attention(_q, _k, _v, attn_mask=bias_hab, scale=scale)
    return out[..., :hd_hab]

ms_no_pad  = bench(hab_no_pad)
ms_with_pad = bench(hab_with_pad)
pad_overhead = ms_with_pad - ms_no_pad

print(f"  HAB attention (no pad, math_sdp):  {ms_no_pad:.2f} ms/call")
print(f"  HAB attention (with pad, mem_eff): {ms_with_pad:.2f} ms/call")
print(f"  Pad overhead per call:             {pad_overhead:.2f} ms")
print(f"  Net savings per call (vs math_sdp): {ms_no_pad - ms_with_pad:.2f} ms")

# 36 HABs × 2 attention calls each = 72 total HAB attention calls
n_hab_calls = 72
print(f"\n  Per tile ({n_hab_calls} HAB attention calls):")
print(f"    Without pad (math_sdp total): {ms_no_pad * n_hab_calls:.1f} ms")
print(f"    With pad (mem_eff total):     {ms_with_pad * n_hab_calls:.1f} ms")
print(f"    Net saving from padding:      {(ms_no_pad - ms_with_pad) * n_hab_calls:.1f} ms/tile")
print(f"    Pad F.pad overhead only:      {pad_overhead * n_hab_calls:.1f} ms/tile")

# ─── PART 3: tile_pad overhead calculation ─────────────────────────────────────
print("\n" + "=" * 70)
print("PART 3: tile_pad overhead analysis")
print("=" * 70)

tile_size = 384
for pad in [0, 8, 16, 32]:
    total = tile_size + 2 * pad
    n_windows = (total // 16) ** 2
    area_ratio = (total ** 2) / (tile_size ** 2)
    overhead_pct = (area_ratio - 1.0) * 100
    print(f"  tile={tile_size}, pad={pad:2d}: total={total:3d}x{total:3d},  "
          f"windows={n_windows:4d},  area_ratio={area_ratio:.3f},  overhead={overhead_pct:+.1f}%")

print(f"\n  Reducing pad 16→8 saves {((416**2 - 400**2) / 416**2)*100:.1f}% of computation per tile")
print(f"  Estimated tile time saving (compiled 5503ms baseline): "
      f"{5503 * (416**2 - 400**2) / 416**2:.0f} ms/tile")

# ─── PART 4: torch.compile mode comparison ─────────────────────────────────────
print("\n" + "=" * 70)
print("PART 4: torch.compile mode comparison")
print("=" * 70)

from basicsr.archs.hat_arch import HAT

def load_model():
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

tile_bench = torch.rand(1, 3, 416, 416, device="cuda", dtype=torch.float16)

def bench_model(model, n_warm=3, n_iter=10):
    with torch.no_grad():
        for _ in range(n_warm):
            model(tile_bench)
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(tile_bench)
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter * 1000

# Baseline (no compile)
print("  Loading model for torch.compile tests …")
model_base = load_model()
ms_base = bench_model(model_base)
print(f"  Baseline (no compile):            {ms_base:7.1f} ms/tile")

del model_base; gc.collect(); torch.cuda.empty_cache()

# mode='default' (already benchmarked as ~5503ms, but test here)
print("  Compiling with mode='default' (warm-up may take ~3min) …")
model_default = load_model()
torch._dynamo.config.cache_size_limit = 64
model_default = torch.compile(model_default, mode='default', dynamic=True)
ms_default_warmup_start = time.perf_counter()
with torch.no_grad():
    model_default(tile_bench)  # first compile
    torch.cuda.synchronize()
warmup_s = time.perf_counter() - ms_default_warmup_start
print(f"    Compile warm-up time: {warmup_s:.1f} s")
ms_default = bench_model(model_default)
print(f"  mode='default':                   {ms_default:7.1f} ms/tile  ({ms_base/ms_default:.2f}x vs baseline)")

del model_default; gc.collect(); torch.cuda.empty_cache()
torch._dynamo.reset()

# mode='reduce-overhead' (CUDA Graphs)
print("  Compiling with mode='reduce-overhead' (dynamic=False required for CUDA graphs) …")
try:
    model_ro = load_model()
    torch._dynamo.config.cache_size_limit = 64
    # reduce-overhead needs static shapes for CUDA graphs — use dynamic=True as fallback
    model_ro = torch.compile(model_ro, mode='reduce-overhead', dynamic=True)
    t_compile = time.perf_counter()
    with torch.no_grad():
        model_ro(tile_bench)
        torch.cuda.synchronize()
    print(f"    Compile warm-up time: {time.perf_counter() - t_compile:.1f} s")
    ms_ro = bench_model(model_ro)
    print(f"  mode='reduce-overhead':           {ms_ro:7.1f} ms/tile  ({ms_base/ms_ro:.2f}x vs baseline)")
    del model_ro; gc.collect(); torch.cuda.empty_cache()
    torch._dynamo.reset()
except Exception as e:
    print(f"  mode='reduce-overhead' FAILED: {e}")

print("\n  NOTE: mode='max-autotune' skipped — requires 10-30+ min warm-up per tile size.")
print("  Expected gain over 'default': typically 10-20% for transformer workloads.")

print("\n✓ Benchmark complete.")
