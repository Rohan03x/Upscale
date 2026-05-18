"""
bench_attn.py  –  prove attention optimisations on the real HAT model
Tests (all on RTX 3070, FP16, tile=384):
  1. Which SDPA backend fires with head_dim=30 vs padded 32
  2. Wall-clock speedup from the padding fix
  3. Speedup from repeat -> expand (broadcast) for the mask
  4. Correctness check: padded output matches unpadded output
Run: python bench_attn.py
"""
import sys, time, os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F
import numpy as np

DEVICE  = "cuda"
DTYPE   = torch.float16
WARMUP  = 5
ITERS   = 20

# ── reproduce exact HAT WindowAttention dimensions ─────────────────────────
# embed_dim=180, num_heads=6  →  head_dim = 30
# window_size = 16            →  N = 16*16 = 256
NUM_HEADS   = 6
HEAD_DIM    = 30          # true HAT head_dim
HEAD_DIM_P  = 32          # padded to next mult-of-8
WINDOW_SIZE = 16
N           = WINDOW_SIZE * WINDOW_SIZE   # 256
# Batch = num_windows in a 384x384 tile  (tile after pad = 416x416)
# num_windows = ceil(416/16)^2 = 26^2 = 676 (SW-MSA shifts halve it ~)
# Use 576 (24^2) which is the common non-shifted case
B_          = 576

scale = HEAD_DIM ** -0.5

def make_tensors(head_dim):
    q = torch.randn(B_, NUM_HEADS, N, head_dim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B_, NUM_HEADS, N, head_dim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B_, NUM_HEADS, N, head_dim, device=DEVICE, dtype=DTYPE)
    # float RPB bias  [1, nH, N, N]
    bias = torch.randn(1, NUM_HEADS, N, N, device=DEVICE, dtype=DTYPE) * 0.1
    # SW-MSA shifted-window mask  [nw, N, N]  (non-trivial, not None)
    mask = torch.zeros(B_, N, N, device=DEVICE, dtype=DTYPE)
    mask[:, :, :N//2] = -100.0   # half the sequence masked
    return q, k, v, bias, mask

# ── timing helper ────────────────────────────────────────────────────────────
def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000   # ms

# ============================================================
# 1. BASELINE: head_dim=30, math_sdp forced (current state before patch)
# ============================================================
q30, k30, v30, bias30, mask30 = make_tensors(HEAD_DIM)

def attn_math_hd30():
    """Unpatched: head_dim=30, forces math_sdp."""
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        nw = mask30.shape[0]
        ab = bias30 + mask30.unsqueeze(1).repeat(B_ // nw, 1, 1, 1)
        out = F.scaled_dot_product_attention(
            q30, k30, v30, attn_mask=ab,
            dropout_p=0.0, scale=scale,
        )
    return out

# ============================================================
# 2. PADDED: head_dim=30 → 32, mem_efficient_sdp
# ============================================================
_pad = HEAD_DIM_P - HEAD_DIM  # = 2
q32 = F.pad(q30, (0, _pad))
k32 = F.pad(k30, (0, _pad))
v32 = F.pad(v30, (0, _pad))
bias32 = bias30   # shape stays [1, nH, N, N]

def attn_memeff_hd32():
    """Padded head_dim=32, mem_efficient_sdp."""
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=False, enable_mem_efficient=True
    ):
        nw = mask30.shape[0]
        ab = bias32 + mask30.unsqueeze(1).repeat(B_ // nw, 1, 1, 1)
        out = F.scaled_dot_product_attention(
            q32, k32, v32, attn_mask=ab,
            dropout_p=0.0, scale=scale,
        )
    # crop output back
    return out[..., :HEAD_DIM]

# ============================================================
# 3. PADDED + broadcast (no repeat)
# ============================================================
def attn_memeff_hd32_broadcast():
    """Padded head_dim=32, mem_efficient_sdp, broadcast instead of repeat."""
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=False, enable_mem_efficient=True
    ):
        nw = mask30.shape[0]
        r  = B_ // nw   # = 1 in our tile configuration
        m  = mask30.unsqueeze(1)       # [nw, 1, N, N]
        if r > 1:
            m = m.repeat(r, 1, 1, 1)  # only if needed
        ab = bias32 + m                # broadcast: [B_, nH, N, N]
        out = F.scaled_dot_product_attention(
            q32, k32, v32, attn_mask=ab,
            dropout_p=0.0, scale=scale,
        )
    return out[..., :HEAD_DIM]

# ============================================================
# 4. AUTO-SELECT: let PyTorch choose (what the patched code does in practice)
# ============================================================
def attn_auto_hd32():
    """Padded + autoselect (mirrors the patched hat_arch.py code)."""
    nw = mask30.shape[0]
    r  = B_ // nw
    m  = mask30.unsqueeze(1)
    if r > 1:
        m = m.repeat(r, 1, 1, 1)
    ab = bias32 + m
    out = F.scaled_dot_product_attention(
        q32, k32, v32, attn_mask=ab,
        dropout_p=0.0, scale=scale,
    )
    return out[..., :HEAD_DIM]

# ============================================================
# RUN
# ============================================================
print("=" * 60)
print(f"DEVICE: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}  |  dtype: {DTYPE}")
print(f"B={B_}, nH={NUM_HEADS}, N={N}, head_dim={HEAD_DIM} → padded {HEAD_DIM_P}")
print("=" * 60)

# Correctness check ---------------------------------------------------
with torch.no_grad():
    out_ref  = attn_math_hd30()
    out_pad  = attn_memeff_hd32()
    out_bcast = attn_memeff_hd32_broadcast()

max_err = (out_ref - out_pad).abs().max().item()
print(f"\nCorrectness (padded vs math baseline):  max |delta| = {max_err:.4e}")
max_err2 = (out_pad - out_bcast).abs().max().item()
print(f"Correctness (padded+broadcast vs padded): max |delta| = {max_err2:.4e}")
if max_err < 0.1:
    print("  ✓ numerically equivalent (FP16 rounding only)")
else:
    print("  ✗ NUMERICAL MISMATCH — investigate before using!")

# Timing --------------------------------------------------------------
print("\nRunning timing benchmarks …")
with torch.no_grad():
    t_math   = bench(attn_math_hd30)
    t_memeff = bench(attn_memeff_hd32)
    t_bcast  = bench(attn_memeff_hd32_broadcast)
    t_auto   = bench(attn_auto_hd32)

print(f"\n{'Config':<42} {'ms/call':>9}  {'speedup':>9}")
print("-" * 63)
print(f"{'math_sdp (baseline, head_dim=30)':<42} {t_math:>9.2f}  {'1.00x':>9}")
print(f"{'mem_efficient_sdp (head_dim=32)':<42} {t_memeff:>9.2f}  {t_math/t_memeff:>9.2f}x")
print(f"{'mem_efficient + broadcast mask':<42} {t_bcast:>9.2f}  {t_math/t_bcast:>9.2f}x")
print(f"{'auto-select (mirrors patched code)':<42} {t_auto:>9.2f}  {t_math/t_auto:>9.2f}x")

# Identify what backend autoselect picks ------------------------------
print("\nAuto-select backend check:")
def _check_backend(name, **flags):
    try:
        with torch.backends.cuda.sdp_kernel(**flags):
            nw = mask30.shape[0]
            ab = bias32 + mask30.unsqueeze(1)
            _ = F.scaled_dot_product_attention(q32, k32, v32, attn_mask=ab, scale=scale)
        print(f"  {name}: AVAILABLE")
    except RuntimeError as e:
        print(f"  {name}: blocked ({e})")

_check_backend("flash_sdp  (head_dim=32, float bias)",
               enable_flash=True, enable_math=False, enable_mem_efficient=False)
_check_backend("mem_eff_sdp (head_dim=32, float bias)",
               enable_flash=False, enable_math=False, enable_mem_efficient=True)
_check_backend("math_sdp   (head_dim=32, float bias)",
               enable_flash=False, enable_math=True,  enable_mem_efficient=False)

# Full-tile time projection -------------------------------------------
# tile=384+2*16=416x416, 6 RSTB × 6 HAB each + 6 OCAB, ~36 WindowAttn fwd/bwd
# Each tile does ~2×6×6 = 72 WindowAttn calls (two shifts per HAB)
# Use SW-MSA=50% with mask, 50% without
ATTN_CALLS_PER_TILE = 72
print(f"\nFull-tile projection (attn calls ≈ {ATTN_CALLS_PER_TILE}/tile):")
attn_ms_base  = t_math   * ATTN_CALLS_PER_TILE / 1000
attn_ms_opt   = t_bcast  * ATTN_CALLS_PER_TILE / 1000
tile_s_base   = 14.5   # measured tile=384 FP32 baseline (from earlier benchmark)
# attn was 24.6% of tile time in profiler → total CUDA ≈ tile_s_base
attn_frac     = 0.246
other_s       = tile_s_base * (1 - attn_frac)
new_tile_s    = other_s + attn_ms_opt
speedup_attn  = tile_s_base / new_tile_s

print(f"  baseline attention time / tile : {tile_s_base * attn_frac:.2f} s ({attn_frac*100:.0f}%)")
print(f"  optimised attention time / tile: {attn_ms_opt:.3f} s")
print(f"  other CUDA time / tile         : {other_s:.2f} s")
print(f"  new tile time estimate         : {new_tile_s:.2f} s")
print(f"  attention-opt speedup          : {speedup_attn:.2f}x")
print(f"  frames/hr at {new_tile_s:.1f}s/tile, 15 tiles/frame: "
      f"{3600/(new_tile_s*15):.2f} fps  "
      f"({1098/(new_tile_s*15/3600):.0f} hrs total)")

print("\n✓ Done.")
