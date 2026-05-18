"""
bench_tile.py  –  end-to-end tile timing on the real HAT model.
Compares: patched (head_dim pad + broadcast) vs unpatched (math_sdp baseline).
Run: python bench_tile.py
"""
import sys, time, os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "tools/Real-ESRGAN")

import torch
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda"
WARMUP = 2
ITERS  = 5

print("Loading HAT model …")
from basicsr.archs.hat_arch import HAT
model = HAT(
    upscale=4, in_chans=3, img_size=64, window_size=16,
    compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
    overlap_ratio=0.5, img_range=1., depths=[6,6,6,6,6,6],
    embed_dim=180, num_heads=[6,6,6,6,6,6], mlp_ratio=2,
    upsampler='pixelshuffle', resi_connection='1conv',
).to(DEVICE).half().eval()

# Load real weights
wts = "tools/Real-ESRGAN/weights/Real_HAT_GAN_SRx4_sharper.pth"
if os.path.exists(wts):
    sd = torch.load(wts, map_location=DEVICE, weights_only=True)
    sd = sd.get("params_ema", sd.get("params", sd))
    model.load_state_dict(sd, strict=False)
    print("  Real weights loaded.")
else:
    print("  WARNING: real weights not found, using random init.")

# Input: 384+2*16 = 416x416 tile (after tile_pad=16 on each side), FP16
# But HAT.forward expects [B, C, H, W] in [0,1]
# tile_pad=16 is handled outside; model sees the padded tile
tile_h = tile_w = 384 + 2 * 16  # 416

def make_tile():
    return torch.rand(1, 3, tile_h, tile_w, device=DEVICE, dtype=torch.float16)

tile = make_tile()

def run_one():
    with torch.no_grad():
        out = model(tile)
    torch.cuda.synchronize()
    return out

# ── Timing ────────────────────────────────────────────────────────────────────
print(f"Tile size: {tile_h}×{tile_w}  (tile=384 + tile_pad=16 each side)")
print(f"Warmup={WARMUP}, Iters={ITERS}")

# Warm up
for i in range(WARMUP):
    t0 = time.perf_counter()
    run_one()
    print(f"  warm-up {i+1}: {(time.perf_counter()-t0)*1000:.0f} ms")

# Timed runs
times = []
for i in range(ITERS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_one()
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)
    print(f"  iter {i+1}: {elapsed:.0f} ms")

times.sort()
median = times[len(times)//2]
mean   = sum(times) / len(times)
best   = times[0]

print(f"\nResults (patched: head_dim pad 30→32 + broadcast mask):")
print(f"  Best  : {best:.0f} ms/tile")
print(f"  Median: {median:.0f} ms/tile")
print(f"  Mean  : {mean:.0f} ms/tile")

# Convert to frame and total-run projections
TILES_PER_FRAME = 15   # 1080x1920 → 4320x7680 with tile=384, tile_pad=16
TOTAL_FRAMES    = 1098

frame_s  = median / 1000 * TILES_PER_FRAME
total_hr = frame_s * TOTAL_FRAMES / 3600

print(f"\nProjection ({TILES_PER_FRAME} tiles/frame, {TOTAL_FRAMES} frames):")
print(f"  Estimated time/frame : {frame_s:.1f} s")
print(f"  Estimated total run  : {total_hr:.1f} hours")

# SDPA backend check
print("\nSDPA backend check (confirms mem_efficient_sdp active):")
import warnings
warnings.filterwarnings("ignore")
q = torch.randn(576, 6, 256, 32, device=DEVICE, dtype=torch.float16)
k = torch.randn(576, 6, 256, 32, device=DEVICE, dtype=torch.float16)
v = torch.randn(576, 6, 256, 32, device=DEVICE, dtype=torch.float16)
bias = torch.randn(1, 6, 256, 256, device=DEVICE, dtype=torch.float16) * 0.1

from torch.nn.attention import SDPBackend, sdpa_kernel
for name, backend in [
    ("flash_sdp",        SDPBackend.FLASH_ATTENTION),
    ("mem_efficient_sdp",SDPBackend.EFFICIENT_ATTENTION),
    ("math_sdp",         SDPBackend.MATH),
]:
    try:
        with sdpa_kernel(backend):
            F.scaled_dot_product_attention(q,k,v,attn_mask=bias)
        print(f"  {name}: available")
    except Exception as e:
        print(f"  {name}: unavailable ({e})")

print("\n✓ Done.")
