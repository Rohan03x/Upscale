"""
profile_current.py  –  profile the PATCHED model to find remaining bottlenecks.
Shows where time is STILL going after head_dim pad + broadcast fixes.
Run: python profile_current.py
"""
import os, sys, time, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
sys.path.insert(0, "tools/Real-ESRGAN")

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

print("Loading HAT model …")
from basicsr.archs.hat_arch import HAT, OCAB
model = HAT(
    upscale=4, in_chans=3, img_size=64, window_size=16,
    compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
    overlap_ratio=0.5, img_range=1., depths=[6,6,6,6,6,6],
    embed_dim=180, num_heads=[6,6,6,6,6,6], mlp_ratio=2,
    upsampler="pixelshuffle", resi_connection="1conv",
).cuda().half().eval()

wts = "tools/Real-ESRGAN/weights/Real_HAT_GAN_SRx4_sharper.pth"
sd = torch.load(wts, map_location="cuda", weights_only=True)
sd = sd.get("params_ema", sd.get("params", sd))
model.load_state_dict(sd, strict=False)
print("  Weights loaded.")

# Check OCAB head_dim
print("\nOCAB architecture check:")
ocab = None
for m in model.modules():
    if isinstance(m, OCAB):
        ocab = m
        break
if ocab:
    d = ocab.dim // ocab.num_heads
    print(f"  dim={ocab.dim}, num_heads={ocab.num_heads}")
    print(f"  head_dim = {d}  ({'PROBLEM: % 8 != 0' if d % 8 != 0 else 'OK: % 8 == 0'})")
    print(f"  window_size={ocab.window_size}, overlap_win_size={ocab.overlap_win_size}")
    nq = ocab.window_size ** 2
    nk = ocab.overlap_win_size ** 2
    print(f"  Q seq_len={nq}, K/V seq_len={nk}  (cross-attention, nk/nq ratio = {nk/nq:.2f}x)")
    print(f"  SDPA cost vs HAB: {nk/nq:.2f}x larger K/V sequence")

# Count OCAB and HAB instances
from basicsr.archs.hat_arch import WindowAttention
n_win_attn = sum(1 for m in model.modules() if isinstance(m, WindowAttention))
n_ocab     = sum(1 for m in model.modules() if isinstance(m, OCAB))
print(f"\n  WindowAttention (HAB) instances: {n_win_attn}")
print(f"  OCAB instances              : {n_ocab}")

tile = torch.rand(1, 3, 416, 416, device="cuda", dtype=torch.float16)

# Warm up
print("\nWarm-up …")
with torch.no_grad():
    for _ in range(2):
        model(tile)
        torch.cuda.synchronize()

# Profile
print("Profiling …")
with torch.no_grad():
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_flops=False,
        profile_memory=False,
    ) as prof:
        model(tile)
    torch.cuda.synchronize()

# Print top ops
print("\nTop 30 CUDA ops (current patched model, FP16, tile=416x416):")
print(f"  {'Op':<55} {'CUDA%':>7}  {'CUDA_ms':>9}")
print("  " + "-" * 75)

events = prof.key_averages()
total_cuda = sum(e.self_device_time_total for e in events)
rows = sorted(events, key=lambda e: e.self_device_time_total, reverse=True)[:30]
for e in rows:
    pct = e.self_device_time_total / total_cuda * 100
    ms  = e.self_device_time_total / 1000
    print(f"  {e.key:<55} {pct:>7.1f}%  {ms:>9.1f} ms")

print(f"\n  Total CUDA time: {total_cuda/1000:.1f} ms")

# Separate SDPA backends
print("\nSDPA backend breakdown:")
sdpa_math    = sum(e.self_device_time_total for e in events if 'math' in e.key.lower())
sdpa_memeff  = sum(e.self_device_time_total for e in events if 'efficient' in e.key.lower() or 'mem_eff' in e.key.lower())
sdpa_total   = sum(e.self_device_time_total for e in events if 'scaled_dot_product' in e.key.lower())
print(f"  math_sdp (slow path)    : {sdpa_math/1000:6.1f} ms  ({sdpa_math/total_cuda*100:.1f}%)")
print(f"  mem_efficient_sdp       : {sdpa_memeff/1000:6.1f} ms  ({sdpa_memeff/total_cuda*100:.1f}%)")
print(f"  total SDPA              : {sdpa_total/1000:6.1f} ms  ({sdpa_total/total_cuda*100:.1f}%)")

# Check im2col (OCAB unfold)
im2col = sum(e.self_device_time_total for e in events if 'im2col' in e.key.lower() or 'unfold' in e.key.lower() or 'col2im' in e.key.lower())
print(f"  im2col / unfold (OCAB)  : {im2col/1000:6.1f} ms  ({im2col/total_cuda*100:.1f}%)")

print("\n✓ Done.")
