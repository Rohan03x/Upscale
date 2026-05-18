"""Script 1: Count Dynamo graph breaks using small input (64x64 - same code path, less VRAM)"""
import sys, os
sys.path.insert(0, 'C:/VideoUpscale/tools/Real-ESRGAN')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = 'C:/VideoUpscale/.inductor_cache'

import torch
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
print("Model loaded. Using 64x64 input for graph analysis.", flush=True)

# 64x64 uses 1 window (no shift-window, simple case), but same code path
# Use window_size multiple to get actual SW-MSA path: use 96x96 (6 windows)
x_small = torch.randn(1, 3, 96, 96, device='cuda', dtype=torch.float16)

explanation = torch._dynamo.explain(model)(x_small)
print(f"\nGraph count:   {explanation.graph_count}")
print(f"Ops per graph: {explanation.ops_per_graph}")
print(f"CONCLUSION: {'SINGLE GRAPH - no breaks' if explanation.graph_count == 1 else str(explanation.graph_count) + ' graphs = graph breaks exist'}")

if hasattr(explanation, 'break_reasons') and explanation.break_reasons:
    print(f"\nGraph break reasons ({len(explanation.break_reasons)} breaks):")
    seen = {}
    for r in explanation.break_reasons:
        key = str(r.reason)[:120]
        seen[key] = seen.get(key, 0) + 1
    for reason, count in sorted(seen.items(), key=lambda x: -x[1]):
        print(f"  x{count:2d}  {reason}")
else:
    print("\nNo graph breaks reported (or full graph compiled).")
