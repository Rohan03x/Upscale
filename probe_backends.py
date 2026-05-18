"""
probe_backends.py -- investigate cuDNN SDPA, TensorRT, Inductor, frame-similarity options
"""
import torch, warnings, time
warnings.filterwarnings('ignore')
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

scale = 30**-0.5

def bench(fn, n=50):
    with torch.no_grad():
        for _ in range(10): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

# ── A: cuDNN vs mem_efficient (HAB self-attn and OCAB cross-attn) ─────────────
print("=" * 65)
print("A: cuDNN SDPA vs mem_efficient — HAB and OCAB")
print("=" * 65)

B_, nH, hd = 676, 6, 32  # hd=32 after our padding patch

# HAB: nq = nk = 256
N = 256
q_h  = torch.randn(B_, nH, N,   hd, device='cuda', dtype=torch.float16)
k_h  = torch.randn(B_, nH, N,   hd, device='cuda', dtype=torch.float16)
v_h  = torch.randn(B_, nH, N,   hd, device='cuda', dtype=torch.float16)
b_h  = torch.randn(1,  nH, N,   N,  device='cuda', dtype=torch.float16) * 0.1

# OCAB: nq=256, nk=576
nq, nk = 256, 576
q_o  = torch.randn(B_, nH, nq, hd, device='cuda', dtype=torch.float16)
k_o  = torch.randn(B_, nH, nk, hd, device='cuda', dtype=torch.float16)
v_o  = torch.randn(B_, nH, nk, hd, device='cuda', dtype=torch.float16)
b_o  = torch.randn(1,  nH, nq, nk, device='cuda', dtype=torch.float16) * 0.1

results = {}
for bk, name in [(SDPBackend.EFFICIENT_ATTENTION, 'mem_eff'),
                 (SDPBackend.CUDNN_ATTENTION,      'cudnn  ')]:
    try:
        with sdpa_kernel(bk):
            h_ms = bench(lambda: F.scaled_dot_product_attention(q_h, k_h, v_h, attn_mask=b_h, scale=scale))
        with sdpa_kernel(bk):
            o_ms = bench(lambda: F.scaled_dot_product_attention(q_o, k_o, v_o, attn_mask=b_o, scale=scale))
        results[name.strip()] = (h_ms, o_ms)
        print(f"  {name} — HAB(256x256): {h_ms:.2f} ms    OCAB(256x576): {o_ms:.2f} ms")
    except Exception as e:
        print(f"  {name} — FAILED: {e}")

if 'mem_eff' in results and 'cudnn' in results:
    me_h, me_o = results['mem_eff']
    cu_h, cu_o = results['cudnn']
    # 36 HABs + 6 OCABs after our OCAB fix (so OCAB uses padded hd=32 and mem_eff/cudnn)
    save_ms = 36*(me_h - cu_h) + 6*(me_o - cu_o)
    print()
    print(f"  Speedup HAB:    {me_h/cu_h:.3f}x  ({36*(me_h-cu_h):.0f} ms saving x36 HABs)")
    print(f"  Speedup OCAB:   {me_o/cu_o:.3f}x  ({6*(me_o-cu_o):.0f} ms saving x6 OCABs)")
    print(f"  TOTAL cuDNN saving (after OCAB patch): {save_ms:.0f} ms/tile")
    print(f"  (HAB was 173ms, OCAB will be ~143ms -> cuDNN would make them ~{36*cu_h+6*cu_o:.0f}ms)")
    compiled_baseline = 5503
    pct = save_ms / compiled_baseline * 100
    print(f"  As % of compiled 5503ms baseline: {pct:.1f}%")
    print(f"  NOTE: auto-select already picks cuDNN when available; explicit forcing may not help.")

# ── B: TensorRT / ONNX availability ──────────────────────────────────────────
print()
print("=" * 65)
print("B: TensorRT / ONNX Runtime availability")
print("=" * 65)
for pkg in ['tensorrt', 'torch_tensorrt']:
    try:
        m = __import__(pkg)
        ver = getattr(m, '__version__', '?')
        print(f"  {pkg}: AVAILABLE  version={ver}")
    except ImportError:
        print(f"  {pkg}: not installed")

try:
    import onnx, onnxruntime as ort
    print(f"  onnx: {onnx.__version__}")
    print(f"  onnxruntime: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"  ORT providers: {providers}")
    if 'TensorrtExecutionProvider' in providers:
        print("    -> TensorRT execution provider AVAILABLE in ORT!")
    elif 'CUDAExecutionProvider' in providers:
        print("    -> CUDA execution provider available (ORT-GPU, not TRT)")
except ImportError as e:
    print(f"  onnx/ort: {e}")

# ── C: torch._inductor config knobs ──────────────────────────────────────────
print()
print("=" * 65)
print("C: torch._inductor config — relevant tuning knobs")
print("=" * 65)
try:
    import torch._inductor.config as ic
    for attr in ['max_autotune', 'coordinate_descent_tuning',
                 'shape_padding', 'epilogue_fusion',
                 'triton_unique_kernel_names', 'benchmark_kernel',
                 'compile_threads', 'cuda_backend', 'triton']:
        val = getattr(ic, attr, 'N/A')
        print(f"  {attr}: {val!r}")
except Exception as e:
    print(f"  Inductor config probe failed: {e}")

# ── D: Frame similarity — measure pixel diff between consecutive frames ───────
print()
print("=" * 65)
print("D: Frame similarity analysis (temporal redundancy)")
print("=" * 65)
import os, struct, subprocess

input_video = "input/14238437_1080_1920_30fps.mp4"
ffmpeg = "tools/ffmpeg/ffmpeg-8.1.1-full_build/bin/ffmpeg.exe"
if os.path.exists(input_video) and os.path.exists(ffmpeg):
    # Extract 10 consecutive frames, compute mean absolute difference
    try:
        import numpy as np, tempfile, glob

        tmpdir = "temp_probe"
        os.makedirs(tmpdir, exist_ok=True)

        # Extract frames 100-109 (avoid black start)
        cmd = [ffmpeg, '-y', '-ss', '5', '-i', input_video,
               '-vframes', '20', '-pix_fmt', 'rgb24',
               '-f', 'rawvideo', '-', '-loglevel', 'quiet']
        raw = subprocess.check_output(cmd, timeout=30)
        # 1080x1920 RGB24 = 6,220,800 bytes per frame
        frame_bytes = 1080 * 1920 * 3
        n_frames = len(raw) // frame_bytes
        print(f"  Extracted {n_frames} frames at 5 sec mark")

        diffs = []
        for i in range(n_frames - 1):
            f1 = np.frombuffer(raw[i*frame_bytes:(i+1)*frame_bytes], dtype=np.uint8).astype(np.float32)
            f2 = np.frombuffer(raw[(i+1)*frame_bytes:(i+2)*frame_bytes], dtype=np.uint8).astype(np.float32)
            mad = np.mean(np.abs(f1 - f2))
            diffs.append(mad)

        diffs = np.array(diffs)
        print(f"  Frame-to-frame MAD (Mean Abs Diff, 0-255 scale):")
        print(f"    min:    {diffs.min():.3f}")
        print(f"    median: {np.median(diffs):.3f}")
        print(f"    max:    {diffs.max():.3f}")
        print(f"    > 5.0 (significant motion):  {(diffs > 5.0).sum()}/{len(diffs)} frames")
        print(f"    > 2.0 (noticeable change):   {(diffs > 2.0).sum()}/{len(diffs)} frames")
        print(f"    < 2.0 (very similar frames): {(diffs < 2.0).sum()}/{len(diffs)} frames")

        skip_threshold = 2.0
        skippable = (diffs < skip_threshold).sum()
        skip_pct = skippable / len(diffs) * 100
        print()
        print(f"  At threshold MAD < {skip_threshold}:")
        print(f"    Frames that could be interpolated: {skippable}/{len(diffs)} ({skip_pct:.1f}%)")
        # Full video estimate
        total_frames = 1098
        est_skip = int(total_frames * skip_pct / 100)
        est_remain = total_frames - est_skip
        cur_hrs = total_frames * 82.5 / 3600
        new_hrs = est_remain * 82.5 / 3600 + est_skip * 0.1 / 3600  # interp ~0.1s
        print(f"    Full video: ~{est_skip} frames skipped, ~{est_remain} upscaled")
        print(f"    Current: {cur_hrs:.1f} hrs  After skip: {new_hrs:.1f} hrs  Speedup: {cur_hrs/new_hrs:.2f}x")
        print("    Quality risk: interpolated frames may have temporal artifacts for fast motion")

        # Also estimate at the input video frame rate (was 30fps, RIFE doubled to 60fps)
        print()
        print("  NOTE: Input is 30fps->60fps (after RIFE 2x). Half of 60fps frames are RIFE-")
        print("  interpolated and already 'between' real frames — naturally lower MAD.")
        print("  These RIFE frames are prime candidates for temporal skip or downgraded upscaler.")

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        print(f"  Frame analysis failed: {e}")
else:
    print(f"  Input video not found: {input_video}")

print("\nDone.")
