#!/usr/bin/env python
"""
EDVR temporal video restoration — 5-frame sliding window.
==========================================================
Uses deformable convolution to align and fuse 5 neighbouring frames,
recovering temporally-coherent detail that single-frame models miss.
Particularly effective on compressed video where different macroblocks
were preserved/lost inconsistently across frames.

~8fps at 1920x1080 on RTX 5090 (FP16).  Runs after NAFNet deblur,
before stabilize and HAT upscale.

Model download (~28 MB):
  https://github.com/XPixelGroup/BasicSR/releases/download/V1.3.5/EDVR_M_deblur_REDS_official-d4ad3da9.pth
  -> rename to EDVR-deblur.pth
  -> save to: C:/VideoUpscale/models/EDVR-deblur.pth

Requires:
  pip install basicsr

Usage (called by upscale.py automatically):
  python tools/edvr_restore.py --input video.mkv --output restored.mkv
                                --model models/EDVR-deblur.pth
                                --ffmpeg path/to/ffmpeg.exe
"""

import argparse
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from basicsr.archs.edvr_arch import EDVR

NUM_FRAME = 5          # EDVR window size — process 5 frames at once
CENTER    = NUM_FRAME // 2   # index 2 = middle frame is the output frame


def load_model(model_path: str) -> torch.nn.Module:
    """
    Load EDVR_L_deblur_REDS onto GPU in FP16.
    Architecture: L (large) deblur variant with pre-deblur module.
      num_feat=128, num_reconstruct_block=40, hr_in=True, with_predeblur=True.
    """
    torch.backends.cudnn.benchmark    = True   # finds fastest conv algorithm per fixed input shape
    # TF32: covers FP32 accumulation paths in deformable conv — zero quality impact on FP16 inference
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    model = EDVR(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=128,
        num_frame=NUM_FRAME,
        deformable_groups=8,
        num_extract_block=5,
        num_reconstruct_block=40,
        center_frame_idx=None,
        hr_in=True,
        with_predeblur=True,
        with_tsa=True,
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    weights = (checkpoint.get("params_ema")
               or checkpoint.get("params")
               or checkpoint)
    model.load_state_dict(weights, strict=True)
    model.eval()
    model = model.cuda().half()
    # FP8 dynamic quantisation for Ada/Hopper/Blackwell (sm_89+).
    # torchao quantises nn.Conv2d + nn.Linear weights; deformable conv stays FP16 (unsupported).
    # ~1.5x on Ada, ~2x on Blackwell.  Quality: <0.2 dB PSNR for temporal restoration.
    _cc = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    if _cc and (_cc.major, _cc.minor) >= (8, 9):
        try:
            from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
            quantize_(model, Float8DynamicActivationFloat8WeightConfig())
            print(f"  EDVR: FP8 dynamic quantisation applied (sm_{_cc.major}{_cc.minor})", flush=True)
        except Exception as _fp8e:
            print(f"  EDVR: FP8 unavailable \u2014 {type(_fp8e).__name__} (install torchao)", flush=True)
    try:
        import triton as _triton  # noqa
        # Disable Inductor autotune on <12 GB VRAM — after ~500 frames of EDVR_L the allocator
        # is fragmented and the benchmark-cache allocation (triton.testing.do_bench) OOMs.
        # On high-VRAM cards autotune is kept (finds better Triton configs, worth the cost).
        import os as _os
        _edvr_avail_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        if _edvr_avail_gb < 12:
            _os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "0")
            _os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "0")
            print(f"  EDVR: autotune disabled ({_edvr_avail_gb:.0f} GB VRAM < 12 GB threshold)", flush=True)
        import torch._inductor.config as _ic
        import torch._dynamo as _dynamo
        _ic.coordinate_descent_tuning = True
        # EDVR's DCNv2 has a data-dependent `if offset_absmean > 50:` branch that
        # causes InternalTorchDynamoError in max-autotune/dynamic=False mode.
        # Use "default" + dynamic=True so Dynamo handles the graph break gracefully.
        _dynamo.config.suppress_errors = True
        model = torch.compile(model, mode="default", dynamic=True)
        print("  EDVR: torch.compile(default, CDT, suppress_errors) enabled", flush=True)
    except Exception as _ce:
        print(f"  EDVR: running in eager mode ({type(_ce).__name__})", flush=True)
    return model


@torch.no_grad()
def restore_batch(model: torch.nn.Module, windows: list) -> list:
    """
    Restore center frames for a batch of 5-frame windows in one GPU call.
    windows: list of B windows, each = list of 5 BGR uint8 frames.
    Returns: list of B restored BGR uint8 center frames.

    GPU-side preprocessing: H2D as compact uint8, channel flip + normalise
    fused on GPU.  Batching B windows cuts per-frame GPU launch overhead by B.
    B is VRAM-adaptive (EDVR_BATCH).
    """
    h_orig, w_orig = windows[0][0].shape[:2]
    h_pad = (h_orig + 15) // 16 * 16
    w_pad = (w_orig + 15) // 16 * 16
    pad_h = h_pad - h_orig
    pad_w = w_pad - w_orig

    batch_frames = []
    for window in windows:
        if pad_h > 0 or pad_w > 0:
            window = [
                cv2.copyMakeBorder(f, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                for f in window
            ]
        batch_frames.append(np.stack(window))  # [5, H_pad, W_pad, 3] uint8

    # ── H2D as uint8 [B, 5, H_pad, W_pad, 3] ──────────────────────────────────
    raw = np.stack(batch_frames)
    t = torch.from_numpy(raw).cuda(non_blocking=True)
    # GPU: BHWC → BCTHW (EDVR convention), uint8 → FP16, /255
    inp = t.permute(0, 1, 4, 2, 3).to(torch.float16).mul_(1.0 / 255.0)  # [B, 5, 3, H_pad, W_pad]

    out = model(inp)  # [B, 3, H_pad, W_pad]

    # ── D2H as uint8 ─────────────────────────────────────────────────────────
    out_u8 = (out[:, :, :h_orig, :w_orig]
                .float().clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8))  # [B, 3, H, W]
    out_bgr = out_u8[:, [2, 1, 0]].permute(0, 2, 3, 1).contiguous()  # [B, H, W, 3] BGR
    return list(out_bgr.cpu().numpy())


def restore_center(model: torch.nn.Module, window: list) -> np.ndarray:
    """Single-window wrapper kept for API compatibility."""
    return restore_batch(model, [window])[0]


# VRAM-adaptive batch: EDVR_L at 1920×1080 FP16 peaks ~8 GB per window (activations + deform conv)
# 8 GB → 1 (safe, current behaviour)  |  32 GB → 4  |  48 GB → 6  |  80 GB → 8
_edvr_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
EDVR_BATCH = max(1, min(8, int(_edvr_vram_gb / 8)))


def probe_video(ffmpeg_bin: str, video_path: str):
    """Return (fps, w, h, total_frames) using cv2 with ffprobe fallback.
    cv2 fails to read metadata from hevc_nvenc MKV files (returns 0/0/0).
    """
    cap   = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0.0 or w == 0 or h == 0:
        # ffprobe lives in the same dir as ffmpeg; replace only the filename
        p = Path(ffmpeg_bin)
        ffprobe = str(p.parent / p.name.replace('ffmpeg', 'ffprobe'))
        try:
            r = subprocess.run([
                ffprobe, '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate,width,height,nb_frames',
                '-of', 'csv=p=0', str(video_path)
            ], capture_output=True, text=True)
            if r.stdout.strip():
                parts = r.stdout.strip().split(',')
                # ffprobe csv output order: width,height,r_frame_rate,nb_frames
                if len(parts) >= 3:
                    if w == 0:
                        w = int(parts[0])
                    if h == 0:
                        h = int(parts[1])
                    if fps <= 0.0 and '/' in parts[2]:
                        num, den = parts[2].split('/')
                        fps = int(num) / max(int(den), 1)
                    if total == 0 and len(parts) > 3 and parts[3].strip().isdigit():
                        total = int(parts[3])
            elif r.returncode != 0:
                print(f"  WARNING: ffprobe exit {r.returncode}: {r.stderr.strip()[:200]}", flush=True)
        except Exception as e:
            print(f"  WARNING: ffprobe fallback failed: {e}", flush=True)
        if fps <= 0.0:
            fps = 30.0
        print(f"  EDVR: cv2 returned bad metadata; ffprobe resolved {w}x{h}@{fps}fps", flush=True)
    return fps, w, h, total


def main() -> None:
    ap = argparse.ArgumentParser(description="EDVR temporal video restoration")
    ap.add_argument("--input",  required=True,  help="Input video path")
    ap.add_argument("--output", required=True,  help="Output video path")
    ap.add_argument("--model",  required=True,  help="EDVR-deblur.pth path")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg binary")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    mdl = Path(args.model)

    if not src.exists():
        sys.exit(f"ERROR: input not found: {src}")
    if not mdl.exists():
        sys.exit(
            f"ERROR: model not found: {mdl}\n"
            "       Download via gdown:\n"
            "       gdown 1_ma2tgHscZtkIY2tEJkVdU-UP8bnqBRE -O models/EDVR-deblur.pth"
        )

    print(f"  EDVR: loading {mdl.name} onto GPU (FP16) ...", flush=True)
    model = load_model(str(mdl))
    print("  EDVR: model ready", flush=True)

    fps, w, h, total = probe_video(args.ffmpeg, str(src))

    cap = cv2.VideoCapture(str(src))

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Use libx264 lossless (CRF=0) for intermediate file — cv2 reliably reads
    # H264/MKV; hevc_nvenc MKV buries Cues at EOF causing cv2 to return w=0/h=0.
    ffmpeg_cmd = [
        args.ffmpeg, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-i", str(src),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-c:a", "copy",
        str(dst),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # ── Sliding window strategy ───────────────────────────────────────────────
    # Read ahead to fill a 5-frame window.  For boundary frames, replicate the
    # edge frame (standard reflection padding for temporal models).
    buf: deque = deque(maxlen=NUM_FRAME)  # rolling 5-frame buffer
    lookahead   = []                       # frames read ahead for next2

    # Pre-read first 4 frames into buffer (replicate first frame for prev2/prev1)
    seed = None
    for _ in range(NUM_FRAME - 1):
        ok, f = cap.read()
        if ok:
            if seed is None:
                seed = f
            buf.append(f)
        elif seed is not None:
            buf.append(seed.copy())

    if not buf:
        cap.release()
        proc.stdin.close()
        proc.wait()
        sys.exit("ERROR: video has no frames")

    # Pad start: replicate first frame into prev2 / prev1 positions
    while len(buf) < NUM_FRAME - 1:
        buf.appendleft(buf[0].copy())

    _window_batch: list = []
    processed = 0

    def _flush_batch() -> None:
        """Run restore_batch for accumulated windows, write output frames."""
        nonlocal processed
        if not _window_batch:
            return
        for out_frame in restore_batch(model, _window_batch):
            proc.stdin.write(out_frame.tobytes())
        processed += len(_window_batch)
        _window_batch.clear()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # End of video: replicate last frame for the tail windows
                last = buf[-1].copy()
                for _ in range(CENTER + 1):
                    buf.append(last)
                    _window_batch.append(list(buf))
                _flush_batch()
                break

            buf.append(frame)
            if len(buf) == NUM_FRAME:
                _window_batch.append(list(buf))
                if len(_window_batch) >= EDVR_BATCH:
                    _flush_batch()
                    if processed % 500 == 0:
                        pct = processed / max(total, 1) * 100
                        print(f"  EDVR: {processed}/{total} frames  ({pct:.1f}%)", flush=True)
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()

    print(f"  EDVR: done — {processed} frames -> {dst}", flush=True)


if __name__ == "__main__":
    main()
