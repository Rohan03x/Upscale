#!/usr/bin/env python3
"""
Video upscale pipeline — RunPod L40S (Ubuntu, 48GB VRAM, 128 vCPU)
Stages:
  1. FFmpeg  — deblock + denoise
  2. (skip)
  3. Real-ESRGAN — 4x GPU upscale (tile=0, FP16, full 48 GB)
  4. FFmpeg  — sharpen + colour correct
  5. RIFE    — 2x FPS interpolation (ncnn-vulkan)
  Final:      HEVC encode (hevc_nvenc -> libx265 fallback)
"""
import argparse, subprocess, sys, shutil, os
from pathlib import Path
from datetime import timedelta

# ---------------------------------------------------------------------------
BASE        = Path("/workspace")
ESRGAN_SCRIPT = BASE / "Real-ESRGAN/inference_realesrgan_video.py"
MODEL_SRC   = BASE / "RealESRGAN_x4plus.pth"
WEIGHTS_DIR = BASE / "Real-ESRGAN/weights"
RIFE_BIN    = BASE / "rife-ncnn-vulkan/rife-ncnn-vulkan"
RIFE_MODEL  = BASE / "rife-ncnn-vulkan/rife-v4.6"
OUTPUT_DIR  = BASE / "output"
TEMP_DIR    = Path("/dev/shm/upscale_temp")  # 88 GB RAM-backed; keeps /workspace free

# Denoise presets  (hqdn3d luma_s:chroma_s:luma_t:chroma_t)
DENOISE_PRESETS = {
    "low":    "hqdn3d=1.5:1:2.5:2",
    "medium": "hqdn3d=3.5:2.5:5.5:4",
}
COLOUR_CORRECT = "eq=contrast=1.08:brightness=0.01:saturation=1.15:gamma=0.95"
SHARPEN_LOW    = "unsharp=5:5:0.35:3:3:0.0"
BATCH_SIZE     = 4   # frames batched into one GPU call; increase to 8 if VRAM allows

# ---------------------------------------------------------------------------
def run(cmd, label, eta_hint=None):
    print(f"\n{'='*65}")
    print(f"  {label}")
    if eta_hint:
        print(f"  ETA: {eta_hint}")
    print(f"{'='*65}\n")
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {label}")

def ffprobe_val(path, stream, entry):
    return subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", stream,
        "-show_entries", f"stream={entry}",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ]).decode().strip()

def get_res(path):
    w = int(ffprobe_val(path, "v:0", "width"))
    h = int(ffprobe_val(path, "v:0", "height"))
    return w, h

def get_duration(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ]).decode().strip()
    return float(out)

def get_fps(path):
    raw = ffprobe_val(path, "v:0", "r_frame_rate")
    num, den = raw.split("/")
    return float(num) / float(den)

# ---------------------------------------------------------------------------
def stage_denoise(src, dst, preset):
    """Stage 1: deblock + temporal denoise.  Uses all CPU threads via -threads."""
    vf = f"pp=hb/vb/dr/al,{DENOISE_PRESETS[preset]}"
    run([
        "ffmpeg", "-y",
        "-threads", "0",          # auto = all 128 vCPU
        "-i", src,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "17",
        "-threads", "0",
        "-c:a", "copy", dst
    ], "Stage 1/5  Deblock + Denoise (128 vCPU)")

# ---------------------------------------------------------------------------
def stage_downscale(src, dst):
    """Stage 2: halve resolution before ESRGAN — 4× fewer pixels = ~4× faster GPU.
       960×414 → x4 ESRGAN → 3840×1656 (4K).  Lanczos keeps sharpness."""
    w, h = get_res(src)
    new_w = (w // 2) & ~1   # halve + ensure even (required by yuv420p)
    new_h = (h // 2) & ~1
    run([
        "ffmpeg", "-y",
        "-threads", "0",
        "-i", src,
        "-vf", f"scale={new_w}:{new_h}:flags=lanczos",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "17",
        "-threads", "0",
        "-c:a", "copy", dst
    ], f"Stage 2/5  Pre-downscale {w}×{h} → {new_w}×{new_h} for ESRGAN")

# ---------------------------------------------------------------------------
def stage_upscale(src, dst, scale, dur_s=0, tile=0):
    """Stage 3: Real-ESRGAN 4x on L40S — tile=0 uses the full 46 GB VRAM."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model_dst = WEIGHTS_DIR / "RealESRGAN_x4plus.pth"
    if not model_dst.exists():
        shutil.copy2(MODEL_SRC, model_dst)

    w, h = get_res(src)
    print(f"\n  Input:  {w}x{h}  ->  Output: {w*scale}x{h*scale}")
    eta = str(timedelta(seconds=int(dur_s * 24 / 6))) if dur_s else None   # ~6 fps on L40S

    out_dir = dst.parent
    stem    = Path(src).stem
    suffix  = "esrgan"
    expected = out_dir / f"{stem}_{suffix}.mp4"

    run([
        "python3", ESRGAN_SCRIPT,
        "-i", src,
        "-o", out_dir,
        "-n", "RealESRGAN_x4plus",
        "-s", str(scale),
        "--suffix", suffix,
        "--tile",     str(tile),
        "--tile_pad", "0" if tile == 0 else "16",
        "--pre_pad",  "0",
        "--ffmpeg_bin", "ffmpeg",
    ], f"Stage 3/5  Real-ESRGAN {scale}x  (L40S / tile={'off' if tile==0 else tile} / FP16)", eta_hint=eta)

    if expected.exists() and expected != dst:
        expected.rename(dst)

# ---------------------------------------------------------------------------
def stage_post(src, dst):
    """Stage 4: sharpen then colour-correct on CPU (runs fast at 7680p)."""
    vf = f"{SHARPEN_LOW},{COLOUR_CORRECT}"
    run([
        "ffmpeg", "-y",
        "-threads", "0",
        "-i", src,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "17",
        "-threads", "0",
        "-c:a", "copy", dst
    ], "Stage 4/5  Sharpen + Colour Correct")

# ---------------------------------------------------------------------------
def stage_rife(src, dst, fps_in):
    """Stage 5: RIFE 2x FPS via rife-ncnn-vulkan."""
    if not RIFE_BIN.exists():
        print(f"\n  [RIFE] Binary not found at {RIFE_BIN} — skipping FPS step")
        shutil.copy2(src, dst)
        return

    frames_in  = TEMP_DIR / "rife_in"
    frames_out = TEMP_DIR / "rife_out"
    frames_in.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)

    run(["ffmpeg", "-y", "-threads", "0", "-i", src,
         "-qscale:v", "1", f"{frames_in}/%08d.png"],
        "Stage 5a Extract frames for RIFE")

    run([RIFE_BIN,
         "-i", frames_in, "-o", frames_out,
         "-m", RIFE_MODEL,
         "-j", "4:4:4",         # threads: input:process:output (default N*2 = 2x)
         "-g", "0"],            # GPU 0
        "Stage 5b RIFE 2x interpolation (GPU)")

    fps_out = fps_in * 2
    run([
        "ffmpeg", "-y",
        "-framerate", str(fps_out),
        "-i", f"{frames_out}/%08d.png",
        "-i", src, "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "17",
        "-threads", "0",
        "-c:a", "copy", dst
    ], "Stage 5c Reassemble RIFE frames")

    shutil.rmtree(frames_in, ignore_errors=True)
    shutil.rmtree(frames_out, ignore_errors=True)

# ---------------------------------------------------------------------------
def stage_encode(src, dst):
    """Final encode: HEVC with NVENC (cq=18) then libx265 fallback."""
    for codec, extra in [
        ("hevc_nvenc", [
            "-preset", "p7", "-tune", "hq",
            "-rc", "vbr", "-cq", "18",
            "-b:v", "0", "-maxrate", "120M", "-bufsize", "240M",
            "-spatial-aq", "1", "-temporal-aq", "1",
        ]),
        ("libx265", ["-preset", "slow", "-crf", "18"]),
    ]:
        try:
            run([
                "ffmpeg", "-y",
                "-threads", "0",
                "-i", src,
                "-c:v", codec, *extra,
                "-tag:v", "hvc1",
                "-c:a", "aac", "-b:a", "320k",
                dst
            ], f"Final Encode  HEVC ({codec})")
            return
        except RuntimeError as e:
            print(f"  [{codec}] failed — trying fallback…")
    raise RuntimeError("All encoders failed")

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Upscale pipeline for RunPod L40S")
    parser.add_argument("--input",   required=True,  help="Input video path")
    parser.add_argument("--scale",   type=int, default=4)
    parser.add_argument("--denoise", choices=["low", "medium", "none"], default="medium")
    parser.add_argument("--tile",    type=int, default=0,
                        help="Real-ESRGAN tile size. 0 = no tiling (uses full 46 GB VRAM)")
    parser.add_argument("--skip-rife", action="store_true", help="Skip RIFE FPS step")
    args = parser.parse_args()

    src  = Path(args.input).resolve()
    stem = src.stem
    work = TEMP_DIR / stem
    work.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dur = get_duration(src)
    fps = get_fps(src)
    print(f"\n  Source : {src.name}")
    print(f"  Duration: {timedelta(seconds=int(dur))}")
    print(f"  FPS    : {fps:.3f}")

    f_denoised   = work / f"{stem}_s1_denoised.mkv"
    f_downscaled = work / f"{stem}_s2_downscaled.mkv"
    f_upscaled   = work / f"{stem}_s3_upscaled.mkv"
    f_post       = work / f"{stem}_s4_post.mkv"
    f_rife       = work / f"{stem}_s5_rife.mkv"
    final_out    = OUTPUT_DIR / f"{stem}_4K_upscaled.mp4"

    if args.denoise != "none":
        stage_denoise(src, f_denoised, args.denoise)
    else:
        f_denoised = src

    stage_downscale(f_denoised, f_downscaled)
    # Free Stage 1 now that downscale is done
    if f_denoised != src and f_denoised.exists():
        f_denoised.unlink()
        print(f"  [cleanup] removed {f_denoised.name}")

    stage_upscale(f_downscaled, f_upscaled, args.scale, dur_s=dur, tile=args.tile)
    # Free Stage 2 intermediate
    if f_downscaled.exists():
        f_downscaled.unlink()
        print(f"  [cleanup] removed {f_downscaled.name}")

    stage_post(f_upscaled, f_post)
    # Free Stage 3 intermediate
    if f_upscaled.exists():
        f_upscaled.unlink()
        print(f"  [cleanup] removed {f_upscaled.name}")

    if not args.skip_rife:
        stage_rife(f_post, f_rife, fps)
        # Free Stage 4 intermediate
        if f_post.exists():
            f_post.unlink()
            print(f"  [cleanup] removed {f_post.name}")
    else:
        f_rife = f_post

    stage_encode(f_rife, final_out)
    # Free last intermediate
    if f_rife.exists() and f_rife != final_out:
        f_rife.unlink()
        print(f"  [cleanup] removed {f_rife.name}")

    print(f"\n{'='*65}")
    print(f"  DONE —> {final_out}")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()
