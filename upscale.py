"""
Video Upscale Pipeline  v4
==========================
Stage order (ALL ON by default):
  1. Deblock + Denoise       medium hqdn3d — cleans source compression
  2. Stabilize               2-pass vidstab smoothing=15 — removes camera shake
  3. Upscale                 Real-HAT 4x (best quality/speed for full movies)
                             STAR video diffusion opt-in via --generative-video (best quality, short clips only)
  4. CodeFormer face restore fidelity=0.7 — AI face enhancement
  5. Sharpen + Colour CC     unsharp 0.35 luma + contrast/saturation/gamma tweak
  6. RIFE 2x FPS             AI optical-flow interpolation (24→48 fps)
  7. Final HEVC NVENC        CQ14 multipass spatial+temporal AQ

Usage (simplest — all defaults):
    C:/VideoUpscale/venv/Scripts/python upscale.py --input "C:/path/to/movie.mkv"

All options:
    --input             path          Required. Input video
    --output            path          Output path (default: output/<name>_4x.mkv)
    --denoise           low|medium    Denoise strength (default: medium)
    --no-stabilize                    Disable video stabilization (default: ON)
    --no-rife                         Skip frame interpolation
    --scale             2|4           Upscale factor (default: 4)
    --tile              int           Tile size for HAT/ESRGAN (default: 0 = no tiling).
                                      0=no tiling (default, RTX 5090 32 GB — fastest, no seam artefacts).
                                      512=8 GB VRAM, 1024=12 GB.
    --model             name          Upscale model (default: Real_HAT_GAN_SRx4_sharper)
                                      Real_HAT_GAN_SRx4_sharper — best full-frame detail, all textures (default)
                                      RealESRGAN_x4plus          — fast GAN fallback, good live-action
                                      realesr-general-x4v3       — fastest; pair with --dn
    --dn                0.0-1.0       Denoising for realesr-general-x4v3 only (default: 0.5)
                                      0 = preserve all texture/grain, 1 = smooth aggressively
    --face-restore                    CodeFormer face restoration (default: ON). Use --no-face-restore to disable.
    --face-fidelity     float         CodeFormer fidelity 0-1 (default: 0.7)
                                      Lower = more enhancement, Higher = more faithful to input
    --generative-video                STAR video diffusion SR (default: OFF — impractical for full movies >5 min).
                                      Temporally consistent generative detail. ~0.4fps, best for short clips.
                                      Requires tools/STAR/pretrained_weight/model.pt (~5 GB from HuggingFace SherryX/STAR).
    --star-prompt       str           Text prompt for STAR scene description.
                                      Default: "A cinematic live-action film with detailed textures"
    --star-steps        int           STAR denoising steps (default: 15 fast). Use 50 for max quality.
    --star-chunk        int           STAR frames per denoising chunk (default: 64 for 32 GB). Reduce to 32/16 if OOM.
    --codec             hevc|av1      Output codec: hevc=H.265 NVENC CQ14 (default), av1=AV1 NVENC CQ28 (~30 pct smaller).
    --rife-exp          1|2           RIFE exponent: 1=2x FPS (default), 2=4x FPS (e.g. 24->96 fps).
    --rife-uhd                        RIFE UHD mode: halves optical-flow scale for <16 GB VRAM. Default OFF (32 GB not needed).
"""

import argparse
import json
import os

# Persist torch.compile / Triton kernel cache across runs.
# Without this, every run recompiles all three models (NAFNet, EDVR, HAT) from
# scratch, wasting 30-120 s per stage.  With the cache, only the first run pays
# the compile cost; subsequent runs load prebuilt kernels in < 1 s.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".torch_compile_cache"),
)
import subprocess
import sys
import shutil
import time
from datetime import timedelta
from pathlib import Path

# -- Paths -------------------------------------------------------------------
BASE           = Path(__file__).resolve().parent          # works on Windows + Linux
PYTHON         = Path(sys.executable)                     # same interpreter running this script
_ff_bundled    = BASE / "tools/ffmpeg/ffmpeg-8.1.1-full_build/bin/ffmpeg.exe"
_fp_bundled    = BASE / "tools/ffmpeg/ffmpeg-8.1.1-full_build/bin/ffprobe.exe"
FFMPEG_BIN     = _ff_bundled if _ff_bundled.exists() else Path(shutil.which("ffmpeg") or "ffmpeg")
FFPROBE_BIN    = _fp_bundled if _fp_bundled.exists() else Path(shutil.which("ffprobe") or "ffprobe")
FFMPEG         = FFMPEG_BIN
FFPROBE        = FFPROBE_BIN
RIFE_EXE       = BASE / "tools/rife-ncnn-vulkan-20221029-windows/rife-ncnn-vulkan.exe"
RIFE_MODEL_DIR = BASE / "tools/rife-ncnn-vulkan-20221029-windows/rife-v4.6"
RIFE_PY        = BASE / "tools/RIFE/inference_video.py"
RIFE_PY_MODEL  = BASE / "tools/RIFE/train_log"
REALESRGAN     = BASE / "tools/Real-ESRGAN/inference_realesrgan_video.py"
MODEL_PATH     = BASE / "models/RealESRGAN_x4plus.pth"
CODEFORMER_DIR = BASE / "tools/CodeFormer"
CODEFORMER_PY  = CODEFORMER_DIR / "inference_codeformer.py"
STAR_DIR       = BASE / "tools/STAR"
STAR_PY        = STAR_DIR / "video_super_resolution/scripts/inference_sr.py"
STAR_MODEL     = STAR_DIR / "pretrained_weight/model.pt"
NAFNET_SCRIPT  = BASE / "tools/nafnet_deblur.py"
NAFNET_MODEL   = BASE / "models/NAFNet-GoPro-width32.pth"
EDVR_SCRIPT    = BASE / "tools/edvr_restore.py"
EDVR_MODEL     = BASE / "models/EDVR-deblur.pth"
OUTPUT_DIR     = BASE / "output"

# -- Denoise presets ---------------------------------------------------------
# hqdn3d=luma_spatial:chroma_spatial:luma_temporal:chroma_temporal
DENOISE_PRESETS = {
    "low":    "hqdn3d=1.5:1:2.5:2",
    "medium": "hqdn3d=3.5:2.5:5.5:4",
}

# -- Colour correct (applied after upscale) ----------------------------------
# Mild boost suitable for faded/flat low-quality sources:
# contrast 1.08 = very slight lift
# saturation 1.15 = restore colour depth
# gamma 0.95 = slight midtone brightening for dark crushed blacks
COLOUR_CORRECT = "eq=contrast=1.08:brightness=0.01:saturation=1.15:gamma=0.95"

# -- Sharpen (applied after upscale, before colour correct) ------------------
# CAS (Contrast Adaptive Sharpening) — measures local contrast and only sharpens
# edges/detail, NOT flat regions or noise. Produces tighter edges than unsharp
# masking with less ringing. strength=0.4 is moderate; range 0.0-1.0.
SHARPEN_CAS = "cas=strength=0.4"


# -- NVENC capability check --------------------------------------------------
def _probe_nvenc(ffmpeg_bin) -> bool:
    """Return True if hevc_nvenc works on this system (driver + SDK version match)."""
    try:
        r = subprocess.run(
            [str(ffmpeg_bin), "-f", "lavfi", "-i", "testsrc=size=64x64:rate=1",
             "-vframes", "1", "-c:v", "hevc_nvenc", "-preset", "lossless",
             "-f", "null", "-"],
            capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def _build_venc_lossless(nvenc_ok: bool) -> list:
    """Return ffmpeg codec args for lossless intermediate files."""
    if nvenc_ok:
        return ["-c:v", "hevc_nvenc", "-preset", "lossless"]
    # libx264 lossless: -qp 0 with ultrafast — typically 200-400fps at 1080p on modern CPU
    return ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0"]


def _build_venc_final(nvenc_ok: bool) -> list:
    """Return ffmpeg codec args for the final high-quality output."""
    if nvenc_ok:
        return ["-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "hq",
                "-rc", "constqp", "-qp", "20", "-b:v", "0"]
    return ["-c:v", "libx265", "-preset", "faster", "-crf", "18",
            "-x265-params", "log-level=error"]


# Probe once at startup — fast (< 1s); result used everywhere hevc_nvenc is needed
_NVENC_OK = _probe_nvenc(FFMPEG_BIN)
_VENC_LOSSLESS = _build_venc_lossless(_NVENC_OK)
_VENC_FINAL    = _build_venc_final(_NVENC_OK)


# -- Helpers -----------------------------------------------------------------
_stage_start = None

def run(cmd, label, eta_hint=None, cwd=None):
    global _stage_start
    _stage_start = time.time()
    print(f"\n{'='*62}")
    print(f"  {label}")
    if eta_hint:
        print(f"  ETA: ~{eta_hint}")
    print(f"  Started: {time.strftime('%H:%M:%S')}")
    print(f"{'='*62}")
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)
    elapsed = time.time() - _stage_start
    print(f"\n  Done in {str(timedelta(seconds=int(elapsed)))}")


# -- Checkpoint helpers -------------------------------------------------------
_CP_FILE = "checkpoint.json"


def checkpoint_load(work_dir: Path) -> set:
    """Return set of completed stage keys from checkpoint file, or empty set."""
    cp = work_dir / _CP_FILE
    if cp.exists():
        try:
            return set(json.loads(cp.read_text()).get("completed", []))
        except Exception:
            pass
    return set()


def checkpoint_save(work_dir: Path, completed: set) -> None:
    """Write completed stage keys to checkpoint file (atomic-ish via write+rename)."""
    cp = work_dir / _CP_FILE
    tmp = work_dir / (_CP_FILE + ".tmp")
    tmp.write_text(json.dumps({"completed": sorted(completed)}, indent=2))
    tmp.replace(cp)


def stage_skip(key: str, output_file: Path, completed: set) -> bool:
    """Return True (and print skip notice) if stage already done and output exists."""
    if key in completed and output_file.exists() and output_file.stat().st_size > 0:
        print(f"\n  [RESUME] {key} already done — skipping ({output_file.name})")
        return True
    return False


def ffprobe_val(video, entries, stream="v:0"):
    r = subprocess.run([
        FFPROBE, "-v", "error", "-select_streams", stream,
        "-show_entries", f"stream={entries}",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video)
    ], capture_output=True, text=True, check=True)
    return r.stdout.strip()


def get_fps(video):
    num, den = ffprobe_val(video, "r_frame_rate").split("/")
    return round(int(num) / int(den), 3)


def get_res(video):
    vals = ffprobe_val(video, "width,height").split("\n")
    return int(vals[0]), int(vals[1])


# -- Stage 1: Deblock + Denoise ----------------------------------------------
def stage_denoise(src, dst, denoise_level, dur_s=0, deinterlace=False):
    """
    GPU: NVDEC decode, NVENC lossless encode.
    CPU: deblock + hqdn3d filters (no CUDA equivalent in any FFmpeg build).
         -threads 0 enables all-core slice-based multithreading (~150fps at 1080p).
    Order: yadif (if deinterlace) -> deblock -> denoise.
    """
    denoise_filter = DENOISE_PRESETS[denoise_level]
    filters = []
    if deinterlace:
        filters.append("yadif=mode=1:parity=-1")   # mode=1: output both fields; parity=-1: auto-detect
    filters.append(f"deblock=filter=strong:block=8,{denoise_filter}")
    vf = ",".join(filters)
    eta = str(timedelta(seconds=int(dur_s / 10))) if dur_s else None

    run([
        FFMPEG, "-threads", "0", "-hwaccel", "cuda", "-y", "-i", src,
        "-vf", vf,
        *_VENC_LOSSLESS,
        "-c:a", "copy", dst
    ], f"Stage 1  Deblock + Denoise ({denoise_level})", eta_hint=eta)


# -- Stage 2: Stabilize (optional) ------------------------------------------
def stage_stabilize(src, dst, work_dir):
    """
    2-pass FFmpeg vidstab:
      Pass 1 - analyse motion, write transforms to .trf file
      Pass 2 - apply smooth stabilization
    smoothing=15: averages over 15 frames for smooth camera motion
    optzoom=1: auto-zoom to hide black border from correction
    """
    trf = work_dir / "vidstab.trf"
    # Use only the filename (no drive letter) so FFmpeg filter parser never sees "C:"
    trf_name = trf.name  # == "vidstab.trf"

    run([
        FFMPEG, "-threads", "0", "-hwaccel", "cuda", "-y", "-i", src,  # NVDEC decode; vidstabdetect is CPU — all cores
        "-vf", f"vidstabdetect=stepsize=6:shakiness=8:accuracy=9:result={trf_name}",
        "-f", "null", "-"
    ], "Stage 2  Stabilize: Pass 1 (motion analysis)", eta_hint="~8 min", cwd=work_dir)

    run([
        FFMPEG, "-threads", "0", "-hwaccel", "cuda", "-y", "-i", src,
        "-vf", (
            f"vidstabtransform=input={trf_name}:"
            f"smoothing=15:optzoom=1:zoom=0:interpol=bicubic,"
            f"unsharp=5:5:0.3:3:3:0.0"
        ),
        *_VENC_LOSSLESS,
        "-c:a", "copy", dst
    ], "Stage 2  Stabilize: Pass 2 (apply)", eta_hint="~8 min", cwd=work_dir)


# -- TRT / ONNX auto-setup ---------------------------------------------------
_HAT_MODELS = {"Real_HAT_GAN_SRx4_sharper", "Real_HAT_GAN_SRx4"}


def _ensure_xformers():
    """Try to get a working xformers build for Flash Attention 2 with float bias.
    Falls back to PyTorch native SDPA (also FA2 on PyTorch 2.6+) if no matching
    pre-built wheel exists for the current torch/CUDA/Python combo.
    Always uses --no-deps to prevent xformers from replacing the CUDA torch build.
    """
    import torch as _torch

    def _test_xformers():
        try:
            import xformers.ops as _xops
            if not _torch.cuda.is_available():
                return False
            _d = _torch.zeros(1, 1, 4, 64, device='cuda', dtype=_torch.float16)
            with _torch.no_grad():
                _xops.memory_efficient_attention(_d, _d, _d)
            del _d
            return True
        except Exception:
            return False

    if _test_xformers():
        return  # already working

    _VER_MAP = {
        (2, 6): "0.0.29.post1",
        (2, 5): "0.0.28",
        (2, 4): "0.0.27",
        (2, 3): "0.0.26.post1",
    }
    _tv = tuple(int(x) for x in _torch.__version__.split('+')[0].split('.')[:2])
    _xf_ver = _VER_MAP.get(_tv, "0.0.29.post1")

    print(f"\n  [OPT] Installing xformers=={_xf_ver} from PyTorch WHL (no-deps) ...")
    try:
        subprocess.run(
            [str(PYTHON), "-m", "pip", "install", f"xformers=={_xf_ver}",
             "--index-url", "https://download.pytorch.org/whl/cu124",
             "--no-deps", "-q"],
            check=True)
        if _test_xformers():
            print("  [OPT] xformers OK \u2014 Flash Attention 2 with float-bias enabled for HAT.")
            return
        subprocess.run([str(PYTHON), "-m", "pip", "uninstall", "xformers", "-y", "-q"],
                       check=False)
    except Exception:
        pass
    print("  [OPT] No compatible xformers wheel \u2014 using PyTorch native SDPA (FA2 built-in).")


def _ensure_ort_export(model_name: str, w: int, h: int, tile: int):
    """
    For HAT models with tile=0, auto-install onnxruntime-gpu (if missing) and
    auto-export the model to ONNX (if not already cached for this resolution).
    Returns the Path to the .onnx file, or None if unavailable.
    This runs once; all subsequent calls return the cached path immediately.
    """
    if tile != 0 or model_name not in _HAT_MODELS:
        return None

    # ONNX tracing stores ALL intermediate activations → needs ~14 KB per pixel.
    # At 1080×1920 (2.07 M px) that is ~28 GB — too large for any single GPU.
    # Skip export for anything larger than 640×640 (409 K px, needs ~6 GB).
    _max_px = 640 * 640
    if w * h > _max_px:
        print(f"  [TRT] Skipping ONNX export: {w}×{h} ({w*h:,} px > {_max_px:,} px limit)"
              f" — resolution too large for ONNX tracing VRAM; using PyTorch fallback.")
        return None

    weights_dir = BASE / "tools/Real-ESRGAN/weights"
    onnx_path   = weights_dir / f"{model_name}_{w}x{h}.onnx"
    if onnx_path.exists():
        return onnx_path

    # ── 1. ensure onnxruntime-gpu is installed ────────────────────────────
    try:
        import importlib
        importlib.import_module("onnxruntime")
    except ImportError:
        print("\n  [TRT] onnxruntime-gpu not found — installing now ...")
        subprocess.run(
            [str(PYTHON), "-m", "pip", "install", "onnxruntime-gpu", "-q"],
            check=True,
        )
        print("  [TRT] onnxruntime-gpu installed.")

    # ── 2. export ONNX (one-time per model+resolution, ~1-2 min) ─────────
    export_script = BASE / "tools/export_hat_trt.py"
    if not export_script.exists():
        print(f"  [TRT] Export script not found at {export_script} — skipping ONNX export.")
        return None

    print(f"\n  [TRT] Exporting {model_name} → ONNX for {w}x{h}"
          f" (one-time, ~2 min — cached afterwards) ...")
    try:
        subprocess.run(
            [
                str(PYTHON), str(export_script),
                "--model",  model_name,
                "--width",  str(w),
                "--height", str(h),
                "--export-only",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  [TRT] ONNX export failed ({e}) — upscaling with PyTorch fallback.")
        return None

    return onnx_path if onnx_path.exists() else None


# -- Stage 3: Real-ESRGAN Upscale -------------------------------------------
def stage_upscale(src, dst, scale, dur_s=0, tile=384, model_name="RealESRGAN_x4plus",
                  dn=0.5, num_procs=1, quant="fp16"):
    """
    Real-ESRGAN video upscale.
    - tile=384 for 8GB VRAM (sweet spot: 2x faster than 512 due to OCAB super-linear cost)
    - tile=512 for 10-12GB VRAM, tile=1024 for 16GB+, tile=0 (no tiling) for 24GB+
    - dn (denoising strength) only applies to realesr-general-x4v3:
        0.0 = keep all grain/texture (best for modern film), 1.0 = heavy smoothing
    - quant: 'fp16' (default), 'int8' (~1.5x speedup, needs calib), 'fp8' (~1.8x, TRT 10+)
    """
    import torch as _torch
    weights_dir = BASE / "tools/Real-ESRGAN/weights"
    weights_dir.mkdir(exist_ok=True)
    # copy model into weights/ if it lives in models/ but not yet there
    model_file = BASE / f"models/{model_name}.pth"
    model_link = weights_dir / f"{model_name}.pth"
    if model_file.exists() and not model_link.exists():
        import shutil as _shutil
        _shutil.copy2(model_file, model_link)

    # VRAM-adaptive tile: full-frame (tile=0) needs 24GB+ for HAT; cap for smaller cards.
    if tile == 0 and _torch.cuda.is_available():
        _vram_gb = _torch.cuda.get_device_properties(0).total_memory / 1e9
        if _vram_gb < 24:
            tile = 384 if _vram_gb < 10 else (512 if _vram_gb < 16 else 1024)
            print(f"  Upscale: tile=0 → tile={tile} (VRAM-adaptive, {_vram_gb:.0f} GB < 24 GB)", flush=True)

    w, h = get_res(src)
    print(f"\n  Input:  {w}x{h}  ->  Output: {w*scale}x{h*scale}")

    # Auto-export to ONNX + install onnxruntime-gpu if not already done.
    # Returns a Path if successful, None if unavailable (graceful fallback).
    onnx_path = _ensure_ort_export(model_name, w, h, tile)

    # L40S-calibrated throughput (fps) per model at 960x414, tile=0, FP16+TRT
    # Baseline: 3.05 fps measured for ESRGAN x4+TRT at 960x414
    # Scaled by pixel area and ~3x TRT advantage: PyTorch fps = 3.05 / (w*h/(960*414)) / 3
    _fps_map = {"Real_HAT_GAN_SRx4_sharper": 0.08, "Real_HAT_GAN_SRx4": 0.09,
                "RealESRGAN_x4plus": 0.19, "RealESRGAN_x2plus": 0.40, "realesr-general-x4v3": 0.25}
    fps_est = _fps_map.get(model_name, 0.10) * (960 * 414) / max(w * h, 1)
    eta = str(timedelta(seconds=int(dur_s * 30 / max(fps_est, 0.001)))) if dur_s else None

    out_dir  = dst.parent
    stem     = Path(src).stem
    suffix   = "esrgan"
    expected = out_dir / f"{stem}_{suffix}.mp4"

    cmd = [
        PYTHON, REALESRGAN,
        "-i", src,
        "-o", out_dir,
        "-n", model_name,
        "-s", str(scale),
        "--suffix", suffix,
        "--tile", str(tile),
        "--tile_pad", "0" if tile == 0 else "8",
        "--pre_pad", "0",
        "--ffmpeg_bin", FFMPEG_BIN,
        "--num_process_per_gpu", str(num_procs),
    ]
    # Pass ONNX model path when available (auto-exported above via _ensure_ort_export)
    if onnx_path is not None and onnx_path.exists():
        cmd += ["--ort-model", str(onnx_path)]
        label = "INT8" if quant == "int8" else ("FP8" if quant == "fp8" else "FP16")
        print(f"  Upscale: TRT/ONNX engine [{label}] \u2192 {onnx_path.name}", flush=True)
    if quant == "fp8" and onnx_path is not None:
        cmd += ["--fp8"]  # tells inference script to set trt_fp8_enable
    if model_name == "realesr-general-x4v3":
        cmd += ["-dn", str(dn)]

    # Force CUDA_VISIBLE_DEVICES=0 so inference_realesrgan_video.py sees exactly
    # one GPU and takes the single-process path (multi-process silently fails when
    # spawned workers can't share a single 8-GB card).
    _saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        run(cmd, f"Stage 3  Real-ESRGAN {scale}x Upscale ({model_name}  dn={dn if model_name == 'realesr-general-x4v3' else 'n/a'})", eta_hint=eta,
            cwd=str(BASE / "tools/Real-ESRGAN"))
    finally:
        if _saved_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _saved_cvd

    if expected.exists() and expected != dst:
        expected.rename(dst)
    if not dst.exists():
        raise RuntimeError(
            f"Stage 3 upscale produced no output (expected {dst}). "
            "Check the log above for errors from inference_realesrgan_video.py."
        )


# -- Stage 3b (optional): STAR Video Diffusion SR ---------------------------
def stage_star(src, dst, scale, prompt="A cinematic live-action film with detailed textures",
               steps=15, chunk=32):
    """
    STAR: Spatial-Temporal Augmentation with Text-to-Video Models.
    ICCV 2025 — video-native diffusion SR with temporal consistency.

    Unlike per-frame diffusion (SUPIR/DiffBIR), STAR processes frames jointly
    in temporal chunks → no flickering. Generates photorealistic texture in
    backgrounds, clothing, skin, architecture — every part of every frame.

    - steps=15  (fast mode, sufficient for most content)
    - steps=50  (maximum quality, ~3x slower)
    - chunk=32  frames per denoising pass. Reduce to 16 if OOM.
    - Needs ~39GB VRAM. Recommended: RunPod L40S pod.
    - License: MIT (I2VGen-XL backbone)

    Install:
        git clone https://github.com/NJU-PCALab/STAR tools/STAR
        cd tools/STAR && pip install -r requirements.txt
        # Download model from https://huggingface.co/SherryX/STAR
        # Place as tools/STAR/pretrained_weight/model.pt
    """
    if not STAR_PY.exists():
        print(f"\n  [WARN] STAR not found at {STAR_DIR}")
        print(f"  [WARN] Install: git clone https://github.com/NJU-PCALab/STAR {STAR_DIR}")
        print(f"  [WARN] Falling back to HAT/ESRGAN upscale.")
        stage_upscale(src, dst, scale)
        return

    if not STAR_MODEL.exists():
        print(f"\n  [WARN] STAR model not found at {STAR_MODEL}")
        print(f"  [WARN] Download light_deg.pt from https://huggingface.co/SherryX/STAR")
        print(f"  [WARN] Rename to model.pt and place at {STAR_MODEL}")
        print(f"  [WARN] Falling back to HAT/ESRGAN upscale.")
        stage_upscale(src, dst, scale)
        return

    orig_dir = os.getcwd()
    os.chdir(str(STAR_DIR))
    try:
        run([
            PYTHON, STAR_PY,
            "--solver_mode", "fast",
            "--steps", str(steps),
            "--input_path", str(src),
            "--model_path", str(STAR_MODEL),
            "--prompt", prompt,
            "--upscale", str(scale),
            "--max_chunk_len", str(chunk),
            "--file_name", dst.name,
            "--save_dir", str(dst.parent),
        ], f"Stage 3  STAR {scale}x Video Diffusion SR  (steps={steps}  chunk={chunk})",
            eta_hint="~1-5 min/chunk on L40S  |  reduce --star-chunk if OOM")
    finally:
        os.chdir(orig_dir)

    # STAR writes to save_dir/file_name
    expected = dst.parent / dst.name
    if not expected.exists():
        candidates = list(dst.parent.glob("*.mp4"))
        if candidates:
            candidates[0].rename(dst)
        else:
            print("  [WARN] STAR output not found — falling back to HAT/ESRGAN")
            stage_upscale(src, dst, scale)


# -- Stage 4 (optional): CodeFormer Face Restoration -----------------------
def stage_face_restore(src, dst, work_dir, fidelity=0.7):
    """
    CodeFormer: blind face restoration using codebook lookup transformer.
    - Detects faces in every frame, restores them, composites back onto frame
    - fidelity 0.0 = maximum enhancement (may change appearance)
    - fidelity 1.0 = maximum faithfulness (minimal change)
    - 0.5-0.7 is a good balance for old/degraded film
    - Requires: tools/CodeFormer/ cloned from https://github.com/sczhou/CodeFormer
    """
    if not CODEFORMER_PY.exists():
        print(f"\n  [WARN] CodeFormer not found at {CODEFORMER_DIR}")
        print(f"  [WARN] Skipping face restoration. To install:")
        print(f"         git clone https://github.com/sczhou/CodeFormer {CODEFORMER_DIR}")
        print(f"         cd {CODEFORMER_DIR} && pip install -r requirements.txt")
        print(f"         python basicsr/setup.py develop")
        print(f"         python scripts/download_pretrained_models.py facelib")
        print(f"         python scripts/download_pretrained_models.py CodeFormer")
        shutil.copy2(src, dst)
        return

    # CodeFormer only accepts .mp4/.mov/.avi — remux MKV if needed (stream copy, fast)
    cf_src = src
    cf_src_tmp = None
    if Path(src).suffix.lower() not in (".mp4", ".mov", ".avi"):
        cf_src_tmp = work_dir / (Path(src).stem + "_cf_input.mp4")
        run([FFMPEG, "-y", "-i", str(src), "-c", "copy", str(cf_src_tmp)],
            "Stage 4  remux for CodeFormer")
        cf_src = cf_src_tmp

    run([
        PYTHON, CODEFORMER_PY,
        "-w", str(fidelity),
        "--bg_upsampler", "None",   # background already upscaled by ESRGAN
        "--face_upsample",          # upsample restored face region
        "--input_path", str(cf_src),
        "--output_path", str(dst.parent),
        "--suffix", "codeformer",
    ], f"Stage 4  CodeFormer Face Restore (fidelity={fidelity})", eta_hint="~15-30 min")

    if cf_src_tmp and cf_src_tmp.exists():
        cf_src_tmp.unlink(missing_ok=True)

    # CodeFormer names output as <stem>_codeformer.<ext>
    stem = Path(cf_src).stem
    expected = dst.parent / f"{stem}_codeformer.mp4"
    if expected.exists() and expected != dst:
        expected.rename(dst)
    elif not dst.exists():
        # fallback: find the output file
        candidates = list(dst.parent.glob(f"{stem}*codeformer*"))
        if candidates:
            candidates[0].rename(dst)
        else:
            print("  [WARN] CodeFormer output not found — copying source as fallback")
            shutil.copy2(src, dst)


# -- Stage 5: Sharpen (low) + Colour Correct ---------------------------------
def stage_post(src, dst):
    """
    GPU: NVDEC decode, NVENC lossless encode.
    CPU: unsharp + eq filters (no CUDA equivalent in standard FFmpeg).
         -threads 0 uses all CPU cores; operates on full 4K output (~80fps effective).
    Applied AFTER upscale so sharpening operates on full-resolution pixels.
    """
    vf = f"{SHARPEN_CAS},{COLOUR_CORRECT}"

    run([
        FFMPEG, "-threads", "0", "-hwaccel", "cuda", "-y", "-i", src,
        "-vf", vf,
        *_VENC_LOSSLESS,
        "-c:a", "copy", dst
    ], "Stage 5  CAS Sharpen + Colour Correct", eta_hint="~15 min")


# -- Stage 1.5: NAFNet Blind Deblur ----------------------------------------
def stage_nafnet_deblur(src, dst, dur_s=0):
    """
    NAFNet-GoPro: blind per-frame deblur / restoration.
    Handles both defocus blur (TELESYNC, old film) and motion blur.
    ~30fps at 1080p on RTX 5090 (FP16).  Runs BEFORE stabilize + upscale
    so HAT gets sharper frames to recover detail from.

    Model download (~18 MB):
      https://github.com/megvii-research/NAFNet/releases/download/tag/NAFNet-GoPro-width32.pth
      -> save to: models/NAFNet-GoPro-width32.pth
    Requires: basicsr >= 1.4.0  (pip install --upgrade basicsr)
    """
    if not NAFNET_MODEL.exists():
        print(f"\n  [SKIP] NAFNet model not found: {NAFNET_MODEL}")
        print(f"  [SKIP] Download (~69 MB):")
        print(f"         gdown 1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj -O {NAFNET_MODEL}")
        shutil.copy2(src, dst)
        return
    if not NAFNET_SCRIPT.exists():
        print(f"  [SKIP] NAFNet script missing: {NAFNET_SCRIPT}")
        shutil.copy2(src, dst)
        return
    eta = str(timedelta(seconds=int(dur_s))) if dur_s else None  # ~30fps -> 1s video = 1s processing
    run([PYTHON, NAFNET_SCRIPT,
         "--input",  src,
         "--output", dst,
         "--model",  NAFNET_MODEL,
         "--ffmpeg", FFMPEG_BIN],
        "Stage 1.5  NAFNet blind deblur (per-frame, FP16)", eta_hint=eta)


# -- Stage 1.6: EDVR Temporal Restoration -----------------------------------
def stage_edvr_restore(src, dst, dur_s=0):
    """
    EDVR: temporal multi-frame video restoration using deformable convolution.
    Uses a 5-frame sliding window — recovers detail that was inconsistently
    preserved/lost across frames due to compression or camera motion.
    ~8fps at 1080p on RTX 5090.  Runs after NAFNet deblur, before upscale.

    Model download (~28 MB):
      https://github.com/XPixelGroup/BasicSR/releases/download/V1.3.5/EDVR_M_deblur_REDS_official-d4ad3da9.pth
      -> rename to EDVR-deblur.pth, save to: models/EDVR-deblur.pth
    Requires: basicsr  (pip install basicsr)
    """
    if not EDVR_MODEL.exists():
        print(f"\n  [SKIP] EDVR model not found: {EDVR_MODEL}")
        print(f"  [SKIP] Download (~95 MB):")
        print(f"         gdown 1_ma2tgHscZtkIY2tEJkVdU-UP8bnqBRE -O {EDVR_MODEL}")
        shutil.copy2(src, dst)
        return
    if not EDVR_SCRIPT.exists():
        print(f"  [SKIP] EDVR script missing: {EDVR_SCRIPT}")
        shutil.copy2(src, dst)
        return
    eta = str(timedelta(seconds=int(dur_s * 3))) if dur_s else None  # ~8fps -> ~3x realtime
    run([PYTHON, EDVR_SCRIPT,
         "--input",  src,
         "--output", dst,
         "--model",  EDVR_MODEL,
         "--ffmpeg", FFMPEG_BIN],
        "Stage 1.6  EDVR temporal restoration (5-frame window, FP16)", eta_hint=eta)


# -- Stage 6.5: Synthetic Grain Add-back ------------------------------------
def stage_grain(src, dst, dur_s=0):
    """
    Add organic synthetic grain after RIFE to counter the 'plastic/CGI' look
    that all AI upscalers produce.  Psycho-visually this is very effective --
    the human eye expects film grain and interprets its absence as artificial.

    Temporal noise (c0f=t) ensures grain is consistent frame-to-frame, so it
    doesn't flicker.  Luma-heavy (c0s=8), light chroma (c1s/c2s=3).
    Applied BEFORE final NVENC encode so the encoder compresses it efficiently.
    Near-zero added time (~3 min).
    """
    eta = str(timedelta(seconds=int(dur_s / 20))) if dur_s else None  # ~20x realtime
    run([
        FFMPEG, "-threads", "0", "-hwaccel", "cuda", "-y", "-i", src,
        "-vf", "noise=c0s=8:c0f=t:c1s=3:c1f=t:c2s=3:c2f=t",
        *_VENC_LOSSLESS,
        "-c:a", "copy", dst
    ], "Stage 6.5  Synthetic grain add-back (luma=8, chroma=3, temporal)", eta_hint=eta)


# -- Stage 6: RIFE 2x FPS ---------------------------------------------------
def stage_rife(src, dst, work_dir, original_fps, exp=1, uhd=False):
    """
    RIFE frame interpolation: multiplies frame rate via AI optical flow.

    Strategy (in order of preference):
    1. Python RIFE (tools/RIFE/inference_video.py) — works on any CUDA device
       (Windows or Linux). exp=1 for 2x (default), exp=2 for 4x FPS.
       uhd=False default: full optical-flow resolution (fine on 32 GB VRAM).
    2. rife-ncnn-vulkan.exe — Windows-only fallback via frame extraction.
    """
    w, h = get_res(src)
    is_4k = (w >= 3840 or h >= 2160)

    # --- Prefer Python RIFE (cross-platform CUDA) ---
    if RIFE_PY.exists() and (RIFE_PY_MODEL / "RIFE_HDv3.py").exists():
        _stage_rife_python(src, dst, work_dir, original_fps, is_4k, exp=exp, uhd=uhd)
    elif RIFE_EXE.exists():
        _stage_rife_ncnn(src, dst, work_dir, original_fps)
    else:
        print("\n  [WARN] No RIFE binary found — skipping frame interpolation.")
        shutil.copy2(src, dst)


def _stage_rife_python(src, dst, work_dir, original_fps, is_4k, exp=1, uhd=False):
    """Python RIFE — CUDA, cross-platform (Windows + Linux)."""
    import sys as _sys
    rife_dir = RIFE_PY.parent
    tmp_out = work_dir / "rife_py_out.mp4"
    mult = 2 ** exp

    cmd = [
        PYTHON, RIFE_PY,
        "--video", str(src),
        "--output", str(tmp_out),
        "--model", str(RIFE_PY_MODEL),
        "--exp", str(exp),      # 1=2x, 2=4x FPS
        "--ext", "mp4",
        "--fp16",               # FP16 — Blackwell tensor core acceleration
    ]
    if uhd:
        cmd.append("--UHD")    # halves optical-flow scale; only needed for <16 GB VRAM

    # inference_video.py uses relative model path — run from its directory
    orig_dir = os.getcwd()
    os.chdir(str(rife_dir))
    try:
        run(cmd, f"Stage 6  RIFE {mult}x FPS Python/CUDA ({original_fps:.3f} -> {original_fps*mult:.3f} fps)",
            eta_hint="~30-60 min on RTX 5090 for 4K movie")
    finally:
        os.chdir(orig_dir)

    # Merge audio back (RIFE drops audio)
    run([
        FFMPEG, "-y",
        "-i", tmp_out,
        "-i", src,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "copy",
        dst
    ], "  RIFE  mux audio back")

    tmp_out.unlink(missing_ok=True)


def _stage_rife_ncnn(src, dst, work_dir, original_fps):
    """ncnn-vulkan RIFE — Windows-only fallback."""
    frames_in  = work_dir / "rife_in"
    frames_out = work_dir / "rife_out"
    frames_in.mkdir(exist_ok=True)
    frames_out.mkdir(exist_ok=True)

    run([
        FFMPEG, "-y", "-i", src,
        "-vsync", "0",
        str(frames_in / "%08d.png")
    ], "  RIFE ncnn  extract frames", eta_hint="~5 min")

    run([
        RIFE_EXE,
        "-i", frames_in, "-o", frames_out,
        "-m", RIFE_MODEL_DIR,
        "-j", "1:2:1",
        "-f", "%08d.png",
    ], f"Stage 6  RIFE v4.6 ncnn  {original_fps}fps -> {original_fps*2}fps", eta_hint="~3-4 hours")

    new_fps = original_fps * 2
    run([
        FFMPEG, "-y",
        "-framerate", str(new_fps),
        "-i", str(frames_out / "%08d.png"),
        "-i", src,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
        "-c:a", "copy",
        dst
    ], f"  RIFE  reassemble at {new_fps}fps", eta_hint="~10 min")

    shutil.rmtree(frames_in, ignore_errors=True)
    shutil.rmtree(frames_out, ignore_errors=True)


# -- Final Encode ------------------------------------------------------------
def stage_encode(video_src, audio_src, dst, codec="hevc"):
    """NVENC encode. codec='hevc' (H.265) or 'av1' (AV1 NVENC, ~30 pct smaller at same quality)."""
    if codec == "av1":
        vc_args = [
            "-c:v", "av1_nvenc", "-preset", "p7",
            "-rc", "vbr", "-cq", "28", "-b:v", "0",
            "-rc-lookahead", "32",            # GPU temporal lookahead — all 3 NVENC engines on RTX 5090
            "-spatial-aq", "1", "-temporal-aq", "1",
        ]
        label = "Final Encode  AV1 NVENC GPU  CQ28"
    else:
        if _NVENC_OK:
            vc_args = [
                "-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "hq",
                "-rc", "vbr", "-cq", "14", "-b:v", "0",
                "-maxrate", "120M", "-multipass", "fullres",
                "-rc-lookahead", "32",
                "-spatial-aq", "1", "-aq-strength", "8", "-temporal-aq", "1",
                "-tag:v", "hvc1",
            ]
            label = "Final Encode  HEVC NVENC GPU  CQ14"
        else:
            vc_args = ["-c:v", "libx265", "-preset", "medium", "-crf", "18",
                       "-x265-params", "log-level=error"]
            label = "Final Encode  HEVC x265 software  CRF18"
    run([
        FFMPEG, "-y",
        "-i", video_src,
        "-i", audio_src,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-map", "1:s?",
        *vc_args,
        "-c:a", "copy",
        "-c:s", "copy",
        dst
    ], label, eta_hint="~5-10 min")


# -- Main --------------------------------------------------------------------
def main():
    # Force UTF-8 on Windows terminals (CP1252 can't encode box-drawing chars)
    import io as _io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

    ap = argparse.ArgumentParser(
        description="Video upscale pipeline — defaults tuned for L40S 48 GB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input/output
    ap.add_argument("input",            help="Source video file (positional or --input)")
    ap.add_argument("--output",         default=None,
                    help="Output file path. Default: output/<name>_<scale>x.mkv")

    # Preprocessing
    ap.add_argument("--denoise",        default="medium", choices=["low", "medium"],
                    help="FFmpeg deblock+hqdn3d strength before upscale.")
    ap.add_argument("--deinterlace",    action="store_true", default=False,
                    help="Deinterlace source with yadif (interlaced TV/broadcast only).")
    ap.add_argument("--stabilize",      action=argparse.BooleanOptionalAction, default=True,
                    help="2-pass vidstab stabilization. Disable: --no-stabilize.")

    # Upscale
    ap.add_argument("--scale",          type=int, default=4, choices=[2, 4],
                    help="Upscale factor.")
    ap.add_argument("--tile",           type=int, default=0,
                    help="Real-ESRGAN tile size. 0=no tiling (best, needs 24 GB+ VRAM). 512=8 GB, 1024=12 GB.")
    ap.add_argument("--model",          default="Real_HAT_GAN_SRx4_sharper",
                    choices=["RealESRGAN_x2plus", "Real_HAT_GAN_SRx4_sharper", "Real_HAT_GAN_SRx4",
                             "RealESRGAN_x4plus", "realesr-general-x4v3"],
                    help="Super-resolution model.")
    ap.add_argument("--dn",             type=float, default=0.5,
                    help="Denoising strength for realesr-general-x4v3 (0=keep grain, 1=smooth).")
    ap.add_argument("--upscale-procs",  type=int, default=1,
                    help="Concurrent HAT processes per GPU. L40S 48 GB fits 2. Use 1 for <= 16 GB.")

    # Quantisation precision for HAT TRT engine
    ap.add_argument("--quant",          default="fp8",
                    choices=["fp16", "int8", "fp8"],
                    help="HAT TRT engine precision. "
                         "fp16 = default FP16 ONNX + TRT. "
                         "int8 = static INT8 QDQ (calibration from input video, ~1.5x speedup, <0.1 dB loss). "
                         "fp8  = FP16 ONNX + trt_fp8_enable on Ada/Hopper GPU, ~1.8x speedup, TRT 10+ required.")

    # RIFE frame interpolation
    ap.add_argument("--no-rife",        action="store_true",
                    help="Disable RIFE frame interpolation.")
    ap.add_argument("--rife-exp",       type=int, default=1, choices=[1, 2],
                    help="RIFE exponent: 1=2x FPS, 2=4x FPS.")
    ap.add_argument("--rife-uhd",       action=argparse.BooleanOptionalAction, default=False,
                    help="RIFE UHD mode: halve optical-flow scale for < 16 GB VRAM.")
    ap.add_argument("--rife-order",     default="late", choices=["early", "late"],
                    help="RIFE timing: 'late'=after upscale at 4K (default, ~1.9x faster HAT); "
                         "'early'=before upscale at native res (legacy, HAT sees all interpolated frames).")

    # Face restoration
    ap.add_argument("--face-restore",   action=argparse.BooleanOptionalAction, default=True,
                    help="CodeFormer face restoration. Disable: --no-face-restore.")
    ap.add_argument("--face-fidelity",  type=float, default=0.7,
                    help="CodeFormer fidelity 0-1. Lower = more correction, less fidelity.")

    # STAR generative (optional, very slow)
    ap.add_argument("--generative-video", action=argparse.BooleanOptionalAction, default=False,
                    help="Use STAR video diffusion SR instead of HAT. Very slow (~0.4 fps). Short clips only.")
    ap.add_argument("--star-prompt",    default="A cinematic live-action film with detailed textures",
                    help="Text prompt for STAR diffusion.")
    ap.add_argument("--star-steps",     type=int, default=50,
                    help="STAR denoising steps. 50=max quality (default), 15=fast.")
    ap.add_argument("--star-chunk",     type=int, default=64,
                    help="STAR temporal chunk size. 64 for 48 GB, 32 for 24 GB, 16 for 16 GB.")

    # Output encoding
    ap.add_argument("--codec",          default="hevc", choices=["hevc", "av1"],
                    help="Output codec. hevc=H.265 NVENC CQ14, av1=AV1 NVENC CQ28 (~30 pct smaller, Ada+).")

    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"ERROR: File not found: {src}")

    # CUDA + CPU thread optimisations — inherited by all subprocess calls
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    _ncpu = str(os.cpu_count() or 16)
    os.environ.setdefault("OMP_NUM_THREADS",  _ncpu)
    os.environ.setdefault("MKL_NUM_THREADS",  _ncpu)

    # Auto-install performance dependencies (one-time, silent if already present)
    _ensure_xformers()   # Flash Attention 2 with float bias for HAT

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = Path(args.output) if args.output else OUTPUT_DIR / f"{src.stem}_{args.scale}x.mkv"

    work = BASE / "temp" / src.stem
    work.mkdir(parents=True, exist_ok=True)

    f_denoised   = work / "s1_denoised.mkv"
    f_nafnet     = work / "s1b_nafnet.mkv"
    f_edvr       = work / "s1c_edvr.mkv"
    f_stabilized = work / "s2_stabilized.mkv"
    f_upscaled   = work / "s3_upscaled.mkv"
    f_rife_4k    = work / "s3b_rife4k.mkv"   # RIFE at 4K res (late-RIFE pipeline)
    f_faces      = work / "s4_faces.mkv"
    f_postproc   = work / "s5_postproc.mkv"
    f_rife       = work / "s2b_rife.mkv"    # RIFE at native res — before upscale
    f_grain      = work / "s6b_grain.mkv"

    original_fps = get_fps(src)
    w, h = get_res(src)
    dur_s = int(ffprobe_val(src, "duration").split("\n")[0].split(".")[0])

    # determine RIFE availability
    has_rife_py  = RIFE_PY.exists() and (RIFE_PY_MODEL / "RIFE_HDv3.py").exists()
    has_rife_exe = RIFE_EXE.exists()
    rife_avail   = has_rife_py or has_rife_exe

    # ETA estimates (calibrated for RTX 5090, typical 24fps source)
    eta_s1  = int(dur_s * original_fps / 150)   # all-core deblock+hqdn3d, ~150fps at 1080p
    eta_s1b = int(dur_s * original_fps / 2.5) if NAFNET_MODEL.exists() else 0  # NAFNet ~2.5fps at 1080p (batched, compiled)
    eta_s1c = int(dur_s * original_fps / 8)  if EDVR_MODEL.exists()  else 0  # EDVR   ~8fps  at 1080p
    # Calibrated from L40S actual run: RealESRGAN x4 at 960x414 TRT = 3.05 fps avg.
    # Scaled to 1080x1920 PyTorch (no TRT): 3.05 / 5.22px_ratio / 3x_TRT = ~0.19 fps.
    # HAT ~2.4x more FLOPs than ESRGAN (102G vs 43G per 64x64 patch) → 0.08 fps.
    _fps_l40s = {"Real_HAT_GAN_SRx4_sharper": 0.08, "Real_HAT_GAN_SRx4": 0.09,
                 "RealESRGAN_x4plus": 0.19, "RealESRGAN_x2plus": 0.40, "realesr-general-x4v3": 0.25}
    _rife_on   = not args.no_rife and rife_avail
    _rife_mult = (2 ** args.rife_exp) if _rife_on else 1
    if args.generative_video:
        eta_s3 = int(dur_s * original_fps / 0.4)   # STAR ~0.4fps — best for short clips (<5 min)
    else:
        # HAT processes _rife_mult × frames when RIFE ran first at native res
        eta_s3 = int(dur_s * original_fps * _rife_mult / _fps_l40s.get(args.model, 0.08))
    eta_s4f = dur_s // 4 if args.face_restore else 0
    eta_s5  = int(dur_s * original_fps / 80)    # all-core unsharp+eq at 4K, ~80fps
    # RIFE at native 1080p on L40S — optical flow is lighter than SR, ~3 fps est
    eta_s6  = int(dur_s * original_fps / 3) if _rife_on else 0
    eta_enc = int(dur_s * 60)   # NVENC at ~1x realtime for 4K HEVC (conservative)
    eta_total = eta_s1 + eta_s1b + eta_s1c + eta_s6 + eta_s3 + eta_s4f + eta_s5 + eta_enc

    pipeline_start = time.time()

    print(f"\n{'#'*62}")
    print(f"  INPUT      : {src.name}")
    print(f"  RESOLUTION : {w}x{h}  ->  {w*args.scale}x{h*args.scale}")
    print(f"  FPS        : {original_fps}  ->  {original_fps*2 if not args.no_rife and rife_avail else original_fps}")
    print(f"  DENOISE    : {args.denoise}" + ("  + yadif deinterlace" if args.deinterlace else ""))
    print(f"  STABILIZE  : {'yes' if args.stabilize else 'no'}")
    if args.generative_video:
        print(f"  UPSCALE    : STAR video diffusion SR  (steps={args.star_steps}  chunk={args.star_chunk})")
        print(f"  STAR PROMPT: {args.star_prompt}")
    else:
        print(f"  MODEL      : {args.model}" + (f"  (dn={args.dn})" if args.model == "realesr-general-x4v3" else ""))
    print(f"  FACE RESTORE: {'yes (fidelity=' + str(args.face_fidelity) + ')' if args.face_restore else 'no'}")
    print(f"  RIFE       : {'Python/CUDA' if has_rife_py else 'ncnn/Vulkan' if has_rife_exe else 'NOT AVAILABLE'}  (exp={args.rife_exp}, uhd={args.rife_uhd})")
    print(f"  CODEC      : {'AV1 NVENC CQ28' if args.codec == 'av1' else ('HEVC NVENC CQ14' if _NVENC_OK else 'HEVC x265 CRF18 (software — NVENC unavailable)')}")
    print(f"  OUTPUT     : {out}")
    print(f"{'-'*62}")
    print(f"  ESTIMATED PIPELINE TIME:")
    print(f"    Stage 1  Deblock+Denoise : ~{str(timedelta(seconds=eta_s1))}")
    if NAFNET_MODEL.exists():
        print(f"    Stage 1.5 NAFNet deblur  : ~{str(timedelta(seconds=eta_s1b))}")
    else:
        print(f"    Stage 1.5 NAFNet deblur  : SKIPPED (model not found — see stage_nafnet_deblur docstring)")
    if EDVR_MODEL.exists():
        print(f"    Stage 1.6 EDVR temporal  : ~{str(timedelta(seconds=eta_s1c))}")
    else:
        print(f"    Stage 1.6 EDVR temporal  : SKIPPED (model not found — see stage_edvr_restore docstring)")
    stage3_label = f"STAR {args.scale}x diffusion" if args.generative_video else f"HAT/ESRGAN {args.scale}x"
    if _rife_on:
        rife_mult = 2 ** args.rife_exp
        print(f"    Stage 2b RIFE {rife_mult}x FPS (native res): ~{str(timedelta(seconds=eta_s6))}")
    print(f"    Stage 3  {stage3_label:<22}: ~{str(timedelta(seconds=eta_s3))}")
    if args.face_restore:
        print(f"    Stage 4  CodeFormer      : ~{str(timedelta(seconds=eta_s4f))}")
    print(f"    Stage 5  Sharpen+Colour  : ~{str(timedelta(seconds=eta_s5))}")
    if _rife_on:
        print(f"    Stage 6.5 Grain add-back : ~3 min")
    print(f"    Final    NVENC encode    : ~{str(timedelta(seconds=eta_enc))}")
    print(f"    {'-'*37}")
    print(f"    TOTAL                    : ~{str(timedelta(seconds=eta_total))}")
    print(f"{'#'*62}")

    completed = checkpoint_load(work)
    if completed:
        print(f"\n  [RESUME] Checkpoint found — already done: {', '.join(sorted(completed))}")

    try:
        if not stage_skip("stage1", f_denoised, completed):
            stage_denoise(src, f_denoised, args.denoise, dur_s=dur_s, deinterlace=args.deinterlace)
            completed.add("stage1"); checkpoint_save(work, completed)

        if not stage_skip("stage1b", f_nafnet, completed):
            stage_nafnet_deblur(f_denoised, f_nafnet, dur_s=dur_s)
            completed.add("stage1b"); checkpoint_save(work, completed)

        if not stage_skip("stage1c", f_edvr, completed):
            stage_edvr_restore(f_nafnet, f_edvr, dur_s=dur_s)
            completed.add("stage1c"); checkpoint_save(work, completed)

        if args.stabilize:
            if not stage_skip("stage2", f_stabilized, completed):
                stage_stabilize(f_edvr, f_stabilized, work)
                completed.add("stage2"); checkpoint_save(work, completed)
            pre_upscale = f_stabilized
        else:
            pre_upscale = f_edvr

        # RIFE at native res: 16× faster than post-upscale; HAT then upscales all interpolated frames
        _rife_order = args.rife_order  # 'early' or 'late'
        if not args.no_rife and rife_avail and _rife_order == 'early':
            if not stage_skip("stage2b", f_rife, completed):
                stage_rife(pre_upscale, f_rife, work, original_fps, exp=args.rife_exp, uhd=args.rife_uhd)
                completed.add("stage2b"); checkpoint_save(work, completed)
            pre_upscale = f_rife
        elif not args.no_rife and not rife_avail:
            print("\n  [WARN] No RIFE available — skipping frame interpolation")

        if args.generative_video:
            if not stage_skip("stage3", f_upscaled, completed):
                stage_star(pre_upscale, f_upscaled, args.scale,
                           prompt=args.star_prompt, steps=args.star_steps, chunk=args.star_chunk)
                completed.add("stage3"); checkpoint_save(work, completed)
        else:
            if not stage_skip("stage3", f_upscaled, completed):
                stage_upscale(pre_upscale, f_upscaled, args.scale, dur_s=dur_s,
                              tile=args.tile, model_name=args.model, dn=args.dn,
                              num_procs=args.upscale_procs, quant=args.quant)
                completed.add("stage3"); checkpoint_save(work, completed)

        # LATE RIFE: interpolate AFTER upscale at full 4K output
        # 1.89x faster overall: 549 HAT frames instead of 1098, RIFE at 4K is only 4.65s/pair.
        _after_upscale = f_upscaled
        if not args.no_rife and rife_avail and _rife_order == 'late':
            _upscaled_fps  = original_fps  # HAT output is still original fps
            if not stage_skip("stage3b", f_rife_4k, completed):
                stage_rife(f_upscaled, f_rife_4k, work, _upscaled_fps, exp=args.rife_exp, uhd=args.rife_uhd)
                completed.add("stage3b"); checkpoint_save(work, completed)
            _after_upscale = f_rife_4k

        if args.face_restore:
            if not stage_skip("stage4", f_faces, completed):
                stage_face_restore(_after_upscale, f_faces, work, fidelity=args.face_fidelity)
                completed.add("stage4"); checkpoint_save(work, completed)
            pre_post = f_faces
        else:
            pre_post = _after_upscale

        if not stage_skip("stage5", f_postproc, completed):
            stage_post(pre_post, f_postproc)
            completed.add("stage5"); checkpoint_save(work, completed)

        if not stage_skip("stage6b", f_grain, completed):
            stage_grain(f_postproc, f_grain, dur_s=dur_s)
            completed.add("stage6b"); checkpoint_save(work, completed)

        stage_encode(f_grain, src, out, codec=args.codec)

        total_elapsed = time.time() - pipeline_start
        print(f"\n{'#'*62}")
        print(f"  DONE!  {out}")
        print(f"  Size:  {out.stat().st_size/1e9:.2f} GB")
        print(f"  Total time: {str(timedelta(seconds=int(total_elapsed)))}")
        print(f"{'#'*62}\n")
        shutil.rmtree(work, ignore_errors=True)   # clean temp only on full SUCCESS

    except Exception:
        print(f"\n{'!'*62}")
        print(f"  Pipeline stopped. Temp files preserved at:")
        print(f"    {work}")
        print(f"  Completed stages: {', '.join(sorted(completed)) or 'none'}")
        print(f"  Re-run the SAME command to resume from last checkpoint.")
        print(f"{'!'*62}\n")
        raise


if __name__ == "__main__":
    main()

