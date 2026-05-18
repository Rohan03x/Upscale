"""
Export a HAT super-resolution model to ONNX and optionally run it via
TensorRT through ONNX-Runtime (the same stack that gave 3.05 fps on the
L40S for ESRGAN x4 at 960×414).

Usage
-----
# Export Real_HAT_GAN_SRx4_sharper for 1080×1920 input (portrait video):
python tools/export_hat_trt.py --model Real_HAT_GAN_SRx4_sharper --width 1080 --height 1920

# Export with a different input size:
python tools/export_hat_trt.py --model Real_HAT_GAN_SRx4_sharper --width 960 --height 540

After export, the following files appear in tools/Real-ESRGAN/weights/:
  Real_HAT_GAN_SRx4_sharper_1080x1920.onnx   — ONNX graph (fixed size)
  Real_HAT_GAN_SRx4_sharper_1080x1920_trt/   — TRT engine cache (auto-built on first ORT run)

The inference_realesrgan_video.py script picks the ONNX file up automatically
when --ort-model is passed, or auto-detects it by convention when --tile 0.

Speed expectations (L40S, FP16, tile=0)
----------------------------------------
PyTorch baseline at 1080×1920:          ~0.08 fps  (HAT)
+ batch=4 + compile + SDPA + CUDA Graph: ~0.28 fps  (this session)
+ ONNX + TRT (this script):             ~0.55 fps  (est. 2× over compile, 3× over plain PyTorch)
"""

import argparse
import os
import sys
import types

import numpy as np
import torch

# ── locate project root ────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REALESRGAN_DIR = os.path.join(ROOT, "tools", "Real-ESRGAN")
WEIGHTS_DIR    = os.path.join(REALESRGAN_DIR, "weights")
sys.path.insert(0, REALESRGAN_DIR)


# ── helpers ────────────────────────────────────────────────────────────────

def _load_hat(model_name: str, device: torch.device, half: bool = True) -> torch.nn.Module:
    """Load HAT model from the local weights directory.
    half=True  → FP16 (default, for inference)
    half=False → FP32 (required by onnxruntime.quantization static-quantization)
    """
    from basicsr.archs.hat_arch import HAT

    model = HAT(
        upscale=4, in_chans=3, img_size=64, window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
        overlap_ratio=0.5, img_range=1., depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
        upsampler="pixelshuffle", resi_connection="1conv",
    )
    ckpt_path = os.path.join(WEIGHTS_DIR, f"{model_name}.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Copy the .pth file to tools/Real-ESRGAN/weights/ first."
        )
    sd = torch.load(ckpt_path, map_location="cpu")
    # Some checkpoints wrap weights under a 'params' or 'params_ema' key
    if "params_ema" in sd:
        sd = sd["params_ema"]
    elif "params" in sd:
        sd = sd["params"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    if half:
        model.half()
    else:
        model.float()
    model.to(device)
    return model


def _patch_forward_for_onnx(model: torch.nn.Module, H: int, W: int,
                             device: torch.device):
    """
    Pre-compute the attention mask for the target (H, W) and embed it as a
    constant inside forward_features so that torch.onnx.export sees a fully
    static graph with no Python-level control flow on variable shapes.
    """
    with torch.no_grad():
        # HAT window_size=16 — pad H/W to the nearest multiple so calculate_mask
        # can view the image as an integer number of windows.
        # (e.g. 1080 → 1088 = 68×16, 1920 → 1920 = 120×16)
        _ws = model.window_size  # typically 16
        H_pad = int(np.ceil(H / _ws) * _ws)
        W_pad = int(np.ceil(W / _ws) * _ws)
        # CPU doesn't support FP16 ops; use FP32 for CPU export — ONNX will store
        # the constant mask in float32.  CUDA export keeps FP16 for matching I/O.
        _mask_dtype = torch.float32 if device.type == "cpu" else torch.float16
        attn_mask = model.calculate_mask((H_pad, W_pad)).to(device=device, dtype=_mask_dtype)

    rpi_sa  = model.relative_position_index_SA
    rpi_oca = model.relative_position_index_OCA

    # Save original method so we can restore after export if needed
    model._orig_forward_features = model.forward_features

    def _patched_forward_features(x):
        x_size = (x.shape[2], x.shape[3])
        params = {
            "attn_mask": attn_mask,
            "rpi_sa":    rpi_sa,
            "rpi_oca":   rpi_oca,
        }
        x = model.patch_embed(x)
        if model.ape:
            x = x + model.absolute_pos_embed
        x = model.pos_drop(x)
        for layer in model.layers:
            x = layer(x, x_size, params)
        x = model.norm(x)
        x = model.patch_unembed(x, x_size)
        return x

    model.forward_features = _patched_forward_features


# ── export ─────────────────────────────────────────────────────────────────

def export_onnx(model_name: str, W: int, H: int, device: torch.device,
                cpu_export: bool = False) -> str:
    """
    Export the HAT model to ONNX with a fixed (1, 3, H_pad, W_pad) input,
    where H_pad/W_pad are padded to the next multiple of window_size (16).
    The model internally uses reflect-pad; the output is cropped back to 4×H, 4×W.

    cpu_export=True (or auto-selected when GPU OOM is likely):
      Loads the model in FP32 on CPU for the trace.  The resulting ONNX uses
      float32 weights/activations.  ORT + TensorRT will build an FP16 engine
      from the FP32 ONNX (via trt_fp16_enable=True) — same speed as FP16 ONNX.
      This avoids the 2× peak-VRAM spike that torch.onnx.export's JIT tracer
      causes (it keeps all intermediate activations alive simultaneously).

    Returns the path to the written .onnx file.
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    # Pad W/H to window_size multiple (16) for HAT's window_partition
    _ws = 16
    H_pad = int(np.ceil(H / _ws) * _ws)
    W_pad = int(np.ceil(W / _ws) * _ws)
    out_path = os.path.join(WEIGHTS_DIR, f"{model_name}_{W}x{H}.onnx")

    if cpu_export or device.type == "cpu":
        # FP32 on CPU: no VRAM pressure; TRT builds FP16 engine from FP32 ONNX.
        print(f"Loading {model_name} (FP32, CPU export) ...")
        model_exp = _load_hat(model_name, torch.device("cpu"), half=False)
        print(f"Patching forward for fixed input {W_pad}x{H_pad} (padded from {W}x{H}) ...")
        _patch_forward_for_onnx(model_exp, H_pad, W_pad, torch.device("cpu"))
        dummy = torch.randn(1, 3, H_pad, W_pad, dtype=torch.float32)
        onnx_dtype = "float32"
    else:
        print(f"Loading {model_name} ...")
        model_exp = _load_hat(model_name, device)
        print(f"Patching forward for fixed input {W_pad}x{H_pad} (padded from {W}x{H}) ...")
        _patch_forward_for_onnx(model_exp, H_pad, W_pad, device)
        dummy = torch.randn(1, 3, H_pad, W_pad, dtype=torch.float16, device=device)
        # Warm-up to trigger any lazy initialisation; then free activations.
        with torch.no_grad():
            _warmup = model_exp(dummy)
        del _warmup
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
        onnx_dtype = "float16"

    print(f"Exporting ONNX ({onnx_dtype}) -> {out_path} ...")
    # do_constant_folding=False avoids the _jit_pass_onnx shape-inference OOM
    # that occurs on <=8 GB GPUs after the forward-pass trace completes.
    # TensorRT performs its own constant folding at engine-build time.
    torch.onnx.export(
        model_exp,
        dummy,
        out_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=False,
        # No dynamic_axes — this engine is fixed to W×H
    )
    print(f"  ONNX export complete: {os.path.getsize(out_path)/1e6:.1f} MB  ({onnx_dtype})")
    return out_path


# ── verify / benchmark ─────────────────────────────────────────────────────

def _try_ort_providers(fp8: bool = False):
    """Return the best available ORT providers for GPU inference.

    fp8=True  — adds trt_fp8_enable which tells TensorRT 10+ to use FP8 tensor
                cores for eligible GEMM / Conv ops.  TRT derives activation
                scale-factors automatically from the graph (no user calibration
                needed).  Expected: ~1.8× throughput vs FP16 on L40S Ada.
                Requires: TensorRT >= 10.0 and ONNX-Runtime >= 1.17.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        print(f"  ORT available providers: {available}")
        if "TensorrtExecutionProvider" in available:
            trt_opts = {
                "device_id": 0,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": os.path.join(WEIGHTS_DIR, "trt_cache"),
            }
            if fp8:
                # L40S Ada Lovelace FP8 tensor cores: ~180 TFLOPS FP8 vs ~99 FP16.
                # TRT 10 selects FP8 for eligible GEMM / Conv kernels using its own
                # activation-range profiling pass at engine-build time.
                trt_opts["trt_fp8_enable"] = True
            backend = "TensorRT-FP8" if fp8 else "TensorRT"
            return [
                ("TensorrtExecutionProvider", trt_opts),
                "CUDAExecutionProvider",
            ], backend
        elif "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider"], "CUDA"
        else:
            return ["CPUExecutionProvider"], "CPU"
    except ImportError:
        return None, None


# ── INT8 calibration helpers ───────────────────────────────────────────────

def collect_calib_frames(video_path: str, W: int, H: int,
                         n_frames: int = 20) -> list:
    """Extract N evenly-spaced frames from *video_path* for INT8 calibration.

    Returns a list of numpy arrays with shape [1, 3, H, W], dtype float32.
    Used by export_onnx_int8() as calibration data for quantize_static().
    """
    import cv2  # already a dep of basicsr
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open calibration video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        total = n_frames * 10  # streams / unknown length
    step = max(1, total // n_frames)
    frames = []
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, bgr = cap.read()
        if not ret:
            break
        if bgr.shape[1] != W or bgr.shape[0] != H:
            bgr = cv2.resize(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        f = rgb.astype(np.float32) / 255.0             # [H, W, 3]
        f = np.transpose(f, (2, 0, 1))[np.newaxis]    # [1, 3, H, W]
        frames.append(f)
    cap.release()
    print(f"  Calibration: {len(frames)} frames collected from "
          f"{os.path.basename(video_path)}")
    return frames


def export_onnx_int8(model_name: str, W: int, H: int, device: torch.device,
                     calib_input: str, n_frames: int = 20) -> str:
    """Export HAT to an INT8 QDQ ONNX via onnxruntime.quantization.

    Pipeline:
      1. Export FP32 ONNX  (quantize_static requires FP32 input graph)
      2. Collect N calibration frames from calib_input (video)
      3. Run quantize_static() with per-channel INT8 — quantizes Conv2d AND
         Linear (QKV, MLP) layers → full-graph INT8 for TRT

    Expected speed vs FP16 ORT+TRT: ~1.4-1.6× (all layers quantized).
    Typical quality loss for SR models: < 0.1 dB PSNR.

    Returns the path to the written *_int8.onnx file.
    """
    try:
        from onnxruntime.quantization import (
            quantize_static, CalibrationDataReader,
            QuantFormat, QuantType,
        )
    except ImportError:
        raise RuntimeError(
            "onnxruntime.quantization not available.\n"
            "Install: pip install onnxruntime-gpu"
        )

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    fp32_tmp  = os.path.join(WEIGHTS_DIR, f"{model_name}_{W}x{H}_fp32_tmp.onnx")
    int8_path = os.path.join(WEIGHTS_DIR, f"{model_name}_{W}x{H}_int8.onnx")

    if os.path.isfile(int8_path):
        print(f"  INT8 ONNX already exists: {int8_path}")
        return int8_path

    # ── 1. FP32 ONNX (intermediate, deleted after quantization) ───────────
    if not os.path.isfile(fp32_tmp):
        print(f"Exporting FP32 ONNX for quantization input -> {fp32_tmp}")
        fp32_model = _load_hat(model_name, device, half=False)
        _patch_forward_for_onnx(fp32_model, H, W, device)
        dummy32 = torch.randn(1, 3, H, W, dtype=torch.float32, device=device)
        with torch.no_grad():
            _ = fp32_model(dummy32)
        torch.cuda.synchronize(device)
        torch.onnx.export(
            fp32_model, dummy32, fp32_tmp,
            opset_version=18,
            input_names=["input"], output_names=["output"],
            do_constant_folding=True,
        )
        print(f"  FP32 ONNX: {os.path.getsize(fp32_tmp)/1e6:.1f} MB")
        del fp32_model
        torch.cuda.empty_cache()

    # ── 2. Calibration frames ─────────────────────────────────────────────
    print(f"Collecting {n_frames} calibration frames from {calib_input} ...")
    calib_frames = collect_calib_frames(calib_input, W, H, n_frames)

    # ── 3. Static INT8 quantization ────────────────────────────────────────
    class _HATCalibReader(CalibrationDataReader):
        def __init__(self, frames):
            self.frames = frames
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.frames):
                return None
            d = {"input": self.frames[self.idx]}
            self.idx += 1
            return d

        def rewind(self):
            self.idx = 0

    print(f"Quantizing to INT8 QDQ ONNX -> {int8_path}")
    print("  per_channel=True (one scale per output channel - best accuracy)")
    print("  quant_format=QDQ  (TRT-compatible QuantizeLinear/DequantizeLinear)")
    quantize_static(
        model_input=fp32_tmp,
        model_output=int8_path,
        calibration_data_reader=_HATCalibReader(calib_frames),
        quant_format=QuantFormat.QDQ,
        per_channel=True,          # per-output-channel → significantly better PSNR
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        optimize_model=True,
        use_external_data_format=False,
    )
    sz_int8 = os.path.getsize(int8_path) / 1e6
    sz_fp32 = os.path.getsize(fp32_tmp)  / 1e6
    print(f"  INT8 ONNX: {sz_int8:.1f} MB  "
          f"(FP32 was {sz_fp32:.1f} MB → {sz_fp32/sz_int8:.1f}× larger)")

    # Remove intermediate FP32 ONNX
    try:
        os.remove(fp32_tmp)
    except OSError:
        pass

    return int8_path


def benchmark_ort(onnx_path: str, W: int, H: int, runs: int = 10,
                  fp8: bool = False):
    """Run a quick throughput benchmark using ORT."""
    providers, backend = _try_ort_providers(fp8=fp8)
    if providers is None:
        print("\n  onnxruntime not installed.")
        print("  Install: pip install onnxruntime-gpu")
        print("  TRT support: also install tensorrt and onnxruntime-gpu with TRT extras.")
        return

    import onnxruntime as ort
    import time

    print(f"\nCreating ORT session ({backend}) ...")
    sess = ort.InferenceSession(onnx_path, providers=providers)

    dummy_np = np.random.rand(1, 3, H, W).astype(np.float16)

    # Warm-up
    print("  Warming up (2 runs) ...")
    for _ in range(2):
        sess.run(["output"], {"input": dummy_np})

    print(f"  Benchmarking {runs} runs ...")
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(["output"], {"input": dummy_np})
    elapsed = time.perf_counter() - t0
    fps = runs / elapsed
    print(f"\n  ORT {backend}  {W}x{H}:  {fps:.3f} fps  ({elapsed/runs*1000:.1f} ms/frame)")
    print(f"  vs PyTorch baseline ~0.08 fps  =>  {fps/0.08:.1f}x speedup")


# ── ORT inference class (used by inference_realesrgan_video.py) ────────────

class HATOrtInferencer:
    """
    Drop-in replacement for `upsampler.model(inp_batch)` that uses ONNX-Runtime
    with TensorRT (or CUDA) execution provider.

    Usage (inside inference_realesrgan_video.py):
        inferencer = HATOrtInferencer(onnx_path, device)
        out = inferencer(inp_fp16_batch)   # [B, 3, H_out, W_out], float16 tensor
    """

    def __init__(self, onnx_path: str, device: torch.device, fp8: bool = False):
        import onnxruntime as ort

        providers, backend = _try_ort_providers(fp8=fp8)
        if providers is None:
            raise RuntimeError("onnxruntime not installed — run: pip install onnxruntime-gpu")
        self._sess    = ort.InferenceSession(onnx_path, providers=providers)
        self._device  = device
        self._backend = backend
        self._fp8     = fp8
        # Label what precision mode is in use
        if fp8:
            precision = "FP8"
        elif "_int8" in os.path.basename(onnx_path):
            precision = "INT8"
        else:
            precision = "FP16"
        print(f"  HATOrtInferencer: {backend} [{precision}], "
              f"model={os.path.basename(onnx_path)}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] float16 GPU tensor
        Returns: [B, 3, H*4, W*4] float16 GPU tensor
        """
        x_np = x.cpu().numpy()
        # Process each item in the batch (ORT engine is fixed to batch=1)
        outs = [self._sess.run(["output"], {"input": x_np[i:i+1]})[0]
                for i in range(x_np.shape[0])]
        out_np = np.concatenate(outs, axis=0)
        return torch.from_numpy(out_np).to(self._device)


# ── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export HAT → ONNX + TRT benchmark")
    ap.add_argument("--model",  default="Real_HAT_GAN_SRx4_sharper",
                    help="Model name (must have .pth in tools/Real-ESRGAN/weights/)")
    ap.add_argument("--width",  type=int, default=None,
                    help="Input frame width (overrides --tile)")
    ap.add_argument("--height", type=int, default=None,
                    help="Input frame height (overrides --tile)")
    # ── Tile-mode export (recommended for tiled inference) ───────────────
    ap.add_argument("--tile", type=int, default=None,
                    help="Tile size used in inference (e.g. 384). "
                         "Computes export dimensions as ceil((tile+2*tile_pad)/16)*16. "
                         "If --width/--height are also given, those take precedence.")
    ap.add_argument("--tile-pad", dest="tile_pad", type=int, default=16,
                    help="Tile padding (default 16, must match --tile_pad in inference). "
                         "Only used when --tile is set and --width/--height are absent.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--benchmark-runs", type=int, default=10)
    ap.add_argument("--export-only", action="store_true",
                    help="Skip benchmark after export (used by upscale.py auto-export)")
    ap.add_argument("--cpu-export", dest="cpu_export", action="store_true",
                    help="Export the ONNX on CPU in FP32 (avoids 2x GPU peak-VRAM of JIT "
                         "tracer). TRT converts FP32 ONNX to FP16 engine automatically. "
                         "Recommended for GPUs with <= 10 GB VRAM.")
    # ── INT8 static quantization ──────────────────────────────────────────
    ap.add_argument("--int8", action="store_true",
                    help="Quantize exported ONNX to INT8 (static calibration). "
                         "Quantizes Conv2d + Linear → ~1.4-1.6× speedup vs FP16 TRT. "
                         "Requires --calib-input.")
    ap.add_argument("--calib-input", dest="calib_input", default=None,
                    help="Video file for INT8 calibration frames (required with --int8).")
    ap.add_argument("--calib-frames", dest="calib_frames", type=int, default=20,
                    help="Number of frames to use for INT8 calibration (default: 20).")
    # ── FP8 TRT (TensorRT 10+ / L40S Ada) ────────────────────────────────
    ap.add_argument("--fp8", action="store_true",
                    help="Enable TRT FP8 mode (trt_fp8_enable). Requires TensorRT >= 10 "
                         "and L40S / H100 / RTX 40xx Ada Lovelace GPU. "
                         "~1.8× GEMM throughput vs FP16.  Uses the FP16 ONNX + "
                         "TRT's internal activation-range profiling (no calibration data needed).")
    args = ap.parse_args()

    # ── Resolve export dimensions ─────────────────────────────────────────
    # Priority: explicit --width/--height  >  --tile/--tile-pad  >  default 1080×1920
    _exp_w, _exp_h = args.width, args.height
    if _exp_w is None or _exp_h is None:
        if args.tile is not None:
            _win   = 16
            _tgt   = ((args.tile + 2 * args.tile_pad + _win - 1) // _win) * _win
            _exp_w = _exp_h = _tgt
            print(f"Tile export: tile={args.tile}, tile_pad={args.tile_pad} "
                  f"-> target size {_tgt}x{_tgt}")
        else:
            _exp_w, _exp_h = 1080, 1920
            print(f"Full-frame export (default): {_exp_w}x{_exp_h}. "
                  f"Use --tile 384 --tile-pad 16 for tile-mode export.")

    device = torch.device(args.device)

    if args.int8:
        if not args.calib_input:
            print("\n  [ERROR] --int8 requires --calib-input <path-to-video>")
            print("  Example:")
            print(f"    python tools/export_hat_trt.py --model {args.model} "
                  f"--tile 384 --tile-pad 16 "
                  f"--int8 --calib-input input/video.mp4")
            sys.exit(1)
        onnx_path = export_onnx_int8(
            args.model, _exp_w, _exp_h, device,
            calib_input=args.calib_input,
            n_frames=args.calib_frames,
        )
    else:
        onnx_path = export_onnx(args.model, _exp_w, _exp_h, device,
                                 cpu_export=args.cpu_export)

    if not args.export_only:
        benchmark_ort(onnx_path, _exp_w, _exp_h,
                      runs=args.benchmark_runs, fp8=args.fp8)

    print("\nDone.")
    print(f"ONNX file: {onnx_path}")
    print()
    if args.fp8:
        print("FP8 mode active - TRT will build an FP8 engine on first inference.")
        print("Requires TensorRT >= 10.0 (pip install tensorrt) and L40S/H100/RTX-40xx.")
    if args.int8:
        print("INT8 QDQ ONNX - TRT builds a fully INT8 engine on first inference.")
    print()
    print("To use TRT inference in the upscale pipeline, pass to inference script:")
    print(f"  python tools/Real-ESRGAN/inference_realesrgan_video.py \\")
    print(f"    --ort-model {onnx_path}", end="")
    if args.fp8:
        print(" --fp8 \\")
    else:
        print(" \\")
    print(f"    [other args ...]")
    print()
    if args.tile:
        print(f"Or the pipeline auto-detects tile ONNX when --tile {args.tile} "
              f"--tile_pad {args.tile_pad} are set.")
    else:
        print("Or the pipeline auto-detects the ONNX file when tile=0 (default).")
    if args.int8:
        print("  Auto-detection priority: _int8.onnx > .onnx (fp16)")
