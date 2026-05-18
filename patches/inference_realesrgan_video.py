import argparse
import cv2
import glob
import mimetypes
import numpy as np
import os
import queue
import shutil
import subprocess
import sys
import threading
import torch
import torch.nn.functional as _F
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path as osp
from tqdm import tqdm

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


class _PaddedTileModel:
    """Wraps the compiled HAT model to ensure every tile has dimensions that are
    exact multiples of the model's window_size (16).

    RealESRGANer.tile_process() slices the image into tiles; edge tiles near the
    image boundary are SMALLER than (tile + 2*tile_pad) because the pad is clamped
    at the image edge.  HAT's window_partition requires H and W to be multiples of
    window_size=16, so any non-compliant tile would raise a shape error.

    This wrapper pads each tile (with reflect padding) to the target dimensions
    before calling the inner model, then crops the output back to the exact
    upscaled region that RealESRGANer expects.

    Usage (after torch.compile):
        win = 16
        target = ((tile + 2*tile_pad + win - 1) // win) * win   # e.g. 416
        upsampler.model = _PaddedTileModel(compiled_hat, target, target, scale=4)
    """
    def __init__(self, model, target_w: int, target_h: int, scale: int = 4):
        self.model    = model
        self.target_w = target_w
        self.target_h = target_h
        self.scale    = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ph = self.target_h - h   # ≥0; 0 for full-size interior tiles
        pw = self.target_w - w
        if ph > 0 or pw > 0:
            # reflect requires pad < input_dim; fall back to replicate for tiny edge tiles
            _pad_mode = 'reflect' if (ph < h and pw < w) else 'replicate'
            x = _F.pad(x, (0, pw, 0, ph), mode=_pad_mode)
        out = self.model(x)
        if ph > 0 or pw > 0:
            out = out[:, :, : h * self.scale, : w * self.scale]
        return out


class _OrtTileModel:
    """ORT-backed tile model — same pad/crop contract as _PaddedTileModel but
    forwards each tile through an ONNX-Runtime session (TRT or CUDA EP).

    The ONNX session expects fixed (1, 3, target_h, target_w) FP16 input.
    Edge tiles are padded to that size with reflect padding, and the output is
    cropped back to (h*scale, w*scale) before returning.

    Called by RealESRGANer.tile_process() via ``self.model(input_tile)``.
    """
    def __init__(self, sess, target_w: int, target_h: int,
                 scale: int = 4, device: torch.device = None):
        self._sess     = sess
        self.target_w  = target_w
        self.target_h  = target_h
        self.scale     = scale
        self._device   = device
        # Detect whether the ONNX model expects float16 or float32 input.
        # export_onnx() with cpu_export=True produces float32; CUDA export gives float16.
        _inp_type = sess.get_inputs()[0].type  # e.g. "tensor(float16)" or "tensor(float)"
        self._np_dtype = "float16" if "float16" in _inp_type else "float32"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ph = self.target_h - h
        pw = self.target_w - w
        if ph > 0 or pw > 0:
            x = _F.pad(x, (0, pw, 0, ph), mode='reflect')
        # Cast to the dtype the ONNX model expects (float16 for CUDA export,
        # float32 for CPU/cpu_export=True export).
        if self._np_dtype == "float16":
            x_np = x.half().cpu().numpy()
        else:
            x_np = x.float().cpu().numpy()
        out_np = self._sess.run(["output"], {"input": x_np})[0]
        # Return on device; keep whatever dtype ORT produced (usually matches input)
        out = torch.from_numpy(out_np).to(self._device)
        if self._np_dtype == "float32":
            out = out.half()   # upsampler expects FP16 downstream
        if ph > 0 or pw > 0:
            out = out[:, :, : h * self.scale, : w * self.scale]
        return out

# Frames to batch per GPU pass (tile=0 only).  Overridden inside inference_video()
# by VRAM-adaptive logic.  This module-level value is a safe fallback.
UPSCALE_BATCH = 8

# Detect xformers. When active we rely on torch.compile's internal CUDA-graph
# mechanism instead of our manual CUDAGraph capture (xformers Triton kernels
# use workspace allocation that breaks manual torch.cuda.CUDAGraph).
try:
    import xformers.ops as _xf  # noqa  (side-effect: registers FA2 kernels in basicsr)
    _XFORMERS_ACTIVE = True
except ImportError:
    _XFORMERS_ACTIVE = False

# NVENC availability probe (lazy, first Writer instantiation).
# hevc_nvenc is on a dedicated NVENC engine on L40S — it consumes zero CUDA
# cores, so encoding is truly free while the GPU is upscaling.
_NVENC_AVAILABLE = None

def _probe_nvenc(ffmpeg_bin: str) -> bool:
    """Return True if hevc_nvenc encoder is present in the FFmpeg binary."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        r = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        _NVENC_AVAILABLE = "hevc_nvenc" in r.stdout
    except Exception:
        _NVENC_AVAILABLE = False
    tag = "hevc_nvenc (NVENC H.265)" if _NVENC_AVAILABLE else "libx264 (CPU H.264)"
    print(f"  Writer encoder: {tag}", flush=True)
    return _NVENC_AVAILABLE

try:
    import ffmpeg
except ImportError:
    import pip
    pip.main(['install', '--user', 'ffmpeg-python'])
    import ffmpeg


def get_video_meta_info(video_path):
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    _nb = video_streams[0].get('nb_frames') or video_streams[0].get('nb_read_packets')
    if _nb is None:
        # MKV and some other containers don't store nb_frames in stream headers;
        # fall back to duration * fps (may be off by ±1 frame).
        _dur = float(video_streams[0].get('duration') or
                     probe.get('format', {}).get('duration', 0))
        _nb = int(round(_dur * ret['fps'])) if _dur else 0
    ret['nb_frames'] = int(_nb)
    return ret


def get_sub_video(args, num_process, process_idx):
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    print(f'duration: {duration}, part_time: {part_time}')
    os.makedirs(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'), exist_ok=True)
    out_path = osp.join(args.output, f'{args.video_name}_inp_tmp_videos', f'{process_idx:03d}.mp4')
    cmd = [
        args.ffmpeg_bin, f'-i {args.input}', '-ss', f'{part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '', '-async 1', out_path, '-y'
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path


class Reader:

    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']

        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])
            self.width, self.height = tmp_img.size
        self.idx = 0

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.get_frame_from_stream()
        else:
            return self.get_frame_from_list()

    def close(self):
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            # Close stdout BEFORE wait() to prevent pipe-full deadlock:
            # ffmpeg blocks writing to its stdout pipe when the buffer fills up
            # (e.g. probe reader that never reads frames).  Closing the read end
            # delivers SIGPIPE so ffmpeg exits and wait() returns immediately.
            try:
                self.stream_reader.stdout.close()
            except Exception:
                pass
            self.stream_reader.wait()


class Writer:

    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        # hevc_nvenc runs on the dedicated NVENC engine, but its BGR→YUV
        # colour-conversion pipeline still allocates CUDA VRAM.  For outputs
        # larger than 4K this competes with the upscaler model on 8 GB cards
        # and causes CUDA OOM in the encoder after ~3 frames.  Force CPU
        # (libx264) for >4K intermediates so the upscaler gets all the VRAM.
        _over_4k = (out_width * out_height) > (3840 * 2160)
        use_nvenc = (not _over_4k) and _probe_nvenc(args.ffmpeg_bin)
        if _over_4k:
            print(f'  Writer: >4K output ({out_width}x{out_height}), near-lossless ultrafast '
                  f'intermediate (CPU x264 crf=1)', flush=True)
        if use_nvenc:
            # CQ 18 = high-quality intermediate (re-encoded to CQ 14 in final stage).
            # -preset p4 = balanced quality/speed; p6/p7 are slower with minimal gain
            # for intermediate files.
            vcodec_kwargs = dict(
                vcodec="hevc_nvenc",
                preset="p4",
                rc="vbr",
                cq=18,
                pix_fmt="yuv420p",
                loglevel="error",
            )
        elif _over_4k:
            # Intermediate >4K: near-lossless ultrafast crf=1 — visually indistinguishable
            # from lossless (PSNR ~60dB; final CRF=18 encode dominates quality budget).
            # crf=1 encodes 5-15x faster than crf=0 (lossless) at 8K resolution.
            vcodec_kwargs = dict(
                vcodec="libx264",
                preset="ultrafast",
                crf=1,
                pix_fmt="yuv420p",
                loglevel="error",
            )
        else:
            vcodec_kwargs = dict(
                vcodec="libx264",
                crf=18,
                preset="medium",
                pix_fmt="yuv420p",
                loglevel="error",
            )

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24',
                             s=f'{out_width}x{out_height}', framerate=fps)
                .output(audio, video_save_path, acodec='copy', **vcodec_kwargs)
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24',
                             s=f'{out_width}x{out_height}', framerate=fps)
                .output(video_save_path, **vcodec_kwargs)
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

        # Async encoder queue: GPU inference puts frames here and continues
        # immediately; a background thread feeds the x264 pipe in parallel.
        # Queue depth of 8 frames gives ~0.7 s headroom at 12 fps before stalling.
        self._enc_error: Exception | None = None
        self._enc_queue: queue.Queue = queue.Queue(maxsize=8)
        self._enc_thread = threading.Thread(target=self._encode_loop, daemon=True)
        self._enc_thread.start()

    def _encode_loop(self):
        try:
            while True:
                item = self._enc_queue.get()
                if item is None:  # sentinel — close() signals done
                    break
                self.stream_writer.stdin.write(item)
        except (BrokenPipeError, OSError) as e:
            self._enc_error = RuntimeError(f"Writer pipe broken (encoder crashed?): {e}")

    def write_frame(self, frame):
        if self._enc_error is not None:
            raise self._enc_error
        self._enc_queue.put(frame.astype(np.uint8).tobytes())  # blocks only when queue full

    def close(self):
        if self._enc_error is not None:
            raise self._enc_error
        self._enc_queue.put(None)   # sentinel: tell encoder thread to exit
        self._enc_thread.join()     # wait for all frames to be written
        if self._enc_error is not None:
            raise self._enc_error
        self.stream_writer.stdin.close()
        self.stream_writer.wait()


def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
    # ---------------------- determine models according to model names ---------------------- #
    args.model_name = args.model_name.split('.pth')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    elif args.model_name in ('Real_HAT_GAN_SRx4', 'Real_HAT_GAN_SRx4_sharper'):  # x4 HAT model
        try:
            # Use the local (patched) hat_arch — already registered at module import time
            # via realesrgan.archs.__init__. Using basicsr.archs.hat_arch would trigger a
            # second @ARCH_REGISTRY.register() call → AssertionError "HAT already registered".
            from realesrgan.archs.hat_arch import HAT
        except ImportError:
            raise RuntimeError(
                "HAT architecture not found. "
                "Ensure realesrgan/archs/hat_arch.py is present and realesrgan is importable.")
        model = HAT(
            upscale=4, in_chans=3, img_size=64, window_size=16,
            compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
            overlap_ratio=0.5, img_range=1., depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
            upsampler='pixelshuffle', resi_connection='1conv')
        netscale = 4
        file_url = []  # no auto-download; place model in weights/ manually

    # ---------------------- determine model paths ---------------------- #
    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        device=device,
    )

    # ---- ONNX / TRT inference path (auto-detected or explicit --ort-model) ----
    _ort_inferencer = None
    _ort_model_path = getattr(args, "ort_model", None)
    _ort_fp8        = getattr(args, "fp8", False)   # passed via --fp8 CLI flag

    # Auto-detect: prefer INT8 ONNX (faster), then FP16 ONNX.
    # Files are named: {model}_{w}x{h}_int8.onnx  and  {model}_{w}x{h}.onnx
    _weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    if _ort_model_path is None and args.tile == 0:
        reader_probe = Reader(args, total_workers, worker_idx)
        _ph, _pw = reader_probe.get_resolution()
        reader_probe.close()
        _auto_int8    = os.path.join(_weights_dir, f"{args.model_name}_{_pw}x{_ph}_int8.onnx")
        _auto_fp16    = os.path.join(_weights_dir, f"{args.model_name}_{_pw}x{_ph}.onnx")
        if os.path.isfile(_auto_int8):
            _ort_model_path = _auto_int8
            _ort_fp8 = False   # INT8 and FP8 are mutually exclusive
            print(f"  Upscale: auto-detected INT8 ONNX -> {_auto_int8}", flush=True)
        elif os.path.isfile(_auto_fp16):
            _ort_model_path = _auto_fp16
            print(f"  Upscale: auto-detected FP16 ONNX -> {_auto_fp16}", flush=True)
    elif _ort_model_path is None and args.tile > 0:
        # Tile-mode ONNX: model exported at the padded tile target size.
        # target = ceil((tile + 2*tile_pad) / 16) * 16  →  e.g. 416×416
        _tw = 16
        _tile_target = ((args.tile + 2 * args.tile_pad + _tw - 1) // _tw) * _tw
        _auto_tile = os.path.join(_weights_dir,
                                  f"{args.model_name}_{_tile_target}x{_tile_target}.onnx")
        if os.path.isfile(_auto_tile):
            _ort_model_path = _auto_tile
            print(f"  Upscale: auto-detected tile ONNX ({_tile_target}x{_tile_target}) "
                  f"-> {_auto_tile}", flush=True)

    if _ort_model_path is not None:
        try:
            import onnxruntime as _ort_mod
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
            from tools.export_hat_trt import _try_ort_providers
            _providers, _ort_backend = _try_ort_providers(fp8=_ort_fp8)
            if args.tile > 0:
                # Per-tile ORT path: the ONNX model runs at fixed target_w × target_h.
                # _OrtTileModel pads edge tiles and crops output, same as _PaddedTileModel.
                _sess = _ort_mod.InferenceSession(_ort_model_path, providers=_providers)
                _tw2 = 16
                _tile_target2 = ((args.tile + 2 * args.tile_pad + _tw2 - 1) // _tw2) * _tw2
                upsampler.model = _OrtTileModel(
                    _sess, _tile_target2, _tile_target2, scale=netscale, device=device)
                print(f"  Upscale: ORT tile inference ({_ort_backend}), "
                      f"target={_tile_target2}x{_tile_target2}", flush=True)
                _ort_inferencer = None   # not used in _run_batch; upsampler.model is the ORT model
            else:
                from tools.export_hat_trt import HATOrtInferencer
                _ort_inferencer = HATOrtInferencer(_ort_model_path, device, fp8=_ort_fp8)
                print(f"  Upscale: using ONNX-Runtime inference (TRT or CUDA)", flush=True)
        except Exception as _oe:
            print(f"  Upscale: ONNX-Runtime unavailable ({_oe}), falling back to PyTorch", flush=True)
            _ort_inferencer = None

    # ---- speed optimisations ----
    torch.backends.cudnn.benchmark    = True
    torch.backends.cuda.matmul.allow_tf32 = True   # ~3x faster FP32 accum on Ampere+; no quality loss
    torch.backends.cudnn.allow_tf32       = True

    # VRAM-adaptive batch: HAT at 1920x1080 tile=0 peaks ~4 GB/frame (feature maps + attention).
    # max-autotune-no-cudagraphs pre-allocates ALL intermediate buffers simultaneously.
    # Use _uvram/8 capped at 4: RTX 3070 8 GB → 1 | RTX 4090 24 GB → 3 | RTX 5090 32 GB → 4
    _uvram = (torch.cuda.get_device_properties(device).total_memory / 1e9
              if device is not None and str(device).startswith('cuda')
              else (torch.cuda.get_device_properties(0).total_memory / 1e9
                    if torch.cuda.is_available() else 0))
    UPSCALE_BATCH = min(4, max(1, int(_uvram / 8)))
    print(f"  Upscale: UPSCALE_BATCH={UPSCALE_BATCH} ({_uvram:.0f} GB VRAM detected)", flush=True)

    if _ort_inferencer is None:
        # FP8 dynamic quantisation for Blackwell (sm_120+) only.
        # HAT attention precision is more sensitive than conv models; restrict to Blackwell
        # where MX-FP8 Tensor Cores give genuine 2x throughput over FP16.
        _cc2 = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        if _cc2 and _cc2.major >= 12:
            try:
                from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
                quantize_(upsampler.model, Float8DynamicActivationFloat8WeightConfig())
                print(f"  Upscale: FP8 quantisation applied (Blackwell sm_{_cc2.major}{_cc2.minor})", flush=True)
            except Exception as _fp8e:
                print(f"  Upscale: FP8 unavailable \u2014 {type(_fp8e).__name__} (install torchao)", flush=True)

        # Ensure model.mean is on CUDA with correct dtype BEFORE compile.
        # mean is a plain tensor (not register_buffer), so model.half() skips it.
        # Without this, the dtype mismatch causes a graph break under reduce-overhead.
        if hasattr(upsampler.model, 'mean'):
            upsampler.model.mean = upsampler.model.mean.to(
                device=upsampler.device, dtype=torch.float16)

        # INT8 weight-only quantization via torchao (sm_86+: RTX 3070 compatible).
        # Converts Conv2d/Linear weight tensors to int8 scales; activations stay fp16.
        # Speedup is modest (~0.9% on HAT's mostly-attention workload) but zero quality loss.
        try:
            from torchao.quantization import quantize_, int8_weight_only
            quantize_(upsampler.model, int8_weight_only())
            print("  Upscale: INT8 weight-only quantization applied (torchao)", flush=True)
        except Exception as _int8e:
            print(f"  Upscale: INT8 skipped - {_int8e}", flush=True)

        # Inductor GEMM coordinate-descent tuning: profile GEMM kernel configs once,
        # cache the winners — 5-15% on attention/MLP matmuls inside HAT.
        # epilogue_fusion: fuse pointwise ops after GEMM into the same kernel.
        try:
            import torch._inductor.config as _ic
            _ic.coordinate_descent_tuning = True
            _ic.epilogue_fusion = True
            _ic.coordinate_descent_search_radius = 1
            print("  Upscale: inductor GEMM coordinate-descent tuning + epilogue fusion enabled",
                  flush=True)
        except Exception:
            pass

        # torch.compile: mode controlled by --compile-mode (default: max-autotune-no-cudagraphs).
        # max-autotune-no-cudagraphs: exhaustive Triton kernel search, no CUDA graphs.
        #   Longer first warmup (10-30 min, cached); faster per-tile steady state.
        # reduce-overhead: CUDA graph capture, one warmup pass (~30-90 s).
        # Flash SDPA explicitly enabled; cuDNN SDPA uses native FA3 on Blackwell:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # FA2 supersedes this
        torch.backends.cuda.enable_cudnn_sdp(True)           # cuDNN FA3 on sm_120+ (Blackwell)
        _cmode     = getattr(args, 'compile_mode', 'max-autotune-no-cudagraphs')
        _fullgraph = getattr(args, 'fullgraph', False)
        _use_cudagraphs = (_cmode == 'reduce-overhead')
        try:
            _compiled_hat = torch.compile(
                upsampler.model, mode=_cmode, dynamic=False, fullgraph=_fullgraph)
            _cmode_label = 'CUDA graphs' if _use_cudagraphs else 'Triton kernel search'
            print(f"  Upscale: torch.compile({_cmode}, fullgraph={_fullgraph}) — {_cmode_label}", flush=True)
        except Exception as _ce:
            _compiled_hat = upsampler.model
            print(f"  Upscale: torch.compile failed ({_ce}), using eager", flush=True)

        # PaddedTileModel: pads edge tiles to target dimensions (next multiple of
        # window_size=16 that fits tile+2*tile_pad).  For tile=384 + tile_pad=16:
        # target=416 (26×16).  Interior tiles are already 416×416; edge tiles
        # (e.g. 328×416) are reflect-padded up then output is cropped back.
        # This is only needed for the tiled inference path (args.tile > 0).
        if args.tile > 0:
            _win    = 16   # HAT window_size
            _target = ((args.tile + 2 * args.tile_pad + _win - 1) // _win) * _win
            _orig_hat = upsampler.model  # original uncompiled model — needed for mask pre-warm
            upsampler.model = _PaddedTileModel(_compiled_hat, _target, _target, scale=netscale)
            print(f"  Upscale: _PaddedTileModel target={_target}x{_target} (all tiles uniform)", flush=True)

            # --- Tile batching: process N tiles per forward pass ---
            # _PaddedTileModel ensures all tiles pad to _target×_target, so a batch of
            # N tiles gives an [N, C, _target, _target] tensor → single model call.
            # VRAM budget: empirical ~12-13 GB/tile on RTX 5090 (31 GB) at tile=512.
            # Formula: max(1, int(_uvram / 14)) → 2 tiles on ≥28 GB, 1 tile on <14 GB.
            _TILE_BATCH = max(1, int(_uvram / 14))
            print(f"  Upscale: TILE_BATCH={_TILE_BATCH} "
                  f"(tile batching: {_TILE_BATCH} tiles per forward pass)", flush=True)

            import types as _types_tb, math as _math_tb

            def _make_batched_tile_process(tile_batch, inner_model, tile_scale, t_h, t_w):
                """Return a tile_process method that batches up to tile_batch tiles."""
                from torch.nn import functional as _Ftb

                def _tile_process(self):
                    bat, ch, height, width = self.img.shape
                    self.output = self.img.new_zeros(
                        (bat, ch, height * tile_scale, width * tile_scale))
                    tiles_x = _math_tb.ceil(width  / self.tile_size)
                    tiles_y = _math_tb.ceil(height / self.tile_size)

                    tiles, placements = [], []
                    for y in range(tiles_y):
                        for x in range(tiles_x):
                            # Core output region for this tile (placement target)
                            osx_in = x * self.tile_size
                            oex_in = min(osx_in + self.tile_size, width)
                            osy_in = y * self.tile_size
                            oey_in = min(osy_in + self.tile_size, height)
                            # Slide back edge tiles to keep full tile_size input window:
                            # tiny edge tiles (e.g. 64px wide) would need oversized reflect
                            # padding (464px for 64→528) which PyTorch rejects; with
                            # full-size window we always have ≤tile_pad px to reflect-pad.
                            isx = osx_in if (oex_in - osx_in) == self.tile_size \
                                else max(0, width - self.tile_size)
                            iex = isx + min(self.tile_size, width)
                            isy = osy_in if (oey_in - osy_in) == self.tile_size \
                                else max(0, height - self.tile_size)
                            iey = isy + min(self.tile_size, height)
                            isx_p = max(isx - self.tile_pad, 0)
                            iex_p = min(iex + self.tile_pad, width)
                            isy_p = max(isy - self.tile_pad, 0)
                            iey_p = min(iey + self.tile_pad, height)
                            raw = self.img[:, :, isy_p:iey_p, isx_p:iex_p]
                            h_r, w_r = raw.shape[-2], raw.shape[-1]
                            ph, pw = t_h - h_r, t_w - w_r
                            if ph > 0 or pw > 0:
                                _pad_mode = 'reflect' if (ph < h_r and pw < w_r) else 'replicate'
                                raw = _Ftb.pad(raw, (0, pw, 0, ph), _pad_mode)
                            tiles.append(raw)
                            placements.append((
                                osy_in * tile_scale, oey_in * tile_scale,
                                osx_in * tile_scale, oex_in * tile_scale,
                                (osx_in - isx_p) * tile_scale,
                                (oex_in - isx_p) * tile_scale,
                                (osy_in - isy_p) * tile_scale,
                                (oey_in - isy_p) * tile_scale,
                            ))

                    for start in range(0, len(tiles), tile_batch):
                        chunk = tiles[start:start + tile_batch]
                        places = placements[start:start + tile_batch]
                        inp = torch.cat(chunk, dim=0)  # [N, C, t_h, t_w]
                        try:
                            with torch.no_grad():
                                torch.compiler.cudagraph_mark_step_begin()
                                outs = inner_model(inp)  # [N, C, t_h*scale, t_w*scale]
                        except RuntimeError as _e:
                            print(f'  Tile batch OOM (N={len(chunk)}): {_e}; '
                                  f'falling back to N=1', flush=True)
                            outs_list = []
                            for t in chunk:
                                torch.compiler.cudagraph_mark_step_begin()
                                outs_list.append(inner_model(t))
                            outs = torch.cat(outs_list, dim=0)
                        for j, (osy, oey, osx, oex, ostx, oetx, osty, oety) in enumerate(places):
                            self.output[:, :, osy:oey, osx:oex] = \
                                outs[j:j+1, :, osty:oety, ostx:oetx]

                return _tile_process

            def _patch_tile_process(inner):
                upsampler.tile_process = _types_tb.MethodType(
                    _make_batched_tile_process(
                        _TILE_BATCH, inner, netscale, _target, _target),
                    upsampler)

            _patch_tile_process(_compiled_hat)

            # Pre-warm HAT _mask_cache on the UNCOMPILED model before CUDA graph capture.
            # torch.compile is lazy — first actual call triggers compilation/capture.
            # By running uncompiled forward once, _mask_cache/_mask_cache_size are set.
            # The compiled graph then traces the "cache hit" branch: reads _mask_cache
            # as a module constant rather than writing it as a graph output buffer.
            # This prevents "tensor output overwritten by subsequent run" on replay.
            try:
                print(f"  Upscale: pre-warming mask cache ({_target}×{_target})...", flush=True)
                with torch.no_grad():
                    _mwup = torch.rand(1, 3, _target, _target, device=upsampler.device, dtype=torch.float16)
                    _orig_hat(_mwup)  # run ORIGINAL (uncompiled) model to set _mask_cache
                    del _mwup
                del _orig_hat
                torch.cuda.empty_cache()
                print("  Upscale: mask cache ready", flush=True)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as _pe:
                print(f"  Upscale: mask pre-warm OOM ({type(_pe).__name__}); "
                      f"reverting to eager FP8", flush=True)
                _compiled_hat = upsampler.model
                _patch_tile_process(_compiled_hat)
                if '_orig_hat' in dir():
                    del _orig_hat
                torch.cuda.empty_cache()

            # CUDA graph warmup: trigger compilation/capture before inference starts.
            # NOTE: dynamic-activation FP8 (Float8DynamicActivationFloat8WeightConfig)
            # computes per-tensor max(|activation|) at runtime — this data-dependent op
            # breaks CUDA graph capture.  Catch the failure and revert to eager FP8.
            try:
                if _compiled_hat is upsampler.model:
                    raise RuntimeError("pre-warm reverted to eager; skipping CUDA graph warmup")
                # TILE_BATCH=3 probe: FP8 weights reduce per-tile VRAM vs the FP16-era
                # 14 GB/tile empirical formula. The first call triggers Dynamo compilation
                # for that batch shape; if OOM the graph is discarded, stay at batch=2.
                if _TILE_BATCH == 2:
                    try:
                        _probe3 = torch.rand(3, 3, _target, _target,
                                             device=upsampler.device, dtype=torch.float16)
                        with torch.no_grad():
                            _compiled_hat(_probe3)
                        del _probe3
                        torch.cuda.empty_cache()
                        _TILE_BATCH = 3
                        _patch_tile_process(_compiled_hat)
                        print("  Upscale: TILE_BATCH=3 probe succeeded — bumped from 2",
                              flush=True)
                    except (torch.cuda.OutOfMemoryError, RuntimeError):
                        torch.cuda.empty_cache()
                        print("  Upscale: TILE_BATCH=3 probe OOM — staying at 2", flush=True)
                # TILE_BATCH=4 probe: ~1.5 GB headroom on RTX 5090 at B=3 (29.9/31.4 GB).
                if _TILE_BATCH == 3:
                    try:
                        _probe4 = torch.rand(4, 3, _target, _target,
                                             device=upsampler.device, dtype=torch.float16)
                        with torch.no_grad():
                            _compiled_hat(_probe4)
                        del _probe4
                        torch.cuda.empty_cache()
                        _TILE_BATCH = 4
                        _patch_tile_process(_compiled_hat)
                        print("  Upscale: TILE_BATCH=4 probe succeeded — bumped from 3",
                              flush=True)
                    except (torch.cuda.OutOfMemoryError, RuntimeError):
                        torch.cuda.empty_cache()
                        print("  Upscale: TILE_BATCH=4 probe OOM — staying at 3", flush=True)
                _warmup_label = 'CUDA graph' if _use_cudagraphs else 'compile'
                print(f"  Upscale: {_warmup_label} warmup ({_target}x{_target} × B={_TILE_BATCH})...",
                      flush=True)
                _wup = torch.rand(_TILE_BATCH, 3, _target, _target,
                                  device=upsampler.device, dtype=torch.float16)
                with torch.no_grad():
                    for _ in range(3):
                        _compiled_hat(_wup)
                if _TILE_BATCH > 1:
                    _wup1 = torch.rand(1, 3, _target, _target,
                                       device=upsampler.device, dtype=torch.float16)
                    with torch.no_grad():
                        for _ in range(3):
                            _compiled_hat(_wup1)
                    del _wup1
                torch.cuda.synchronize()
                del _wup
                _warmup_done_label = 'CUDA graphs active' if _use_cudagraphs else 'Triton kernels ready'
                print(f"  Upscale: warmup complete — {_warmup_done_label}", flush=True)
            except Exception as _we:
                print(f"  Upscale: compile/warmup failed ({type(_we).__name__}: {_we}); "
                      f"reverting to eager FP8", flush=True)
                _compiled_hat = upsampler.model   # back to uncompiled FP8 model
                _patch_tile_process(_compiled_hat)  # re-patch closure with eager model
        else:
            upsampler.model = _compiled_hat

    if 'anime' in args.model_name and args.face_enhance:
        print('face_enhance is not supported in anime models, we turned this option off for you. '
              'if you insist on turning it on, please manually comment the relevant lines of code.')
        args.face_enhance = False

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)  # TODO support custom device
    else:
        face_enhancer = None

    reader = Reader(args, total_workers, worker_idx)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    writer = Writer(args, audio, height, width, video_save_path, fps)

    # Helper functions for batch inference (tile=0 path)
    def _pre(img):
        """BGR uint8 numpy -> [C, H, W] FP16 tensor on device.
        Transfers uint8 (3 bytes/px) vs float32 (12 bytes/px) — 4x less PCIe.
        All channel-flip + normalize + half ops execute on the GPU.
        """
        # torch.from_numpy shares memory (no CPU copy); .to() is the PCIe transfer
        t = torch.from_numpy(img).to(upsampler.device, non_blocking=True)  # [H,W,3] uint8
        return t.permute(2, 0, 1)[[2, 1, 0]].half().mul_(1.0 / 255.0)     # [C,H,W] fp16

    def _post(t):
        """[C, H_out, W_out] FP16 tensor on device -> BGR uint8 numpy.
        All processing stays on GPU; only a compact uint8 frame is copied to CPU.
        """
        t = t.float().clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)  # GPU uint8
        t = t[[2, 1, 0]].permute(1, 2, 0)                                    # RGB->BGR [H,W,3]
        return t.contiguous().cpu().numpy()                                   # uint8 PCIe

    _use_batch = (args.tile == 0 and not args.face_enhance)
    _buf = []

    # CUDA Graph state — captured after the first real batch (warm-up for compile)
    _cg = None           # CUDAGraph object
    _cg_inp = None       # static input tensor (pinned in the graph)
    _cg_out = None       # static output tensor
    _cg_batch = -1       # batch size the graph was captured for
    _warmup_done = False

    def _run_batch(inp):
        """Run the model, using ORT/TRT if available, else CUDA Graph after warm-up.
        When xformers is active we skip manual CUDAGraph (torch.compile handles it
        internally via cudagraphify; xformers Triton kernels break manual capture).
        """
        nonlocal _cg, _cg_inp, _cg_out, _cg_batch, _warmup_done
        with torch.no_grad():
            # ORT/TRT path — no CUDA Graph needed (ORT manages its own engine)
            if _ort_inferencer is not None:
                return _ort_inferencer(inp)

            # First call: warm-up (torch.compile traces here)
            if not _warmup_done:
                out = upsampler.model(inp)
                torch.cuda.synchronize(upsampler.device)
                _warmup_done = True
                return out

            # If xformers is active, torch.compile's internal graphs are sufficient.
            # Manual CUDAGraph conflicts with xformers workspace allocation.
            if _XFORMERS_ACTIVE:
                return upsampler.model(inp)

            # Second call with same batch size: capture CUDA Graph
            if _cg is None or (_cg is not False and _cg_batch != inp.shape[0]):
                _cg_batch = inp.shape[0]
                _cg_inp = inp.clone()
                try:
                    _cg = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(_cg):
                        _cg_out = upsampler.model(_cg_inp)
                    torch.cuda.synchronize(upsampler.device)
                    print(f"  Upscale: CUDA Graph captured (batch={_cg_batch})", flush=True)
                except Exception as _cg_err:
                    print(f"  Upscale: CUDA Graph capture failed ({_cg_err}), using eager", flush=True)
                    _cg = False  # sentinel: skip capture on future calls

            if _cg is False:
                return upsampler.model(inp)

            # Replay graph: just copy data and re-use the captured kernel stream
            _cg_inp.copy_(inp)
            _cg.replay()
            return _cg_out

    pbar = tqdm(total=len(reader), unit='frame', desc='inference')
    while True:
        img = reader.get_frame()

        if not _use_batch:
            # ---- original single-frame path (tiled or face_enhance) ----
            if img is None:
                break
            try:
                if args.face_enhance:
                    _, _, output = face_enhancer.enhance(
                        img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    _pre_ds = getattr(args, 'pre_downscale', 1.0)
                    if _pre_ds != 1.0:
                        _ph, _pw = img.shape[:2]
                        import cv2 as _cv2_ds
                        img = _cv2_ds.resize(img, (int(_pw / _pre_ds), int(_ph / _pre_ds)),
                                             interpolation=_cv2_ds.INTER_AREA)
                    output, _ = upsampler.enhance(img, outscale=args.outscale)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            else:
                writer.write_frame(output)
            torch.cuda.synchronize(device)
            pbar.update(1)
            continue

        # ---- batched full-frame path ----
        if img is not None:
            _buf.append(img)

        flush = (img is None) or (len(_buf) >= UPSCALE_BATCH)
        if flush and _buf:
            inp = torch.stack([_pre(f) for f in _buf])  # [B, C, H, W]
            try:
                out_batch = _run_batch(inp)
                torch.cuda.synchronize(upsampler.device)
                for out_t in out_batch:
                    writer.write_frame(_post(out_t))
                    pbar.update(1)
            except RuntimeError as error:
                print(f'Batch error (batch={len(_buf)}): {error}')
                print('Falling back to single-frame. Consider --tile 512 to reduce memory.')
                _cg = None  # invalidate graph on error
                for f in _buf:
                    try:
                        _pre_ds2 = getattr(args, 'pre_downscale', 1.0)
                        if _pre_ds2 != 1.0:
                            _fh, _fw = f.shape[:2]
                            import cv2 as _cv2_ds2
                            f = _cv2_ds2.resize(f, (int(_fw / _pre_ds2), int(_fh / _pre_ds2)),
                                                interpolation=_cv2_ds2.INTER_AREA)
                        o, _ = upsampler.enhance(f, outscale=args.outscale)
                        writer.write_frame(o)
                    except Exception as e2:
                        print(f'  Single-frame fallback also failed: {e2}')
                    pbar.update(1)
            _buf = []

        if img is None:
            break

    reader.close()
    writer.close()


def run(args):
    args.video_name = osp.splitext(os.path.basename(args.input))[0]
    video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_videos')
        os.makedirs(tmp_frames_folder, exist_ok=True)
        os.system(f'ffmpeg -i {args.input} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {tmp_frames_folder}/frame%08d.png')
        args.input = tmp_frames_folder

    num_gpus = torch.cuda.device_count()
    num_process = num_gpus * args.num_process_per_gpu
    if num_process == 1:
        inference_video(args, video_save_path)
        return

    ctx = torch.multiprocessing.get_context('spawn')
    pool = ctx.Pool(num_process)
    os.makedirs(osp.join(args.output, f'{args.video_name}_out_tmp_videos'), exist_ok=True)
    pbar = tqdm(total=num_process, unit='sub_video', desc='inference')
    for i in range(num_process):
        sub_video_save_path = osp.join(args.output, f'{args.video_name}_out_tmp_videos', f'{i:03d}.mp4')
        pool.apply_async(
            inference_video,
            args=(args, sub_video_save_path, torch.device(i % num_gpus), num_process, i),
            callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()

    # combine sub videos
    # prepare vidlist.txt
    with open(f'{args.output}/{args.video_name}_vidlist.txt', 'w') as f:
        for i in range(num_process):
            f.write(f'file \'{args.video_name}_out_tmp_videos/{i:03d}.mp4\'\n')

    cmd = [
        args.ffmpeg_bin, '-f', 'concat', '-safe', '0', '-i', f'{args.output}/{args.video_name}_vidlist.txt', '-c',
        'copy', f'{video_save_path}'
    ]
    print(' '.join(cmd))
    subprocess.call(cmd)
    shutil.rmtree(osp.join(args.output, f'{args.video_name}_out_tmp_videos'))
    if osp.exists(osp.join(args.output, f'{args.video_name}_inp_tmp_videos')):
        shutil.rmtree(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'))
    os.remove(f'{args.output}/{args.video_name}_vidlist.txt')


def main():
    """Inference demo for Real-ESRGAN.
    It mainly for restoring anime videos.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video, image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='realesr-animevideov3',
        help=('Model names: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | RealESRNet_x4plus |'
              ' RealESRGAN_x2plus | realesr-general-x4v3'
              'Default:realesr-animevideov3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--compile-mode', dest='compile_mode', type=str,
        default='max-autotune-no-cudagraphs',
        choices=['reduce-overhead', 'max-autotune-no-cudagraphs', 'max-autotune', 'default', 'eager'],
        help='torch.compile mode. max-autotune-no-cudagraphs: full Triton search (slower first run, faster steady state). reduce-overhead: CUDA graph capture.')
    parser.add_argument('--fullgraph', action='store_true', default=False,
        help='torch.compile fullgraph=True — forces no Python graph breaks. May error if HAT has control flow.')
    parser.add_argument('--pre-downscale', dest='pre_downscale', type=float, default=1.0,
        help='Downscale input frames by this factor before HAT (e.g. 1.5 = 1080p→720p, ~2x faster, slight quality tradeoff).')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='The path to ffmpeg')
    parser.add_argument('--extract_frame_first', action='store_true')
    parser.add_argument('--num_process_per_gpu', type=int, default=1)
    parser.add_argument(
        '--ort-model', dest='ort_model', type=str, default=None,
        help='Path to a pre-exported .onnx file for ONNX-Runtime / TensorRT inference.')
    parser.add_argument(
        '--fp8', action='store_true', default=False,
        help='Enable TRT FP8 precision (trt_fp8_enable). Requires TensorRT >= 10 and '
             'Ada Lovelace / Hopper GPU (L40S, H100, RTX 40xx). '
             'Uses the FP16 ONNX + TRT built-in activation-range profiling. '
             '~1.8x GEMM throughput vs FP16.')

    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args = parser.parse_args()

    args.input = args.input.rstrip('/').rstrip('\\')
    os.makedirs(args.output, exist_ok=True)

    if mimetypes.guess_type(args.input)[0] is not None and mimetypes.guess_type(args.input)[0].startswith('video'):
        is_video = True
    else:
        is_video = False

    if is_video and args.input.endswith('.flv'):
        mp4_path = args.input.replace('.flv', '.mp4')
        os.system(f'ffmpeg -i {args.input} -codec copy {mp4_path}')
        args.input = mp4_path

    if args.extract_frame_first and not is_video:
        args.extract_frame_first = False

    run(args)

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        shutil.rmtree(tmp_frames_folder)


if __name__ == '__main__':
    main()
