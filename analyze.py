# -*- coding: utf-8 -*-
"""
Video Analysis Tool  v2
=======================
Full diagnostic scan of an input video.  Measures every quality dimension
that affects the upscale pipeline and recommends exact upscale.py flags.

Measures
--------
  Metadata    : resolution, FPS (avg vs r_frame_rate VFR check), codec, bit-rate,
                pixel format, color space / transfer / range, HDR MaxCLL/MaxFALL,
                sample aspect ratio (anamorphic detection), container, duration,
                audio tracks (codec/bitrate/channels/language), subtitle tracks
  Sharpness   : Laplacian variance (median + StdDev across frames)
  Noise       : high-freq Gaussian residual (median)
  Grain       : dark-area vs bright-area noise ratio (film grain fingerprint)
  Blocking    : DCT 8x8 boundary gradient ratio
  Ringing     : post-sharpening halo detection (near-edge oscillation)
  Shake       : ECC global-motion jitter from consecutive frames
  Faces       : Haar cascade fraction + relative face size
  Letterbox   : median black-bar detection
  Color       : mean luminance, saturation, contrast StdDev,
                black-crush %, white-clip %, color cast (a/b in LAB)
  Animation   : unique-color palette + flat-region ratio
  Interlace   : ffprobe field_order + visual comb-ratio test
  Scene variety: histogram correlation between spaced frames
  Temporal    : frame-to-frame brightness variance (flicker detection)

Output
------
  Full report printed to stdout.
  Pre-processing commands (deinterlace, crop, SAR fix) if needed.
  Estimated per-stage and total pipeline time on RTX 5090.
  Single ready-to-run upscale.py command with all flags.

Usage
-----
    python analyze.py "C:/path/to/movie.mkv"
    python analyze.py "movie.mkv" --samples 40
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
FFPROBE = Path("C:/VideoUpscale/tools/ffmpeg/ffmpeg-8.1.1-full_build/bin/ffprobe.exe")
FFMPEG  = Path("C:/VideoUpscale/tools/ffmpeg/ffmpeg-8.1.1-full_build/bin/ffmpeg.exe")

# RTX 5090 throughput at 1080p (frames/sec) for each stage
_FPS_HAT_SHARPER = 5
_FPS_HAT         = 5
_FPS_ESRGAN      = 18
_FPS_GENERAL     = 30


# ---------------------------------------------------------------------------
# ffprobe
# ---------------------------------------------------------------------------

def ffprobe_json(path):
    r = subprocess.run([
        str(FFPROBE), "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path)
    ], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_spaced(path, n=25):
    """n evenly-spaced frames from the middle 90% (avoids black/credits)."""
    cap   = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = int(total * 0.05)
    end   = int(total * 0.95)
    idxs  = np.linspace(start, end, n, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    return frames


def extract_consecutive(path, n=40):
    """40 consecutive frames from the middle -- needed for shake analysis."""
    cap   = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, total // 2 - n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(n):
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Per-frame quality metrics
# ---------------------------------------------------------------------------

def measure_blur(frame):
    """Laplacian variance: higher = sharper. <100 blurry, >500 sharp."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure_noise(frame):
    """Gaussian residual MAE: proxy for grain/noise."""
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    return float(np.mean(np.abs(gray - blurred)))


def measure_block_artifacts(frame):
    """DCT 8x8 boundary gradient ratio. >1.7 = noticeable blocking."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx   = np.abs(np.diff(gray, axis=1))
    dy   = np.abs(np.diff(gray, axis=0))
    bc = list(range(7, dx.shape[1], 8))
    br = list(range(7, dy.shape[0], 8))
    if not bc or not br:
        return 1.0
    on_h  = float(dx[:, bc].mean())
    on_v  = float(dy[br, :].mean())
    off_h = float(np.delete(dx, bc, axis=1).mean())
    off_v = float(np.delete(dy, br, axis=0).mean())
    return float((on_h / (off_h + 1e-6) + on_v / (off_v + 1e-6)) / 2)


def measure_ringing(frames):
    """
    Detect pre-existing over-sharpening ringing artefacts.
    Measures high-frequency oscillation in pixels just OUTSIDE detected edges
    (ringing halos appear 2-8px away from edges, not on them).
    Returns mean ringing score (higher = more existing ringing).
    """
    scores = []
    for f in frames:
        gray  = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        edges = cv2.Canny(gray.astype(np.uint8), 40, 120)
        k3    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        k7    = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        inner = cv2.dilate(edges, k3)
        outer = cv2.dilate(edges, k7)
        halo  = (outer > 0) & (inner == 0)
        if halo.sum() < 200:
            continue
        lap = np.abs(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_32F))
        scores.append(float(lap[halo].mean()))
    return float(np.median(scores)) if scores else 0.0


def measure_grain_profile(frames):
    """
    Film grain fingerprint: noise in dark vs bright regions.
    Film grain is stronger in shadows; digital/compression noise is uniform.
    Returns dict with dark_noise, bright_noise, ratio (>1.5 = film grain).
    """
    dark_n, bright_n = [], []
    for f in frames:
        gray     = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred  = cv2.GaussianBlur(gray, (7, 7), 0)
        residual = np.abs(gray - blurred)
        dark_mask   = gray < 80
        bright_mask = gray > 180
        if dark_mask.sum() > 500:
            dark_n.append(float(residual[dark_mask].mean()))
        if bright_mask.sum() > 500:
            bright_n.append(float(residual[bright_mask].mean()))
    d = float(np.median(dark_n))   if dark_n   else 0.0
    b = float(np.median(bright_n)) if bright_n else 1.0
    return {"dark_noise": d, "bright_noise": b, "ratio": d / (b + 1e-6)}


# ---------------------------------------------------------------------------
# Color analysis
# ---------------------------------------------------------------------------

def measure_color_stats(frames):
    """
    Returns per-channel stats averaged across all sample frames.
    mean_luminance   : 0-255  (exposure level)
    mean_saturation  : 0-255  (colour richness)
    contrast_std     : StdDev of luminance (0=flat, 60+=punchy)
    black_crush_pct  : % pixels < 10  (crushed blacks)
    white_clip_pct   : % pixels > 245 (blown highlights)
    cast_a, cast_b   : LAB a/b offset from neutral (0=neutral)
                       a>0=red/magenta, a<0=green, b>0=yellow, b<0=blue
    """
    lum, sat, cstd, bc, wc, ca, cb = [], [], [], [], [], [], []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lum.append(float(gray.mean()))
        cstd.append(float(gray.std()))
        bc.append(float((gray < 10).mean() * 100))
        wc.append(float((gray > 245).mean() * 100))
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        sat.append(float(hsv[:, :, 1].mean()))
        lab = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        ca.append(float(lab[:, :, 1].mean()) - 128.0)
        cb.append(float(lab[:, :, 2].mean()) - 128.0)
    return {
        "mean_luminance":  float(np.median(lum)),
        "mean_saturation": float(np.median(sat)),
        "contrast_std":    float(np.median(cstd)),
        "black_crush_pct": float(np.median(bc)),
        "white_clip_pct":  float(np.median(wc)),
        "cast_a": float(np.median(ca)),
        "cast_b": float(np.median(cb)),
    }


# ---------------------------------------------------------------------------
# Content-type detection
# ---------------------------------------------------------------------------

def detect_animation(frames):
    """
    Anime/cartoon detection.
    Animation has: flat colour regions (low local variance) + limited colour palette.
    Returns (score 0-1, median_local_var, median_unique_colours).
    score > 0.52 -> likely animation.
    """
    lv_list, pal_list = [], []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        k    = np.ones((8, 8), np.float32) / 64
        mean = cv2.filter2D(gray, -1, k)
        sq   = cv2.filter2D(gray ** 2, -1, k)
        lv_list.append(float(np.median(sq - mean ** 2)))
        small = cv2.resize(f, (64, 64))
        quant = (small.astype(np.int32) // 32) * 32
        uniq  = len(np.unique(quant.reshape(-1, 3), axis=0))
        pal_list.append(uniq)
    med_lv  = float(np.median(lv_list))
    med_pal = float(np.median(pal_list))
    var_score = max(0.0, 1.0 - med_lv  / 250.0)
    pal_score = max(0.0, 1.0 - med_pal / 140.0)
    return float((var_score + pal_score) / 2), med_lv, med_pal


# ---------------------------------------------------------------------------
# Temporal metrics
# ---------------------------------------------------------------------------

def measure_shake(consecutive_frames):
    """ECC rigid-translation jitter. Returns (jitter_stddev, mean_motion_px)."""
    if len(consecutive_frames) < 4:
        return 0.0, 0.0
    tx_list, ty_list = [], []
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)
    for i in range(len(consecutive_frames) - 1):
        g1 = cv2.resize(cv2.cvtColor(consecutive_frames[i],   cv2.COLOR_BGR2GRAY),
                        (0, 0), fx=0.5, fy=0.5)
        g2 = cv2.resize(cv2.cvtColor(consecutive_frames[i+1], cv2.COLOR_BGR2GRAY),
                        (0, 0), fx=0.5, fy=0.5)
        M  = np.eye(2, 3, dtype=np.float32)
        try:
            _, M = cv2.findTransformECC(g1, g2, M, cv2.MOTION_TRANSLATION,
                                        criteria, None, 5)
            tx_list.append(M[0, 2])
            ty_list.append(M[1, 2])
        except cv2.error:
            continue
    if not tx_list:
        return 0.0, 0.0
    tx, ty = np.array(tx_list), np.array(ty_list)
    ksz    = max(3, len(tx) // 5) | 1
    txs    = cv2.GaussianBlur(tx.reshape(-1, 1), (1, ksz), 0).flatten()
    tys    = cv2.GaussianBlur(ty.reshape(-1, 1), (1, ksz), 0).flatten()
    jitter = float((np.std(tx - txs) + np.std(ty - tys)) / 2)
    motion = float(np.mean(np.sqrt(tx**2 + ty**2)))
    return jitter, motion


def _hist_corr(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    h1 = cv2.calcHist([ga], [0], None, [64], [0, 256])
    h2 = cv2.calcHist([gb], [0], None, [64], [0, 256])
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def measure_scene_variety(spaced_frames):
    """
    Histogram correlation between consecutive spaced frames.
    Returns mean_corr (1.0 = same content everywhere, 0.0 = completely varied).
    Low correlation = many scene changes -- RIFE across cuts will artefact.
    """
    if len(spaced_frames) < 2:
        return 1.0
    corrs = [_hist_corr(a, b) for a, b in zip(spaced_frames, spaced_frames[1:])]
    return float(np.mean(corrs))


def measure_temporal_flicker(spaced_frames):
    """
    Frame-to-frame mean-brightness variance.
    High variance = source has flickering (old film / scan artefacts).
    Returns StdDev of per-frame mean luminance.
    """
    means = [float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean())
             for f in spaced_frames]
    return float(np.std(means))


# ---------------------------------------------------------------------------
# Interlace detection
# ---------------------------------------------------------------------------

def detect_interlace_visual(frames):
    """
    Comb-artefact test: in interlaced content odd/even rows of moving areas
    look different (horizontal combing).
    Returns median comb_ratio (>2.2 strongly suggests interlacing).
    """
    scores = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        even = gray[::2, :]
        odd  = gray[1::2, :]
        mn   = min(even.shape[0], odd.shape[0])
        comb_diff   = float(np.abs(even[:mn] - odd[:mn]).mean())
        within_diff = float(np.abs(np.diff(even[:mn], axis=0)).mean())
        scores.append(comb_diff / (within_diff + 1e-6))
    return float(np.median(scores))


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

def detect_faces(frames):
    """
    Returns (face_fraction, mean_face_area_pct).
    face_fraction      : fraction of sampled frames with >= 1 face
    mean_face_area_pct : average largest-face bounding-box as % of frame area
    """
    xml     = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(xml)
    hits, area_pcts = 0, []
    for f in frames:
        gray  = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) > 0:
            hits += 1
            biggest = max(faces, key=lambda r: r[2] * r[3])
            area_pcts.append(biggest[2] * biggest[3] / (fw * fh) * 100)
    face_frac = hits / len(frames) if frames else 0.0
    mean_area = float(np.median(area_pcts)) if area_pcts else 0.0
    return face_frac, mean_area


# ---------------------------------------------------------------------------
# Letterbox detection
# ---------------------------------------------------------------------------

def detect_letterbox(frames):
    """Returns (top, bottom, left, right) active-pixel bounds (median over samples)."""
    threshold = 16
    results   = []
    for f in frames[:10]:
        gray    = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        h, w    = gray.shape
        row_max = gray.max(axis=1)
        col_max = gray.max(axis=0)
        top    = int(np.argmax(row_max > threshold))
        bottom = int(h - np.argmax(row_max[::-1] > threshold))
        left   = int(np.argmax(col_max > threshold))
        right  = int(w - np.argmax(col_max[::-1] > threshold))
        results.append((top, bottom, left, right))
    return (int(np.median([r[0] for r in results])),
            int(np.median([r[1] for r in results])),
            int(np.median([r[2] for r in results])),
            int(np.median([r[3] for r in results])))


# ---------------------------------------------------------------------------
# Source quality tier detection from filename
# ---------------------------------------------------------------------------

_SOURCE_TIERS = [
    ("TELESYNC",  ["TELESYNC", ".TS.", "-TS-", "TS.x", "TS.H", "TSRIP"]),
    ("CAMRIP",    ["CAMRIP", "CAMINST", "CAM.x", "CAM.H", "HDCAM"]),
    ("DVD",       ["DVDRIP", "DVD.RIP", "DVDSCR", "DVDREMUX"]),
    ("HDTV",      ["HDTV", "PDTV", "DVB"]),
    ("WEB-DL",    ["WEBDL", "WEB-DL", "WEB.DL", "AMZN", "DSNP", "NF.", "HMAX"]),
    ("WEBRip",    ["WEBRIP", "WEB.RIP", "WEBRip"]),
    ("BluRay",    ["BLURAY", "BLU-RAY", "BDRIP", "BDREMUX", "BDMV"]),
]


def detect_source_quality(filename):
    """Return source quality tier string from filename patterns, or None."""
    name = filename.upper()
    for tier, patterns in _SOURCE_TIERS:
        if any(p.upper() in name for p in patterns):
            return tier
    return None


# ---------------------------------------------------------------------------
# Estimated pipeline time (RTX 5090 baselines)
# ---------------------------------------------------------------------------

_ETAs = {
    "denoise":                       150,   # FFmpeg deblock+hqdn3d, all-core CPU
    "stabilize":                      80,   # vidstab 2-pass, all-core CPU
    "Real_HAT_GAN_SRx4_sharper":       5,
    "Real_HAT_GAN_SRx4":               5,
    "RealESRGAN_x4plus":              18,
    "realesr-general-x4v3":           30,
    "RealESRGAN_x2plus":              40,   # 2x model: faster than 4x variants
    "codeformer":                     20,   # at upscaled resolution
    "post":                           80,   # FFmpeg unsharp+eq at 4x resolution
    "rife":                           60,   # RIFE 2x at 4x resolution
}


def estimate_times(total_frames, model_name, rife_exp, fps,
                   do_stabilize=True, do_faces=True, do_rife=True,
                   input_w=1920, input_h=1080):
    """Returns dict of {stage: seconds} and 'total'."""
    scale_factor = (input_w * input_h) / (1920.0 * 1080.0)
    upscale_fps  = _ETAs.get(model_name, 5) / max(scale_factor, 0.5)

    def secs(base_fps):
        return int(total_frames / (base_fps / max(scale_factor, 0.5)))

    t = {}
    t["denoise"]    = secs(_ETAs["denoise"])
    t["stabilize"]  = secs(_ETAs["stabilize"]) if do_stabilize else 0
    t["upscale"]    = int(total_frames / max(upscale_fps, 0.01))
    t["codeformer"] = secs(_ETAs["codeformer"]) if do_faces else 0
    t["post"]       = secs(_ETAs["post"])
    t["rife"]       = secs(_ETAs["rife"]) * rife_exp if do_rife else 0
    t["encode"]     = 600   # NVENC multipass: ~10 min flat
    t["total"]      = sum(t.values())
    return t


def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze(video_path, n_samples=25):
    path = Path(video_path)
    if not path.exists():
        sys.exit(f"ERROR: File not found: {path}")

    SEP  = "-" * 66
    SEP2 = "=" * 66

    print(f"\n{SEP2}")
    print(f"  VIDEO ANALYSIS   {path.name}")
    print(SEP2)

    # -----------------------------------------------------------------------
    # [1/6] Metadata
    # -----------------------------------------------------------------------
    print("  [1/6] Reading metadata ...", end=" ", flush=True)
    source_quality = detect_source_quality(path.name)
    probe = ffprobe_json(path)
    vs = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    if vs is None:
        sys.exit("ERROR: No video stream found.")

    audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    sub_streams   = [s for s in probe["streams"] if s["codec_type"] == "subtitle"]
    fmt           = probe["format"]

    width  = int(vs["width"])
    height = int(vs["height"])

    def parse_rate(s):
        n, d = s.split("/")
        return round(int(n) / max(int(d), 1), 3)

    r_fps   = parse_rate(vs.get("r_frame_rate",  "24/1"))
    avg_fps = parse_rate(vs.get("avg_frame_rate", "24/1"))
    fps     = avg_fps
    is_vfr  = abs(r_fps - avg_fps) > 0.5 and avg_fps > 0

    codec      = vs.get("codec_name", "unknown").upper()
    pix_fmt    = vs.get("pix_fmt", "unknown")
    profile    = vs.get("profile", "")
    bit_depth  = int(vs.get("bits_per_raw_sample", "8") or "8")

    color_space     = vs.get("color_space",     "")
    color_transfer  = vs.get("color_transfer",  "")
    color_primaries = vs.get("color_primaries", "")
    color_range     = vs.get("color_range",     "")   # "tv" or "pc"

    is_hdr = ("smpte2084"    in color_transfer
           or "arib-std-b67" in color_transfer
           or "bt2020"       in (color_space + color_primaries).lower())

    # HDR MaxCLL / MaxFALL from side_data
    max_cll = max_fall = None
    for sd in vs.get("side_data_list", []):
        sdt = sd.get("side_data_type", "").lower()
        if "content light" in sdt:
            max_cll  = sd.get("max_content", sd.get("MaxCLL"))
            max_fall = sd.get("max_average", sd.get("MaxFALL"))

    # Sample Aspect Ratio (anamorphic detection)
    sar_str = (vs.get("sample_aspect_ratio", "1:1") or "1:1").replace("/", ":")
    dar_str = (vs.get("display_aspect_ratio", "") or "").replace("/", ":")
    try:
        sar_n, sar_d = (int(x) for x in sar_str.split(":")[:2])
    except Exception:
        sar_n, sar_d = 1, 1
    is_anamorphic = (sar_n != sar_d) and sar_n > 0 and sar_d > 0

    # Bitrate / bpp / duration
    total_bitrate = int(fmt.get("bit_rate", 0))
    video_bitrate = int(vs.get("bit_rate", 0)) or int(total_bitrate * 0.85)
    duration_s    = float(fmt.get("duration", 0))
    bpp           = video_bitrate / max(width * height * fps, 1)
    hh = int(duration_s // 3600)
    mm = int((duration_s % 3600) // 60)
    ss = int(duration_s % 60)
    dur_str = f"{hh:02d}:{mm:02d}:{ss:02d}"
    container = Path(path).suffix.lstrip(".").upper()

    # Field order
    field_order           = vs.get("field_order", "progressive") or "progressive"
    is_interlaced_ffprobe = field_order not in ("progressive", "unknown", "")
    if "interlaced" in profile.lower():
        is_interlaced_ffprobe = True

    # Total frames
    nb_str = vs.get("nb_frames", "")
    total_frames = int(nb_str) if (nb_str and nb_str.isdigit()) else int(duration_s * fps)

    # Audio info
    audio_info = []
    for i, a in enumerate(audio_streams):
        tags    = a.get("tags", {})
        lang    = tags.get("language", tags.get("LANGUAGE", "?"))
        disp    = a.get("disposition", {})
        flags   = [k for k in ("default", "forced", "commentary") if disp.get(k)]
        acodec  = a.get("codec_name", "?").upper()
        abr     = int(a.get("bit_rate", 0))
        achan   = a.get("channels", "?")
        asrate  = int(a.get("sample_rate", 0))
        alayout = a.get("channel_layout", "")
        audio_info.append({
            "index": i+1, "lang": lang, "codec": acodec,
            "bitrate_kbps": abr // 1000 if abr else 0,
            "channels": achan, "layout": alayout,
            "sample_rate_hz": asrate, "flags": flags,
        })

    # Subtitle info
    sub_info = []
    for s in sub_streams:
        tags   = s.get("tags", {})
        lang   = tags.get("language", tags.get("LANGUAGE", "?"))
        sfmt   = s.get("codec_name", "?").upper()
        forced = bool(s.get("disposition", {}).get("forced", 0))
        sub_info.append({"lang": lang, "format": sfmt, "forced": forced})

    print("done")

    # -----------------------------------------------------------------------
    # [2/6] Extract frames
    # -----------------------------------------------------------------------
    print(f"  [2/6] Extracting {n_samples} spaced frames ...", end=" ", flush=True)
    spaced = extract_spaced(path, n=n_samples)
    print(f"done ({len(spaced)})")
    if not spaced:
        sys.exit("ERROR: Could not decode any frames. Check the file.")

    print("  [3/6] Extracting 40 consecutive frames ...", end=" ", flush=True)
    consec = extract_consecutive(path, n=40)
    print(f"done ({len(consec)})")

    # -----------------------------------------------------------------------
    # [4/6] Per-frame quality metrics
    # -----------------------------------------------------------------------
    print("  [4/6] Measuring frame quality ...", end=" ", flush=True)
    blur_scores  = [measure_blur(f)            for f in spaced]
    noise_scores = [measure_noise(f)           for f in spaced]
    block_scores = [measure_block_artifacts(f) for f in spaced]
    ring_score   = measure_ringing(spaced)
    grain        = measure_grain_profile(spaced)

    med_blur  = float(np.median(blur_scores))
    std_blur  = float(np.std(blur_scores))
    min_blur  = float(np.min(blur_scores))
    p10_blur  = float(np.percentile(blur_scores, 10))
    med_noise = float(np.median(noise_scores))
    med_block = float(np.median(block_scores))
    print("done")

    # -----------------------------------------------------------------------
    # [5/6] Temporal + scene + colour + interlace + animation
    # -----------------------------------------------------------------------
    print("  [5/6] Temporal, colour, interlace, animation ...", end=" ", flush=True)
    jitter, mean_motion = measure_shake(consec)
    scene_corr          = measure_scene_variety(spaced)
    flicker             = measure_temporal_flicker(spaced)
    color               = measure_color_stats(spaced)
    anim_score, lv, pal = detect_animation(spaced)
    comb_ratio          = detect_interlace_visual(spaced)
    print("done")

    # -----------------------------------------------------------------------
    # [6/6] Face + letterbox
    # -----------------------------------------------------------------------
    print("  [6/6] Faces, letterbox ...", end=" ", flush=True)
    face_frac, face_area = detect_faces(spaced[::3])
    top, bottom, left, right = detect_letterbox(spaced)
    has_letterbox = (top  > height * 0.03
                  or (height - bottom) > height * 0.03
                  or left > width  * 0.03
                  or (width  - right)  > width  * 0.03)
    print("done")

    # -----------------------------------------------------------------------
    # Labels
    # -----------------------------------------------------------------------
    def blur_lbl(v):
        if v < 80:   return "very soft/blurry"
        if v < 300:  return "moderate sharpness"
        if v < 800:  return "sharp"
        return "very sharp"

    def noise_lbl(v):
        if v < 3.5:  return "clean"
        if v < 6.0:  return "light grain/noise"
        if v < 10.0: return "moderate noise"
        return "heavy noise/grain"

    def block_lbl(v):
        if v < 1.3:  return "minimal"
        if v < 1.7:  return "light"
        if v < 2.2:  return "moderate"
        return "heavy compression artefacts"

    def shake_lbl(v):
        if v < 0.3:  return "stable"
        if v < 0.8:  return "slight jitter"
        if v < 1.5:  return "moderate shake"
        return "significant handheld shake"

    def bpp_lbl(v):
        if v > 0.15: return "HIGH quality"
        if v > 0.06: return "MEDIUM quality"
        if v > 0.02: return "LOW -- compressed"
        return "VERY LOW -- heavily compressed"

    is_interlaced = is_interlaced_ffprobe or (comb_ratio > 2.2)

    # Animation false-positive guard: very blurry images (blur < 80) cannot be
    # animation -- animation always has crisp hand-drawn edges.  Dark/flat
    # live-action (e.g. TELESYNC) triggers the palette/variance test falsely.
    if med_blur < 80:
        anim_score *= max(0.0, med_blur / 80.0)
    is_animation  = anim_score > 0.52

    # Estimate cuts per minute from histogram drops across spaced frames
    spacing_sec  = duration_s / max(n_samples - 1, 1)
    scene_cuts   = sum(1 for a, b in zip(spaced, spaced[1:])
                       if _hist_corr(a, b) < 0.5)
    cuts_per_min = scene_cuts / (duration_s / 60.0) if duration_s > 0 else 0.0

    # -----------------------------------------------------------------------
    # Build recommendations
    # -----------------------------------------------------------------------

    # -- Denoise
    if grain["ratio"] > 1.8:
        rec_denoise  = "low"
        denoise_note = ("film-grain pattern (noise stronger in shadows) -- "
                        "low pass preserves organic grain texture")
    elif med_noise < 3.5 and med_block < 1.3:
        rec_denoise  = "low"
        denoise_note = "very clean source -- light pass only"
    else:
        rec_denoise  = "medium"
        denoise_note = "noise/compression artefacts -- standard clean"

    # -- Model + Scale (default 2x -> 4K; animation/extreme-recovery cases use 4x)
    if is_animation:
        rec_scale  = 4
        rec_model  = "RealESRGAN_x4plus"
        model_note = ("animation detected (flat regions, limited palette) -- "
                      "ESRGAN 4x avoids over-sharpening clean lines")
    elif med_blur > 500 and med_noise < 4.0:
        rec_scale  = 2
        rec_model  = "RealESRGAN_x2plus"
        model_note = "already-sharp clean source -- 2x native model to 4K, fast"
    else:
        rec_scale  = 2
        rec_model  = "RealESRGAN_x2plus"
        model_note = "standard 2x upscale to 4K"

    # -- TELESYNC / CAMRIP source overrides
    source_warn = None
    is_camlike = source_quality in ("TELESYNC", "CAMRIP")
    if is_camlike:
        source_warn = (
            f"{source_quality} SOURCE DETECTED -- camera-recorded from cinema projection.\n"
            "    Characteristics: projector-beat flicker, focus drift, warm colour cast,\n"
            "    dark exposure, possible audience/ambient audio on all tracks.\n"
            "    Recommendation: HAT_sharper model (not ESRGAN), deflicker pre-processing,\n"
            "    and consider --no-stabilize (tripod footage).")
        # Force 4x + sharpest model regardless of blur/animation scores
        rec_scale  = 4
        rec_model  = "Real_HAT_GAN_SRx4_sharper"
        model_note = (f"{source_quality} source -- extremely blurry; 4x HAT_sharper recovers "
                      "maximum detail from defocused cinema frames")
        # Force medium denoise to handle projector-screen grain
        rec_denoise  = "medium"
        denoise_note = (f"{source_quality} source -- projector-screen grain/noise; "
                        "medium pass cleans without over-smoothing")

    # -- Ringing override
    ring_warn = None
    if ring_score > 18:
        ring_warn = (f"EXISTING RINGING/HALOS (score {ring_score:.1f}) -- "
                     "source already over-sharpened; switching to 2x model, --denoise low")
        rec_scale   = 2
        rec_model   = "RealESRGAN_x2plus"
        model_note  = "ringing detected -- 2x native model avoids amplifying edge halos"
        rec_denoise = "low"

    # -- Stabilize
    if jitter < 0.2:
        rec_stabilize = False
        stab_note = ("footage is tripod-stable -- RECOMMEND --no-stabilize "
                     "(saves ~" + fmt_time(int(total_frames/80)) + " and has no benefit)")
    elif jitter < 0.8:
        rec_stabilize = True
        stab_note = "slight jitter -- stabilize will help"
    else:
        rec_stabilize = True
        stab_note = "clear shake -- stabilize strongly recommended"

    # -- Deinterlace
    rec_deinterlace = is_interlaced
    di_note = ("INTERLACED -- use --deinterlace; yadif runs before denoise"
               if is_interlaced else None)

    # -- Face restore
    if face_frac < 0.04:
        rec_face_restore = False
        rec_fidelity     = 0.7
        fidelity_note    = "no faces detected -- skip CodeFormer to save time"
    elif face_area > 12.0:
        rec_face_restore = True
        rec_fidelity     = 0.8
        fidelity_note    = (f"large closeup faces (~{face_area:.0f}% of frame) -- "
                            "0.8 fidelity preserves natural skin texture")
    elif med_blur < 150 or med_noise > 7.0:
        rec_face_restore = True
        rec_fidelity     = 0.5
        fidelity_note    = (f"degraded/soft faces ({face_frac*100:.0f}% frames) -- "
                            "0.5 fidelity = more correction")
    else:
        rec_face_restore = True
        rec_fidelity     = 0.7
        fidelity_note    = f"faces in {face_frac*100:.0f}% frames -- balanced fidelity"

    # -- RIFE
    rec_no_rife  = fps >= 47.9
    rec_rife_exp = 1
    if rec_no_rife:
        rife_note = f"already {fps}fps -- RIFE doubles to {fps*2:.0f}fps; use --no-rife to skip"
    elif cuts_per_min > 30:
        rife_note = (f"high cut rate (~{cuts_per_min:.0f} cuts/min) -- "
                     "2x safe; avoid --rife-exp 2 (ghosting at cuts)")
    elif fps <= 15:
        rife_note = f"very low FPS ({fps}) -- consider --rife-exp 2 for 4x if motion permits"
    else:
        rife_note = f"{fps}fps -> {fps*2:.0f}fps"

    # -- Codec
    rec_codec  = "av1" if is_hdr else "hevc"
    codec_note = ("AV1 NVENC preserves HDR colour volume" if is_hdr
                  else "HEVC NVENC CQ14 -- standard for SDR")

    # -- Scale advice
    out_w, out_h = width * rec_scale, height * rec_scale
    scale_warn = None
    if rec_scale == 4 and width >= 1920:
        scale_warn = (f"4x from {width}x{height} = {out_w}x{out_h} (8K). "
                      "Add --scale 2 for 4K output instead.")
    elif rec_scale == 2 and width <= 720:
        scale_warn = (f"SD source ({width}x{height}): 2x = {out_w}x{out_h}. "
                      "Add --scale 4 for HD/4K if source quality permits.")

    # -- Color warnings
    lum = color["mean_luminance"]
    sat = color["mean_saturation"]
    bc  = color["black_crush_pct"]
    wc  = color["white_clip_pct"]
    ca  = color["cast_a"]
    cb  = color["cast_b"]
    cst = color["contrast_std"]

    color_warns = []
    if lum < 70:
        color_warns.append(f"DARK source (mean lum {lum:.0f}/255) -- eq gamma=0.95 will brighten")
    if lum > 200:
        color_warns.append(f"BRIGHT source (mean lum {lum:.0f}/255) -- eq may over-expose")
    if bc > 5.0:
        color_warns.append(f"BLACK CRUSH: {bc:.1f}% pixels crushed -- shadow detail already lost")
    if wc > 3.0:
        color_warns.append(f"WHITE CLIP: {wc:.1f}% pixels blown -- highlight detail lost")
    if sat > 140:
        color_warns.append(f"OVER-SATURATED (sat {sat:.0f}/255) -- eq adds 1.15x, may over-saturate")
    if cst < 25:
        color_warns.append(f"FLAT / LOW CONTRAST (sigma={cst:.0f}) -- eq contrast=1.08 will help")
    if abs(ca) > 8:
        color_warns.append(f"COLOR CAST: {'red/magenta' if ca > 0 else 'green'} (LAB a={ca:+.1f})")
    if abs(cb) > 8:
        color_warns.append(f"COLOR CAST: {'yellow/warm' if cb > 0 else 'blue/cool'} (LAB b={cb:+.1f})")
    if color_range == "pc":
        color_warns.append("FULL RANGE (0-255) source -- pipeline assumes limited range; minor clip possible")

    # -- STAR prompt suggestion
    if is_camlike:
        star_prompt = "A dark cinematic action film with natural textures and realistic detail"
    elif is_animation:
        star_prompt = "A vibrant anime animation with clean sharp lines and vivid colors"
    elif lum < 80:
        star_prompt = "A dark cinematic film with detailed shadows and atmospheric textures"
    else:
        star_prompt = "A cinematic live-action film with detailed textures"

    # -- Estimated pipeline times
    t = estimate_times(
        total_frames = total_frames,
        model_name   = rec_model,
        rife_exp     = 0 if rec_no_rife else rec_rife_exp,
        fps          = fps,
        do_stabilize = rec_stabilize,
        do_faces     = rec_face_restore,
        do_rife      = not rec_no_rife,
        input_w      = width,
        input_h      = height,
    )

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    print(f"\n{SEP2}")
    print(f"  ANALYSIS REPORT   {path.name}")
    print(SEP2)

    # -- Source metadata
    print(f"\n  SOURCE METADATA")
    print(f"  {SEP}")
    if source_quality:
        tier_note = {"TELESYNC": " !! camera-recorded cinema (worst quality tier)",
                     "CAMRIP":   " !! camera-recorded cinema (worst quality tier)",
                     "DVD":      " (standard DVD rip)",
                     "HDTV":     " (broadcast capture)",
                     "WEB-DL":   " (streaming download -- good quality)",
                     "WEBRip":   " (streaming re-encode)",
                     "BluRay":   " (disc source -- best quality)",
                     }.get(source_quality, "")
        print(f"    Source quality : {source_quality}{tier_note}")
    print(f"    Container      : {container}")
    print(f"    Resolution     : {width}x{height}  ->  {out_w}x{out_h}  ({rec_scale}x upscale)")
    if dar_str:
        anam = "  ** ANAMORPHIC -- non-square pixels **" if is_anamorphic else ""
        print(f"    Aspect ratio   : {dar_str}  (SAR {sar_str}){anam}")
    vfr_flag = f"  ** VFR (r={r_fps} avg={avg_fps:.3f}) **" if is_vfr else ""
    print(f"    FPS            : {fps}{vfr_flag}")
    intf_flag = f"  ** {field_order} **" if is_interlaced_ffprobe else ""
    print(f"    Field order    : {field_order}{intf_flag}")
    print(f"    Codec          : {codec}  {profile}")
    print(f"    Pixel format   : {pix_fmt}  ({bit_depth}-bit  "
          + ("HDR" if is_hdr else "SDR") + ")")
    if is_hdr:
        hdr_str = color_transfer
        if max_cll  is not None: hdr_str += f"  MaxCLL={max_cll}"
        if max_fall is not None: hdr_str += f"  MaxFALL={max_fall}"
        print(f"    HDR            : {hdr_str}")
    print(f"    Color space    : {color_space or 'unknown'}  |  primaries: "
          f"{color_primaries or 'unknown'}  |  range: {color_range or 'unknown'}")
    bps_mb = video_bitrate / 1_000_000
    print(f"    Video bitrate  : {bps_mb:.1f} Mbps  ({bpp:.4f} bpp  {bpp_lbl(bpp)})")
    print(f"    Duration       : {dur_str}  ({total_frames:,} frames)")

    # -- Audio
    print(f"\n  AUDIO TRACKS  ({len(audio_streams)})")
    print(f"  {SEP}")
    if audio_info:
        for a in audio_info:
            flags_str = "  [" + ", ".join(a["flags"]) + "]" if a["flags"] else ""
            br_str    = f"{a['bitrate_kbps']}kbps" if a["bitrate_kbps"] else "?kbps"
            layout    = a["layout"] or f"{a['channels']}ch"
            print(f"    Track {a['index']}: {a['codec']:<8} {br_str:<10} "
                  f"{layout:<14} {a['sample_rate_hz']//1000}kHz  [{a['lang']}]{flags_str}")
    else:
        print("    (none)")

    if sub_info:
        print(f"\n  SUBTITLE TRACKS  ({len(sub_info)})")
        print(f"  {SEP}")
        for i, s in enumerate(sub_info):
            forced = "  [forced]" if s["forced"] else ""
            print(f"    Track {i+1}: {s['format']:<8} [{s['lang']}]{forced}")

    # -- Frame quality
    print(f"\n  FRAME QUALITY  ({len(spaced)} samples)")
    print(f"  {SEP}")
    print(f"    Sharpness (median)   : {med_blur:>8.1f}    {blur_lbl(med_blur)}")
    print(f"    Sharpness (StdDev)   : {std_blur:>8.1f}    "
          + ("consistent" if std_blur < med_blur * 0.6
             else "HIGH VARIANCE -- inconsistent focus or many scene types"))
    print(f"    Sharpness (p10/min)  : {p10_blur:>8.1f} / {min_blur:.1f}   "
          + ("worst frames ok" if p10_blur > 80 else "worst frames very blurry"))
    print(f"    Noise/grain (median) : {med_noise:>8.2f}    {noise_lbl(med_noise)}")
    print(f"    Grain profile        : dark={grain['dark_noise']:.2f}  "
          f"bright={grain['bright_noise']:.2f}  ratio={grain['ratio']:.2f}  "
          + ("-> film grain" if grain["ratio"] > 1.8 else "-> digital/compression noise"))
    print(f"    Block artefacts      : {med_block:>8.3f}    {block_lbl(med_block)}")
    print(f"    Ringing/halos        : {ring_score:>8.1f}    "
          + ("** pre-existing over-sharpening **" if ring_score > 18 else "minimal"))
    if ring_warn:
        print(f"    !! {ring_warn}")
    print(f"    Interlace (visual)   : comb_ratio={comb_ratio:.2f}  "
          + ("** INTERLACED **" if is_interlaced else "progressive"))

    # -- Colour
    print(f"\n  COLOUR ANALYSIS")
    print(f"  {SEP}")
    print(f"    Mean luminance   : {lum:>6.1f} / 255  "
          + ("dark" if lum < 80 else "bright" if lum > 180 else "normal"))
    print(f"    Mean saturation  : {sat:>6.1f} / 255  "
          + ("muted" if sat < 50 else "vivid" if sat > 120 else "normal"))
    print(f"    Contrast (sigma) : {cst:>6.1f}        "
          + ("flat" if cst < 25 else "punchy" if cst > 60 else "normal"))
    print(f"    Black crush      : {bc:>6.1f}%        "
          + ("ok" if bc < 2 else "WARNING -- shadow detail lost"))
    print(f"    White clip       : {wc:>6.1f}%        "
          + ("ok" if wc < 1 else "WARNING -- highlight detail lost"))
    print(f"    Color cast       : a={ca:+.1f}  b={cb:+.1f}  "
          + ("neutral" if abs(ca) < 5 and abs(cb) < 5 else "CAST DETECTED"))
    for w in color_warns:
        print(f"    !! {w}")

    # -- Temporal
    print(f"\n  TEMPORAL ANALYSIS")
    print(f"  {SEP}")
    print(f"    Camera jitter    : {jitter:.3f}px  {shake_lbl(jitter)}"
          f"  (mean motion {mean_motion:.1f}px/frame)")
    print(f"    Scene variety    : corr={scene_corr:.3f}  ~{cuts_per_min:.0f} cuts/min  "
          + ("action/fast-cut" if cuts_per_min > 30 else
             "varied"          if cuts_per_min > 10 else "slow-paced"))
    if flicker > 12 and is_camlike:
        flicker_desc = "** DETECTED -- projector-vs-camera beat frequency **"
    elif flicker > 12:
        flicker_desc = "** DETECTED -- possible old film scan **"
    else:
        flicker_desc = "none"
    print(f"    Temporal flicker : {flicker:.2f}  {flicker_desc}")

    # -- Content type
    print(f"\n  CONTENT TYPE")
    print(f"  {SEP}")
    print(f"    Animation score  : {anim_score:.2f}  "
          + ("** ANIMATION / CARTOON **" if is_animation else "live-action"))
    print(f"      (local variance={lv:.0f}  unique colours={pal:.0f})")
    face_size_note = ""
    if face_area > 0:
        if face_area > 12:
            face_size_note = "  (closeup / portrait)"
        elif face_area < 2:
            face_size_note = "  (small faces / crowd)"
    print(f"    Faces detected   : {face_frac*100:.0f}% of frames"
          + (f"  largest ~{face_area:.1f}% of frame{face_size_note}" if face_area > 0 else "  (none)"))
    if has_letterbox:
        crop_w = right - left
        crop_h = bottom - top
        print(f"    Letterbox        : YES -- active area {crop_w}x{crop_h}  "
              f"(bars: top={top} bottom={height-bottom} left={left} right={width-right}px)")
    else:
        print("    Letterbox        : none")

    # -- Recommendations
    print(f"\n  RECOMMENDATIONS")
    print(f"  {SEP}")
    if source_warn:
        for line in source_warn.split("\n"):
            print(f"    !! {line}")
        print()
    if rec_deinterlace:
        print(f"    --deinterlace                        {di_note}")
    print(f"    --denoise {rec_denoise:<8}                  {denoise_note}")
    if rec_stabilize:
        print(f"    stabilize (default ON)               {stab_note}")
    else:
        print(f"    --no-stabilize                       {stab_note}")
    print(f"    --model {rec_model}")
    print(f"      -> {model_note}")
    if rec_scale != 2:
        print(f"    --scale {rec_scale}")
    if rec_face_restore:
        print(f"    --face-fidelity {rec_fidelity:<5}               {fidelity_note}")
    else:
        print(f"    --no-face-restore                    {fidelity_note}")
    rife_flag = "--no-rife" if rec_no_rife else f"--rife-exp {rec_rife_exp}"
    print(f"    {rife_flag:<37}  {rife_note}")
    print(f"    --codec {rec_codec:<8}                   {codec_note}")
    if scale_warn:
        print(f"    SCALE NOTE: {scale_warn}")
    if is_vfr:
        print(f"    VFR WARNING: r_fps={r_fps} avg_fps={avg_fps:.3f} -- "
              "convert to CFR before upscaling (see pre-processing below)")

    # -- Estimated times
    print(f"\n  ESTIMATED PIPELINE TIME  (RTX 5090 baseline, {width}x{height} input)")
    print(f"  {SEP}")
    print(f"    Stage 1  Denoise         : {fmt_time(t['denoise'])}")
    if rec_stabilize:
        print(f"    Stage 2  Stabilize       : {fmt_time(t['stabilize'])}")
    print(f"    Stage 3  Upscale ({rec_model}) : {fmt_time(t['upscale'])}")
    if rec_face_restore:
        print(f"    Stage 4  CodeFormer      : {fmt_time(t['codeformer'])}")
    print(f"    Stage 5  Post/sharpen    : {fmt_time(t['post'])}")
    if not rec_no_rife:
        rife_mult = 2 ** rec_rife_exp
        print(f"    Stage 6  RIFE {rife_mult}x          : {fmt_time(t['rife'])}")
    print(f"    Stage 7  Encode NVENC    : {fmt_time(t['encode'])}")
    print(f"    {'- '*22}")
    print(f"    TOTAL                    : {fmt_time(t['total'])}")

    # -- Pre-processing commands
    pre_cmds = []
    if is_interlaced:
        pre_cmds.append(
            "NOTE: --deinterlace flag runs yadif inside the pipeline automatically.")
    if is_anamorphic:
        display_w = int(width * sar_n / sar_d)
        pre_cmds.append(
            f"# Fix non-square pixels (SAR {sar_str}) -- pre-scale before upscaling:\n"
            f"  {FFMPEG} -i \"{path}\" -vf scale={display_w}:{height} -c:a copy rescaled.mkv")
    if is_vfr:
        pre_cmds.append(
            f"# Convert VFR -> CFR ({avg_fps:.3f}fps) for RIFE/stabilize compatibility:\n"
            f"  {FFMPEG} -i \"{path}\" -vsync cfr -r {avg_fps:.3f} -c:a copy cfr.mkv")
    if flicker > 12:
        flicker_label = "projector-beat flicker" if is_camlike else "old film scan flicker"
        pre_cmds.append(
            f"# Deflicker {flicker_label} (flicker={flicker:.1f}):\n"
            f"  {FFMPEG} -i \"{path}\" -vf deflicker -c:a copy deflickered.mkv")
    if has_letterbox:
        crop_w = right - left
        crop_h = bottom - top
        pre_cmds.append(
            f"# Crop letterbox (active area {crop_w}x{crop_h}):\n"
            f"  {FFMPEG} -i \"{path}\" -vf \"crop={crop_w}:{crop_h}:{left}:{top}\" "
            f"-c:a copy cropped.mkv")

    if pre_cmds:
        print(f"\n  PRE-PROCESSING COMMANDS")
        print(f"  {SEP}")
        for cmd in pre_cmds:
            print(f"  {cmd}")

    # -- STAR prompt
    print(f"\n  STAR PROMPT  (only needed with --generative-video)")
    print(f"  {SEP}")
    print(f'    Suggested: --star-prompt "{star_prompt}"')

    # -- Suggested command
    cmd_parts = [f'python upscale.py --input "{path}"']
    if rec_deinterlace:
        cmd_parts.append("--deinterlace")
    if not rec_stabilize:
        cmd_parts.append("--no-stabilize")
    if rec_denoise != "medium":
        cmd_parts.append(f"--denoise {rec_denoise}")
    if rec_model != "RealESRGAN_x2plus":
        cmd_parts.append(f"--model {rec_model}")
    if rec_scale != 2:
        cmd_parts.append(f"--scale {rec_scale}")
    if rec_face_restore:
        if rec_fidelity != 0.7:
            cmd_parts.append(f"--face-fidelity {rec_fidelity}")
    else:
        cmd_parts.append("--no-face-restore")
    if rec_no_rife:
        cmd_parts.append("--no-rife")
    elif rec_rife_exp != 1:
        cmd_parts.append(f"--rife-exp {rec_rife_exp}")
    if rec_codec != "hevc":
        cmd_parts.append(f"--codec {rec_codec}")

    indent = " \\\n      "
    print(f"\n  SUGGESTED COMMAND")
    print(f"  {SEP}")
    print("  " + indent.join(cmd_parts))
    print(f"  {SEP}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Analyse a video and recommend optimal upscale.py settings.")
    ap.add_argument("input",     help="Path to input video file")
    ap.add_argument("--samples", type=int, default=25,
                    help="Number of evenly-spaced frames to sample (default: 25)")
    args = ap.parse_args()

    try:
        import cv2
        import numpy
    except ImportError:
        sys.exit("ERROR: opencv-python-headless and numpy are required.\n"
                 "Run: pip install opencv-python-headless numpy")

    analyze(args.input, n_samples=args.samples)