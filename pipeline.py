#!/usr/bin/env python3
"""
Adaptive Artwork Damage Analysis and Interpolation Pipeline
===========================================================

Goal
----
Detect what kind of degradation is present at each pixel/region and apply
an appropriate interpolation / restoration transition strategy:

- photometric change      -> linear / color-domain blending
- geometric displacement  -> optical-flow-guided warping
- texture mismatch        -> Laplacian pyramid blending
- missing material / hole -> inpainting + substrate-aware blending

This script is designed to be more robust and easier to defend in a project
submission than an over-specialized simulation-only pipeline.

Key design choices
------------------
1. Deterministic outputs: all stochastic components use a fixed seed.
2. Monotonic progression: masks grow with time, so intermediate frames do not
   stagnate or repeat.
3. Exact endpoint: the final frame is the true damaged image.
4. Per-pixel routing: the method is chosen by issue maps, not by one global
   interpolation mode.
5. Diagnostics: saves signal maps, issue maps, routing map, confidence maps,
   contact sheets, and optional evaluation.

Usage
-----
python artwork_restoration_project.py \
    --original original.jpg \
    --damaged damaged.jpg \
    --num_frames 7 \
    --output_dir results

Optional:
    --ground_truth midpoint.jpg
    --no_align
    --seed 7
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import glob
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_image(path: str | Path, img: np.ndarray) -> None:
    ensure_dir(Path(path).parent)
    cv2.imwrite(str(path), img)


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_to_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if src.shape[:2] == ref.shape[:2]:
        return src
    return cv2.resize(src, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)


def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def colorize_map(x: np.ndarray, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    return cv2.applyColorMap((norm01(x) * 255).astype(np.uint8), cmap)


def overlay(base: np.ndarray, mask_or_color: np.ndarray, a: float = 0.55, b: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(base, a, mask_or_color, b, 0)


def grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def section(title: str) -> None:
    bar = "─" * 72
    print(f"\n{bar}\n{title}\n{bar}")


# ---------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------

def foreground_mask(img: np.ndarray, margin: float = 0.05) -> np.ndarray:
    h, w = img.shape[:2]
    m = int(min(h, w) * margin)
    rect = (m, m, max(1, w - 2 * m), max(1, h - 2 * m))
    gc = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        fg = np.full((h, w), 255, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)


def warp_h(src: np.ndarray, H: np.ndarray, ref_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = ref_shape[:2]
    return cv2.warpPerspective(
        src,
        H,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


def align_feature_based(src: np.ndarray, ref: np.ndarray, use_sift: bool = True) -> np.ndarray | None:
    ms = foreground_mask(src)
    mr = foreground_mask(ref)

    detector = None
    norm = None
    if use_sift:
        try:
            detector = cv2.SIFT_create(nfeatures=5000)
            norm = "flann"
        except Exception:
            detector = None
    if detector is None:
        detector = cv2.ORB_create(nfeatures=5000)
        norm = "orb"

    ks, ds = detector.detectAndCompute(src, ms)
    kr, dr = detector.detectAndCompute(ref, mr)
    if ds is None or dr is None or len(ks) < 8 or len(kr) < 8:
        return None

    if norm == "flann":
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = matcher.knnMatch(ds, dr, k=2)
    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return None

    ps = np.float32([ks[m.queryIdx].pt for m in good])
    pr = np.float32([kr[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(ps, pr, cv2.RANSAC, 5.0)
    if H is None:
        return None
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"  feature alignment: {len(good)} matches, {inliers} inliers")
    return warp_h(src, H, ref.shape)


def align_ecc(src: np.ndarray, ref: np.ndarray) -> np.ndarray | None:
    gs = grayscale(src).astype(np.float32) / 255.0
    gr = grayscale(ref).astype(np.float32) / 255.0
    H = np.eye(3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6)
    try:
        cv2.findTransformECC(gr, gs, H, cv2.MOTION_HOMOGRAPHY, crit)
        print("  ECC alignment succeeded")
        return warp_h(src, H, ref.shape)
    except cv2.error:
        return None


def align_images(src: np.ndarray, ref: np.ndarray, debug_path: str | None = None) -> np.ndarray:
    out = align_feature_based(src, ref, use_sift=True)
    if out is None:
        out = align_feature_based(src, ref, use_sift=False)
    if out is None:
        out = align_ecc(src, ref)
    if out is None:
        raise RuntimeError("Alignment failed for SIFT/ORB/ECC. Use --no_align if already registered.")
    if debug_path is not None:
        save_image(debug_path, overlay(ref, out, 0.5, 0.5))
    return out


# ---------------------------------------------------------------------
# Signal extraction and issue analysis
# ---------------------------------------------------------------------

@dataclass
class Signals:
    delta_e: np.ndarray
    grad_diff: np.ndarray
    entropy_drop: np.ndarray
    chroma_loss: np.ndarray
    hue_shift: np.ndarray
    value_drop: np.ndarray
    local_var_drop: np.ndarray
    motion_mag: np.ndarray
    flow_conf: np.ndarray
    edge_break: np.ndarray


@dataclass
class IssueMaps:
    photometric: np.ndarray
    geometric: np.ndarray
    texture: np.ndarray
    missing: np.ndarray
    clean: np.ndarray
    severity: np.ndarray


@dataclass
class RoutingWeights:
    linear: np.ndarray
    flow: np.ndarray
    laplacian: np.ndarray
    inpaint: np.ndarray



def local_entropy_fallback(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray.astype(np.float32), (11, 11), 0)
    dev = gray.astype(np.float32) - blur
    return np.sqrt(cv2.GaussianBlur(dev * dev, (11, 11), 0))



def compute_signals(orig: np.ndarray, damaged: np.ndarray) -> Signals:
    lab_o = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_d = cv2.cvtColor(damaged, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta_e = np.sqrt(np.sum((lab_o - lab_d) ** 2, axis=2))

    gray_o = grayscale(orig)
    gray_d = grayscale(damaged)
    grad_o = cv2.Laplacian(gray_o, cv2.CV_32F, ksize=3)
    grad_d = cv2.Laplacian(gray_d, cv2.CV_32F, ksize=3)
    grad_diff = np.abs(grad_o - grad_d)

    try:
        from skimage.filters.rank import entropy as rank_entropy
        from skimage.morphology import disk
        ent_o = rank_entropy(gray_o, disk(5)).astype(np.float32)
        ent_d = rank_entropy(gray_d, disk(5)).astype(np.float32)
    except Exception:
        ent_o = local_entropy_fallback(gray_o)
        ent_d = local_entropy_fallback(gray_d)
    entropy_drop = np.clip(ent_o - ent_d, 0, None)

    hsv_o = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_d = cv2.cvtColor(damaged, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_o, sat_d = hsv_o[..., 1], hsv_d[..., 1]
    val_o, val_d = hsv_o[..., 2], hsv_d[..., 2]
    chroma_loss = np.clip(sat_o - sat_d, 0, None)
    value_drop = np.clip(val_o - val_d, 0, None)

    hue_o = hsv_o[..., 0]
    hue_d = hsv_d[..., 0]
    hue_diff = np.abs(hue_o - hue_d)
    hue_shift = np.minimum(hue_diff, 180.0 - hue_diff)
    hue_shift *= sat_o / (sat_o.max() + 1e-8)

    def local_variance(img: np.ndarray) -> np.ndarray:
        vars_ = []
        for c in range(3):
            ch = img[..., c].astype(np.float32)
            mean = cv2.GaussianBlur(ch, (15, 15), 0)
            sq = cv2.GaussianBlur(ch * ch, (15, 15), 0)
            vars_.append(np.clip(sq - mean * mean, 0, None))
        return np.mean(vars_, axis=0)

    local_var_drop = np.clip(local_variance(orig) - local_variance(damaged), 0, None)

    flow = cv2.calcOpticalFlowFarneback(gray_o, gray_d, None, 0.5, 4, 21, 5, 7, 1.5, 0)
    motion_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    warped_o = cv2.remap(
        gray_o,
        (np.arange(gray_o.shape[1])[None, :] + flow[..., 0]).astype(np.float32),
        (np.arange(gray_o.shape[0])[:, None] + flow[..., 1]).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warp_err = np.abs(warped_o.astype(np.float32) - gray_d.astype(np.float32))
    flow_conf = 1.0 - norm01(warp_err)

    sob_o = cv2.Canny(gray_o, 80, 180)
    sob_d = cv2.Canny(gray_d, 80, 180)
    edge_break = np.clip(sob_o.astype(np.float32) / 255.0 - sob_d.astype(np.float32) / 255.0, 0, None)

    return Signals(
        delta_e=delta_e,
        grad_diff=grad_diff,
        entropy_drop=entropy_drop,
        chroma_loss=chroma_loss,
        hue_shift=hue_shift,
        value_drop=value_drop,
        local_var_drop=local_var_drop,
        motion_mag=motion_mag,
        flow_conf=flow_conf,
        edge_break=edge_break,
    )



def compute_issue_maps(sig: Signals) -> IssueMaps:
    n = {k: norm01(v) for k, v in sig.__dict__.items()}

    photometric = (
        0.34 * n["delta_e"]
        + 0.24 * n["chroma_loss"]
        + 0.24 * n["hue_shift"]
        + 0.18 * n["value_drop"]
    )

    geometric = (
        0.42 * n["motion_mag"]
        + 0.25 * n["flow_conf"]
        + 0.18 * n["grad_diff"]
        + 0.15 * n["edge_break"]
    )

    texture = (
        0.40 * n["entropy_drop"]
        + 0.25 * n["grad_diff"]
        + 0.20 * n["delta_e"]
        + 0.15 * (1.0 - n["flow_conf"])
    )

    missing = (
        0.34 * n["local_var_drop"]
        + 0.24 * n["edge_break"]
        + 0.22 * n["grad_diff"]
        + 0.20 * n["value_drop"]
    )

    severity = norm01(np.maximum.reduce([photometric, geometric, texture, missing]))

    stack = np.stack([photometric, geometric, texture, missing], axis=-1)
    peak = np.max(stack, axis=-1)
    clean = np.clip(1.0 - norm01(peak), 0, 1)

    k = 31
    photometric = cv2.GaussianBlur(norm01(photometric), (k, k), 0)
    geometric = cv2.GaussianBlur(norm01(geometric), (k, k), 0)
    texture = cv2.GaussianBlur(norm01(texture), (k, k), 0)
    missing = cv2.GaussianBlur(norm01(missing), (k, k), 0)
    clean = cv2.GaussianBlur(norm01(clean), (k, k), 0)
    severity = cv2.GaussianBlur(norm01(severity), (k, k), 0)

    return IssueMaps(
        photometric=photometric.astype(np.float32),
        geometric=geometric.astype(np.float32),
        texture=texture.astype(np.float32),
        missing=missing.astype(np.float32),
        clean=clean.astype(np.float32),
        severity=severity.astype(np.float32),
    )



# def routing_from_issues(issues: IssueMaps) -> RoutingWeights:
#     # linear = 1.15 * issues.photometric + 0.20 * issues.clean
#     # flow = 1.10 * issues.geometric + 0.15 * issues.photometric
#     # laplacian = 1.05 * issues.texture + 0.20 * issues.photometric
#     # inpaint = 1.20 * issues.missing + 0.10 * issues.texture
#     linear = 0.95 * issues.photometric + 0.15 * issues.clean
#     flow = 0.95 * issues.geometric + 0.10 * issues.photometric
#     laplacian = 0.90 * issues.texture + 0.10 * issues.photometric
#     inpaint = 1.80 * issues.missing + 0.35 * issues.texture + 0.20 * issues.photometric



#     stack = np.stack([linear, flow, laplacian, inpaint], axis=-1).astype(np.float32)
#     exp = np.exp(np.clip(stack / 0.25, -40, 40))
#     soft = exp / (exp.sum(axis=-1, keepdims=True) + 1e-8)
#     return RoutingWeights(
#         linear=soft[..., 0].astype(np.float32),
#         flow=soft[..., 1].astype(np.float32),
#         laplacian=soft[..., 2].astype(np.float32),
#         inpaint=soft[..., 3].astype(np.float32),
#     )
def routing_from_issues(issues: IssueMaps) -> RoutingWeights:
    linear = 0.80 * issues.photometric + 0.10 * issues.clean
    flow = 0.80 * issues.geometric + 0.08 * issues.photometric
    laplacian = 0.75 * issues.texture + 0.08 * issues.photometric
    inpaint = 2.80 * issues.missing + 0.60 * issues.texture + 0.25 * issues.photometric

    stack = np.stack([linear, flow, laplacian, inpaint], axis=-1).astype(np.float32)
    exp = np.exp(np.clip(stack / 0.25, -40, 40))
    soft = exp / (exp.sum(axis=-1, keepdims=True) + 1e-8)

    return RoutingWeights(
        linear=soft[..., 0].astype(np.float32),
        flow=soft[..., 1].astype(np.float32),
        laplacian=soft[..., 2].astype(np.float32),
        inpaint=soft[..., 3].astype(np.float32),
    )


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def save_signal_diagnostics(orig: np.ndarray, sig: Signals, out_dir: str) -> None:
    ensure_dir(out_dir)
    for name, arr in sig.__dict__.items():
        cm = colorize_map(arr)
        save_image(os.path.join(out_dir, f"signal_{name}.jpg"), cm)
        save_image(os.path.join(out_dir, f"signal_{name}_overlay.jpg"), overlay(orig, cm))



def save_issue_diagnostics(orig: np.ndarray, issues: IssueMaps, routing: RoutingWeights, out_dir: str) -> None:
    ensure_dir(out_dir)
    for name, arr in issues.__dict__.items():
        cm = colorize_map(arr)
        save_image(os.path.join(out_dir, f"issue_{name}.jpg"), cm)
        save_image(os.path.join(out_dir, f"issue_{name}_overlay.jpg"), overlay(orig, cm))

    labels = np.argmax(np.stack([routing.linear, routing.flow, routing.laplacian, routing.inpaint], axis=-1), axis=-1)
    colors = {
        0: (60, 220, 60),    # linear -> green
        1: (255, 120, 0),    # flow -> orange
        2: (220, 50, 180),   # laplacian -> magenta
        3: (30, 30, 220),    # inpaint -> red
    }
    route_map = np.zeros((*labels.shape, 3), np.uint8)
    for idx, col in colors.items():
        route_map[labels == idx] = col
    save_image(os.path.join(out_dir, "routing_map.jpg"), route_map)
    save_image(os.path.join(out_dir, "routing_overlay.jpg"), overlay(orig, route_map, 0.55, 0.45))


# ---------------------------------------------------------------------
# Interpolation methods
# ---------------------------------------------------------------------

def remap_with_flow(img: np.ndarray, flow: np.ndarray, scale: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    fx = flow[..., 0] * scale
    fy = flow[..., 1] * scale
    mx = (np.arange(w)[None, :] + fx).astype(np.float32)
    my = (np.arange(h)[:, None] + fy).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def frames_to_video(
    frames_dir: str | Path,
    output_path: str | Path = "output_video.mp4",
    fps: int = 10,
    pattern: str = "frame_*.jpg",
) -> None:
    frames_dir = str(frames_dir)
    output_path = str(output_path)
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, pattern)))

    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir} with pattern {pattern}")

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {frame_paths[0]}")

    h, w = first_frame.shape[:2]
    ensure_dir(Path(output_path).parent)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_path}")

    for fp in frame_paths:
        frame = cv2.imread(fp)
        if frame is None:
            print(f"Skipping unreadable frame: {fp}")
            continue

        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)

        writer.write(frame)

    writer.release()
    print(f"Video saved to: {output_path}")



def linear_blend(a: np.ndarray, b: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    aa = alpha[..., None].astype(np.float32)
    return to_u8(a.astype(np.float32) * (1.0 - aa) + b.astype(np.float32) * aa)



def flow_blend(a: np.ndarray, b: np.ndarray, flow: np.ndarray,
               alpha: np.ndarray, t: float) -> np.ndarray:
    if t <= 1e-6:
        return a.copy()
    if t >= 1.0 - 1e-6:
        return b.copy()

    scale = np.full(flow.shape[:2], t, dtype=np.float32)
    warped = remap_with_flow(a, flow, scale)

    aa = np.clip(alpha[..., None].astype(np.float32), 0.0, 1.0)
    out = warped.astype(np.float32) * (1.0 - aa) + b.astype(np.float32) * aa
    return np.clip(out, 0, 255).astype(np.uint8)



def lap_pyramid(img: np.ndarray, levels: int = 6) -> List[np.ndarray]:
    g = [img.astype(np.float32)]
    for _ in range(levels - 1):
        g.append(cv2.pyrDown(g[-1]))
    lap = []
    for i in range(levels - 1):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        lap.append(g[i] - up)
    lap.append(g[-1])
    return lap



def recon_pyramid(pyr: List[np.ndarray]) -> np.ndarray:
    img = pyr[-1]
    for p in reversed(pyr[:-1]):
        img = cv2.pyrUp(img, dstsize=(p.shape[1], p.shape[0])) + p
    return img



def laplacian_blend(a: np.ndarray, b: np.ndarray, alpha: np.ndarray, levels: int = 5) -> np.ndarray:
    # valid region in damaged image: not near-black padding
    valid = (np.mean(b.astype(np.float32), axis=2) > 8).astype(np.float32)

    # only allow blend where damaged image is valid
    alpha_safe = np.clip(alpha * valid, 0.0, 1.0).astype(np.float32)

    pa = lap_pyramid(a, levels)
    pb = lap_pyramid(b, levels)
    blended = []

    for la, lb in zip(pa, pb):
        m = cv2.resize(alpha_safe, (la.shape[1], la.shape[0]), interpolation=cv2.INTER_LINEAR)
        m = np.clip(m, 0.0, 1.0)[..., None]
        blended.append(la * (1.0 - m) + lb * m)

    recon = recon_pyramid(blended)
    recon = np.clip(recon, 0, 255)

    # small linear anchor for stability
    base = a.astype(np.float32) * (1.0 - alpha_safe[..., None]) + b.astype(np.float32) * alpha_safe[..., None]
    out = 0.85 * recon + 0.15 * base
    return np.clip(out, 0, 255).astype(np.uint8)


def make_progressive_missing_mask(base_missing: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    # Noise std shrinks with alpha so early frames have gentle variation
    # but later frames converge to a stable, deterministic mask.
    # Fixed std=0.15 was causing pixels to randomly flip in/out between
    # consecutive frames, producing visible flicker in the inpaint output.
    base_missing = cv2.GaussianBlur(base_missing.astype(np.float32), (9, 9), 0)
    base_missing = norm01(base_missing)
    noise_std = 0.10 * (1.0 - alpha)
    noise = rng.normal(0.0, noise_std, size=base_missing.shape).astype(np.float32)
    score = norm01(base_missing + noise)
    thresh = 1.0 - alpha
    mask = (score >= thresh).astype(np.uint8) * 255
    k = max(3, int(3 + alpha * 11)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


# def make_progressive_missing_mask(base_missing: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
#     base = cv2.GaussianBlur(base_missing.astype(np.float32), (9, 9), 0)
#     base = norm01(base)

#     noise_std = 0.03 * (1.0 - alpha)
#     noise = rng.normal(0.0, noise_std, size=base.shape).astype(np.float32)

#     score = norm01(base + noise)

#     # broader mask that still grows with time
#     thresh = 0.78 - 0.38 * alpha
#     mask = (score >= thresh).astype(np.uint8) * 255

#     k = max(5, int(5 + alpha * 11)) | 1
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

#     mask = cv2.dilate(mask, kernel, iterations=1)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.GaussianBlur(mask, (k, k), 0)

#     return mask

def residual_damage_map(orig: np.ndarray, damaged: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(orig.astype(np.float32) - damaged.astype(np.float32)), axis=2)
    diff = cv2.GaussianBlur(diff, (9, 9), 0)
    return norm01(diff)


# def inpaint_progressive(orig: np.ndarray, damaged: np.ndarray, missing_map: np.ndarray,
#                         alpha: float, rng: np.random.Generator) -> np.ndarray:
#     mask = make_progressive_missing_mask(missing_map, alpha, rng)
#     binary = (mask > 20).astype(np.uint8) * 255

#     inpainted = cv2.inpaint(orig, binary, 5, cv2.INPAINT_TELEA)

#     gray_d = cv2.cvtColor(damaged, cv2.COLOR_BGR2GRAY)
#     bad = (gray_d < 8).astype(np.uint8) * 255
#     if bad.max() > 0:
#         damaged_safe = cv2.inpaint(damaged, bad, 3, cv2.INPAINT_TELEA)
#     else:
#         damaged_safe = damaged.copy()

#     substrate = cv2.GaussianBlur(damaged_safe, (41, 41), 0)

#     hybrid = to_u8(0.80 * inpainted.astype(np.float32) + 0.20 * substrate.astype(np.float32))

#     aa = (mask.astype(np.float32) / 255.0)[..., None]
#     # inside = hybrid.astype(np.float32) * (1.0 - alpha) + damaged_safe.astype(np.float32) * alpha
#     inside = hybrid.astype(np.float32) * (1.0 - 0.65 * alpha) + damaged_safe.astype(np.float32) * (0.65 * alpha)
#     result = inside * aa + orig.astype(np.float32) * (1.0 - aa)
#     return to_u8(result)

def inpaint_progressive(orig: np.ndarray, damaged: np.ndarray, missing_map: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    mask = make_progressive_missing_mask(missing_map, alpha, rng)
    binary = (mask > 20).astype(np.uint8) * 255
    inpainted = cv2.inpaint(orig, binary, 5, cv2.INPAINT_TELEA)
    substrate = cv2.GaussianBlur(damaged, (41, 41), 0)
    hybrid = to_u8(0.55 * inpainted.astype(np.float32) + 0.45 * substrate.astype(np.float32))
    aa = (mask.astype(np.float32) / 255.0)[..., None]
    # Inside mask:  transition hybrid substrate → damaged image colour
    # Outside mask: keep original exactly — previous formula darkened
    #               undamaged regions by mixing (1-alpha)*orig outside mask
    inside = hybrid.astype(np.float32) * (1.0 - alpha) + damaged.astype(np.float32) * alpha
    result = inside * aa + orig.astype(np.float32) * (1.0 - aa)
    return to_u8(result)

# ---------------------------------------------------------------------
# Unified adaptive synthesis
# ---------------------------------------------------------------------

# def temporal_curve(alpha: float) -> float:
#     return alpha * alpha * (3.0 - 2.0 * alpha)

def temporal_curve(alpha: float) -> float:
    return alpha



def build_transition_frame(
    orig: np.ndarray,
    damaged: np.ndarray,
    issues: IssueMaps,
    routing: RoutingWeights,
    flow: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if alpha >= 1.0:
        return damaged.copy()

    t = temporal_curve(alpha)
    sev = issues.severity

    # linear_alpha = np.clip(t * sev * (0.65 + 0.35 * routing.linear), 0, 1)
    # flow_alpha = np.clip(t * sev * (0.65 + 0.35 * routing.flow), 0, 1)
    # lap_alpha = np.clip(t * sev * (0.65 + 0.35 * routing.laplacian), 0, 1)

    progress = np.clip(0.10 + 0.90 * t, 0, 1)

    linear_alpha = np.clip(progress * (0.35 + 0.65 * sev) * (0.55 + 0.45 * routing.linear), 0, 1)
    flow_alpha   = np.clip(progress * (0.35 + 0.65 * sev) * (0.55 + 0.45 * routing.flow), 0, 1)
    lap_alpha    = np.clip(progress * (0.35 + 0.65 * sev) * (0.55 + 0.45 * routing.laplacian), 0, 1)


    out_linear = linear_blend(orig, damaged, linear_alpha)
    out_flow = flow_blend(orig, damaged, flow, flow_alpha, t)
    out_lap = laplacian_blend(orig, damaged, lap_alpha)
    # out_inpaint = inpaint_progressive(orig, damaged, issues.missing, t, rng)
    damage_core = np.maximum.reduce([
    issues.missing,
    0.65 * issues.photometric,
    0.45 * issues.texture
    ]).astype(np.float32)
    resid = residual_damage_map(orig, damaged)
    damage_map = np.maximum.reduce([
        1.00 * issues.missing,
        0.85 * issues.photometric,
        0.75 * issues.texture,
        0.55 * issues.geometric
    ]).astype(np.float32)

    damage_map = norm01(cv2.GaussianBlur(damage_map, (9, 9), 0))

    out_inpaint = inpaint_progressive(orig, damaged, damage_map, t, rng)
    

    # wl = routing.linear[..., None]
    # wf = routing.flow[..., None]
    # wp = routing.laplacian[..., None]
    # wi = routing.inpaint[..., None]

    # norm = wl + wf + wp + wi + 1e-8
    # wl, wf, wp, wi = wl / norm, wf / norm, wp / norm, wi / norm 

    wl = routing.linear[..., None]
    wf = routing.flow[..., None]
    wp = routing.laplacian[..., None]
    wi = 1.8 * routing.inpaint[..., None]   # strongest direct boost

    norm = wl + wf + wp + wi + 1e-8
    wl, wf, wp, wi = wl / norm, wf / norm, wp / norm, wi / norm

    mix = (
    out_linear.astype(np.float32) * wl
    + out_flow.astype(np.float32) * wf
    + out_lap.astype(np.float32) * wp
    + out_inpaint.astype(np.float32) * wi
    )

    # low-frequency background / intensity progression
    orig_low = cv2.GaussianBlur(orig, (0, 0), 12).astype(np.float32)
    damaged_low = cv2.GaussianBlur(damaged, (0, 0), 12).astype(np.float32)
    lowfreq = orig_low * (1.0 - alpha) + damaged_low * alpha

    bg_strength = 0.20 + 0.30 * alpha
    bg_strength_map = bg_strength * (0.30 + 0.70 * (issues.clean + 0.5 * issues.photometric))[..., None]
    bg_strength_map = np.clip(bg_strength_map, 0, 0.40)

    mix = mix * (1.0 - bg_strength_map) + lowfreq * bg_strength_map

    # stronger late-stage approach so the penultimate frame is already close
    late_gate = np.clip((alpha - 0.80) / 0.20, 0.0, 1.0)
    late_gate = late_gate * late_gate * (3.0 - 2.0 * late_gate)

    late_strength = (0.18 + 0.32 * late_gate) * (0.35 + 0.65 * sev[..., None])
    late_strength = np.clip(late_strength, 0.0, 0.45)

    result = mix * (1.0 - late_strength) + damaged.astype(np.float32) * late_strength

    if alpha >= 0.999:
        return damaged.copy()
    return to_u8(result)


# ---------------------------------------------------------------------
# Evaluation and sheets
# ---------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float | str]:
    mse_val = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
    psnr_val = float(psnr(target, pred, data_range=255))
    pred_l = cv2.cvtColor(pred, cv2.COLOR_BGR2LAB)[..., 0]
    tgt_l = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)[..., 0]
    ssim_val = float(ssim(tgt_l, pred_l, data_range=255))
    out: Dict[str, float | str] = {
        "MSE": round(mse_val, 4),
        "PSNR": round(psnr_val, 4),
        "SSIM": round(ssim_val, 6),
    }
    try:
        import torch
        import lpips

        fn = lpips.LPIPS(net="alex", verbose=False)

        def tensorize(img: np.ndarray):
            return torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1

        with torch.no_grad():
            out["LPIPS"] = round(float(fn(tensorize(pred), tensorize(target)).item()), 6)
    except Exception:
        out["LPIPS"] = "N/A"
    return out



def print_metrics_table(results: Dict[str, Dict[str, float | str]]) -> None:
    if not results:
        return
    methods = list(results.keys())
    cols = list(next(iter(results.values())).keys())
    col_w = max(len(m) for m in methods) + 2
    bar = "─" * (col_w + len(cols) * 18)
    print(f"\n{bar}")
    print(f"{'Method':<{col_w}}" + "".join(f"{c:>18}" for c in cols))
    print(bar)
    for method, vals in results.items():
        print(f"{method:<{col_w}}" + "".join(f"{str(v):>18}" for v in vals.values()))
    print(bar)



def save_metrics_csv(results: Dict[str, Dict[str, float | str]], path: str) -> None:
    cols = list(next(iter(results.values())).keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write("method," + ",".join(cols) + "\n")
        for method, vals in results.items():
            f.write(method + "," + ",".join(str(v) for v in vals.values()) + "\n")



def label_tile(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, img.shape[1] / 2200)
    cv2.putText(out, text, (10, 28), font, scale, (0, 0, 0), 2)
    cv2.putText(out, text, (10, 28), font, scale, (255, 255, 255), 1)
    return out



def make_contact_sheet(orig: np.ndarray, frames: List[np.ndarray], damaged: np.ndarray, prefix: str) -> np.ndarray:
    tiles = [label_tile(orig, "ORIGINAL")]
    for i, fr in enumerate(frames, start=1):
        tiles.append(label_tile(fr, f"{prefix} {i}"))
    tiles.append(label_tile(damaged, "DAMAGED"))
    return np.hstack(tiles)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run_pipeline(
    original_path: str,
    damaged_path: str,
    num_frames: int = 7,
    output_dir: str = "results",
    ground_truth_path: str | None = None,
    skip_align: bool = False,
    seed: int = 7,
    fps: int = 10,
) -> None:
    rng = np.random.default_rng(seed)

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║ Adaptive Artwork Damage Analysis and Interpolation Pipeline ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    ensure_dir(output_dir)
    orig = load_image(original_path)
    damaged = load_image(damaged_path)

    section("Stage 0 — Alignment")
    if skip_align:
        print("Skipping alignment; resizing damaged image to original size.")
        damaged = resize_to_match(damaged, orig)
    else:
        damaged = align_images(damaged, orig, os.path.join(output_dir, "alignment_debug.jpg"))
        print(f"Aligned damaged image to {damaged.shape[:2]}")

    gt = None
    if ground_truth_path is not None:
        gt = resize_to_match(load_image(ground_truth_path), orig)

    section("Stage 1 — Signal extraction")
    sig = compute_signals(orig, damaged)
    save_signal_diagnostics(orig, sig, os.path.join(output_dir, "signals"))
    print("Saved signal diagnostics.")

    section("Stage 2 — Issue analysis and routing")
    issues = compute_issue_maps(sig)
    routing = routing_from_issues(issues)
    save_issue_diagnostics(orig, issues, routing, os.path.join(output_dir, "analysis"))
    print("Saved issue maps and routing diagnostics.")

    flow = cv2.calcOpticalFlowFarneback(grayscale(orig), grayscale(damaged), None, 0.5, 4, 21, 5, 7, 1.5, 0)

    section("Stage 3 — Adaptive transition synthesis")
    frame_dir = os.path.join(output_dir, "adaptive_transition")
    ensure_dir(frame_dir)

    # Include the exact damaged image as the last generated frame.
    # alphas = np.linspace(0.0, 1.0, num_frames + 2, dtype=np.float32)[1:]
    # u = np.linspace(0, 1, num_frames, dtype=np.float32)
    # alphas = 0.65 * u + 0.35 * (u ** 1.35)
    # alphas[-1] = 1.0
    u = np.linspace(0, 1, num_frames, dtype=np.float32)
    alphas = 1.0 - (1.0 - u) ** 1.8
    alphas[-1] = 1.0
    frames: List[np.ndarray] = []
    for idx, alpha in enumerate(alphas, start=1):
        fr = build_transition_frame(orig, damaged, issues, routing, flow, float(alpha), rng)
        frames.append(fr)
        save_image(os.path.join(frame_dir, f"frame_{idx:03d}_a{alpha:.3f}.jpg"), fr)
        print(f"  frame {idx}/{len(alphas)} alpha={alpha:.3f} saved")

    save_image(os.path.join(frame_dir, "_contact_sheet.jpg"), make_contact_sheet(orig, frames, damaged, "adaptive"))
    frames_to_video(frame_dir, os.path.join(frame_dir, "adaptive_transition.mp4"), fps=fps)

    section("Stage 4 — Baseline method outputs")
    baselines = {
        "linear": [],
        "optical_flow": [],
        "laplacian": [],
        "inpaint": [],
    }
    for idx, alpha in enumerate(alphas, start=1):
        t = temporal_curve(float(alpha))
        baselines["linear"].append(linear_blend(orig, damaged, np.full(orig.shape[:2], t, np.float32)))
        baselines["optical_flow"].append(flow_blend(orig, damaged, flow, np.full(orig.shape[:2], t, np.float32), t))
        baselines["laplacian"].append(laplacian_blend(orig, damaged, np.full(orig.shape[:2], t, np.float32)))
        baselines["inpaint"].append(inpaint_progressive(orig, damaged, issues.missing, t, rng))

    for name, frs in baselines.items():
        out = os.path.join(output_dir, f"baseline_{name}")
        ensure_dir(out)
        for i, fr in enumerate(frs, start=1):
            save_image(os.path.join(out, f"frame_{i:03d}.jpg"), fr)
        save_image(os.path.join(out, "_contact_sheet.jpg"), make_contact_sheet(orig, frs, damaged, name))
        frames_to_video(out, os.path.join(out, f"baseline_{name}.mp4"), fps=fps)

    section("Stage 5 — Evaluation")
    results: Dict[str, Dict[str, float | str]] = {}
    if gt is not None:
        mid_idx = len(frames) // 2
        results["adaptive"] = compute_metrics(frames[mid_idx], gt)
        for name, frs in baselines.items():
            results[name] = compute_metrics(frs[mid_idx], gt)
        print_metrics_table(results)
        save_metrics_csv(results, os.path.join(output_dir, "evaluation.csv"))
        print(f"Saved evaluation.csv")
    else:
        print("No ground truth supplied. Skipping metrics.")

    section("Done")
    print(f"Output directory: {output_dir}")
    print("Main deliverables:")
    print("  alignment_debug.jpg")
    print("  signals/")
    print("  analysis/ (issue maps + routing map)")
    print("  adaptive_transition/ (primary result + adaptive_transition.mp4)")
    print("  baseline_linear/ (includes baseline_linear.mp4)")
    print("  baseline_optical_flow/ (includes baseline_optical_flow.mp4)")
    print("  baseline_laplacian/ (includes baseline_laplacian.mp4)")
    print("  baseline_inpaint/ (includes baseline_inpaint.mp4)")
    if gt is not None:
        print("  evaluation.csv")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive artwork damage analysis and interpolation pipeline")
    p.add_argument("--original", required=True, help="Path to original/reference image")
    p.add_argument("--damaged", required=True, help="Path to damaged/distorted image")
    p.add_argument("--num_frames", type=int, default=7, help="Number of intermediate frames to generate")
    p.add_argument("--output_dir", default="results", help="Directory to save outputs")
    p.add_argument("--ground_truth", default=None, help="Optional midpoint ground-truth frame for evaluation")
    p.add_argument("--no_align", action="store_true", help="Skip registration and only resize")
    p.add_argument("--seed", type=int, default=7, help="Random seed for deterministic progressive masks")
    p.add_argument("--fps", type=int, default=10, help="Frames per second for saved MP4 videos")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        original_path=args.original,
        damaged_path=args.damaged,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        ground_truth_path=args.ground_truth,
        skip_align=args.no_align,
        seed=args.seed,
        fps=args.fps,
    )
