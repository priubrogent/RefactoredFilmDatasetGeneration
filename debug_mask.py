"""
Debug mask quality for a single segment folder.

Reads existing scan/, mask/, mask_raw/ images and lets you experiment with
alternative thresholding strategies — no video files needed.

Usage:
    python debug_mask.py --segment pipeline_output/segment000200-001640
    python debug_mask.py --segment pipeline_output/segment000200-001640 --frame 000600
    python debug_mask.py --segment pipeline_output/segment000200-001640 --all
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ---------------------------------------------------------------------------
# Recompute diffs from raw saved images (scan + restored_1 + restored_2)
# ---------------------------------------------------------------------------

def _grad_mag(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def recompute_diffs(scan_bgr, r1_bgr, r2_bgr):
    """Return (diff1, diff2) = grad(scan) - grad(r1/r2)."""
    g_scan = _grad_mag(scan_bgr)
    g_r1   = _grad_mag(r1_bgr)
    g_r2   = _grad_mag(r2_bgr)
    return g_scan - g_r1, g_scan - g_r2


# ---------------------------------------------------------------------------
# Alternative mask strategies to compare
# ---------------------------------------------------------------------------

def masks_from_diffs(diff1, diff2, threshold=0.10):
    """
    Returns a dict of mask variants for comparison.

    current  — original: product > threshold  (BUG: flags sharpness differences)
    positive — BOTH diffs > threshold  (true defects only)
    abs_both — abs(diff1)*abs(diff2) > threshold  (magnitude-only, sign-agnostic)
    morpho   — 'positive' mask after morphological open (removes isolated pixels)
    """
    product = diff1 * diff2

    # Original pipeline formula
    current = (product > threshold).astype(np.uint8) * 255

    # Only flag where scan has MORE edges than BOTH restorals
    positive = ((diff1 > threshold) & (diff2 > threshold)).astype(np.uint8) * 255

    # Magnitude-only: large difference in either direction
    abs_both = (np.abs(diff1) * np.abs(diff2) > threshold).astype(np.uint8) * 255

    # 'positive' with morphological opening to kill isolated noise pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morpho = cv2.morphologyEx(positive, cv2.MORPH_OPEN, kernel)

    return {
        "current\n(prod > thr)":   current,
        "positive only\n(both diffs > thr)": positive,
        "abs product\n(|d1|*|d2| > thr)":    abs_both,
        "positive +\nmorpho open":            morpho,
    }


# ---------------------------------------------------------------------------
# Single-frame debug visualisation
# ---------------------------------------------------------------------------

def debug_frame(seg_dir: str, frame_tag: str, threshold: float, out_dir: str):
    scan_p  = os.path.join(seg_dir, "scan",       frame_tag)
    r1_p    = os.path.join(seg_dir, "restored_1", frame_tag)
    r2_p    = os.path.join(seg_dir, "restored_2", frame_tag)
    mask_p  = os.path.join(seg_dir, "mask",       frame_tag)
    raw_p   = os.path.join(seg_dir, "mask_raw",   frame_tag)

    missing = [p for p in [scan_p, r1_p, r2_p] if not os.path.exists(p)]
    if missing:
        print(f"  [skip] missing: {missing}")
        return

    scan = cv2.imread(scan_p)
    r1   = cv2.imread(r1_p)
    r2   = cv2.imread(r2_p)
    mask_saved = cv2.imread(mask_p,  cv2.IMREAD_GRAYSCALE)
    raw_saved  = cv2.imread(raw_p,   cv2.IMREAD_GRAYSCALE)

    diff1, diff2 = recompute_diffs(scan, r1, r2)
    variants     = masks_from_diffs(diff1, diff2, threshold=threshold)

    scan_rgb = cv2.cvtColor(scan, cv2.COLOR_BGR2RGB)

    # ── Figure 1: diff maps + mask variants ──────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(f"Mask debug — {frame_tag}  (threshold={threshold})", fontsize=13)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

    # Row 0: diff1, diff2, product, scan
    norm_sym = dict(cmap="seismic", vmin=-0.4, vmax=0.4)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(diff1, **norm_sym); ax.set_title("diff1 = grad(scan)−grad(r1)"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(diff2, **norm_sym); ax.set_title("diff2 = grad(scan)−grad(r2)"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(diff1 * diff2, cmap="plasma", vmin=0, vmax=0.1); ax.set_title("product (diff1 × diff2)"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 3]); ax.imshow(scan_rgb); ax.set_title("scan"); ax.axis("off")

    # Row 1: histogram of diff1, diff2, product
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(diff1.ravel(), bins=200, range=(-0.5, 0.5), color="steelblue", alpha=0.7)
    ax.axvline(threshold,  color="red",   lw=1.5, label=f"+thr={threshold}")
    ax.axvline(-threshold, color="orange",lw=1.5, label=f"-thr={threshold}")
    ax.set_title("diff1 histogram"); ax.legend(fontsize=8); ax.set_yscale("log")

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(diff2.ravel(), bins=200, range=(-0.5, 0.5), color="coral", alpha=0.7)
    ax.axvline(threshold,  color="red",   lw=1.5)
    ax.axvline(-threshold, color="orange",lw=1.5)
    ax.set_title("diff2 histogram"); ax.set_yscale("log")

    ax = fig.add_subplot(gs[1, 2])
    product = diff1 * diff2
    ax.hist(product.ravel(), bins=200, range=(-0.05, 0.15), color="purple", alpha=0.7)
    ax.axvline(threshold, color="red", lw=1.5, label=f"thr={threshold}")
    ax.set_title("product histogram"); ax.legend(fontsize=8); ax.set_yscale("log")

    ax = fig.add_subplot(gs[1, 3])
    if raw_saved is not None:
        ax.imshow(raw_saved, cmap="plasma"); ax.set_title("saved mask_raw")
    ax.axis("off")

    # Row 2: mask variants
    for col, (name, m) in enumerate(variants.items()):
        ax = fig.add_subplot(gs[2, col])
        # Overlay mask on scan in red
        overlay = scan_rgb.copy()
        overlay[m > 0] = [255, 60, 60]
        ax.imshow(overlay)
        pct = 100.0 * (m > 0).sum() / m.size
        ax.set_title(f"{name}\n{pct:.2f}% pixels", fontsize=9)
        ax.axis("off")

    stem = Path(frame_tag).stem
    out_path = os.path.join(out_dir, f"mask_debug_{stem}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # ── Figure 2: quadrant comparison of positive-only vs current ────────────
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(f"Current vs positive-only — {frame_tag}", fontsize=12)

    # False positives: current flags but positive-only doesn't
    # (i.e. both diffs negative — sharpness artefact)
    current_mask = variants["current\n(prod > thr)"]
    pos_mask     = variants["positive only\n(both diffs > thr)"]
    false_pos    = (current_mask > 0) & (pos_mask == 0)

    overlay_curr = scan_rgb.copy(); overlay_curr[current_mask > 0] = [255, 60, 60]
    overlay_pos  = scan_rgb.copy(); overlay_pos[pos_mask > 0]      = [60, 255, 60]
    overlay_fp   = scan_rgb.copy(); overlay_fp[false_pos]          = [255, 165, 0]  # orange = false positives

    axes[0].imshow(overlay_curr); axes[0].set_title(f"Current mask  ({100*(current_mask>0).mean():.2f}%)"); axes[0].axis("off")
    axes[1].imshow(overlay_pos);  axes[1].set_title(f"Positive-only ({100*(pos_mask>0).mean():.2f}%)");     axes[1].axis("off")
    axes[2].imshow(overlay_fp);   axes[2].set_title("Orange = removed by fix\n(sharpness false positives)");axes[2].axis("off")

    out_path2 = os.path.join(out_dir, f"mask_comparison_{stem}.png")
    plt.savefig(out_path2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path2}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Debug mask quality from existing pipeline output")
    parser.add_argument("--segment", required=True, help="Path to segment folder")
    parser.add_argument("--frame",   default=None,  help="Specific frame tag e.g. 000600.png")
    parser.add_argument("--all",     action="store_true", help="Process all frames in segment")
    parser.add_argument("--threshold", type=float, default=0.10, help="Threshold to test (default: 0.10)")
    parser.add_argument("--output",  default=None, help="Output dir for debug PNGs (default: segment/debug/)")
    args = parser.parse_args()

    seg_dir = args.segment
    if not os.path.isdir(seg_dir):
        print(f"Segment folder not found: {seg_dir}")
        return

    out_dir = args.output or os.path.join(seg_dir, "debug", "mask_debug")
    os.makedirs(out_dir, exist_ok=True)

    scan_dir = os.path.join(seg_dir, "scan")
    all_frames = sorted(f for f in os.listdir(scan_dir) if f.endswith(".png"))

    if not all_frames:
        print("No frames found in scan/")
        return

    if args.frame:
        frames = [args.frame if args.frame.endswith(".png") else args.frame + ".png"]
    elif args.all:
        frames = all_frames
    else:
        # Default: pick 3 evenly spaced frames for a quick overview
        idxs = np.linspace(0, len(all_frames) - 1, min(3, len(all_frames)), dtype=int)
        frames = [all_frames[i] for i in idxs]
        print(f"No --frame or --all specified. Sampling {len(frames)} frames: {frames}")

    for tag in frames:
        print(f"  Processing {tag}...")
        debug_frame(seg_dir, tag, threshold=args.threshold, out_dir=out_dir)

    print(f"\nDone. Debug images in: {out_dir}")


if __name__ == "__main__":
    main()
