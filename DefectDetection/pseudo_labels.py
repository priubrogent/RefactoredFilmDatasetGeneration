"""
Pseudo-label generation from aligned film triplets (scan, r1, r2).

Core idea
---------
A pixel in the scan is a *defect* when:
  1. The scan differs significantly from restored_1 AND restored_2
     (both restorations removed it).
  2. restored_1 and restored_2 *agree* with each other
     (independent restorations converged on the same clean pixel).

Both conditions must hold simultaneously, which eliminates two major
false-positive classes from the old gradient heuristic:
  - Sharpness differences: the restored version is globally sharper than the
    scan — both diffs would be large everywhere, but condition 2 might still
    hold; however the diff is globally uniform so local normalisation removes
    it.
  - Style / colour grade differences: a warm-vs-cool grade shift makes the
    raw pixel diff large everywhere, but local normalisation (subtracting the
    local mean from each channel) removes smooth, spatially-varying offsets,
    leaving only local anomalies.

Colour space
------------
We work in CIE L*a*b* by default.  Euclidean distance in Lab is closer to
perceptual dissimilarity than in sRGB, so a threshold of ~12 lab-units
corresponds roughly to a "just noticeable" difference.

Usage
-----
    from pseudo_labels import compute_pseudo_label, PseudoLabelConfig

    cfg  = PseudoLabelConfig()
    mask, cov = compute_pseudo_label(scan_bgr, r1_bgr, r2_bgr, cfg)
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class PseudoLabelConfig:
    diff_colorspace:     str   = "lab"   # "lab" or "rgb"
    local_norm_radius:   int   = 31      # window for local mean subtraction (0=off)
    threshold:           float = 12.0   # scan-vs-restored min diff to flag (0-255 scale)
    agreement_threshold: float = 18.0   # max r1-r2 diff to treat as "agreeing"
    morph_open_kernel:   int   = 3      # morphological opening to remove isolated px

    @classmethod
    def from_dict(cls, d: dict) -> "PseudoLabelConfig":
        return cls(
            diff_colorspace=d.get("diff_colorspace", "lab"),
            local_norm_radius=d.get("local_norm_radius", 31),
            threshold=d.get("threshold", 12.0),
            agreement_threshold=d.get("agreement_threshold", 18.0),
            morph_open_kernel=d.get("morph_open_kernel", 3),
        )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_pseudo_label(
    scan: np.ndarray,   # uint8 BGR H×W×3
    r1:   np.ndarray,   # uint8 BGR H×W×3
    r2:   np.ndarray,   # uint8 BGR H×W×3
    cfg:  Optional[PseudoLabelConfig] = None,
) -> tuple[np.ndarray, float]:
    """
    Returns:
        mask     — uint8 H×W, 255 = defect pixel
        coverage — float, fraction of defect pixels in [0, 1]
    """
    if cfg is None:
        cfg = PseudoLabelConfig()

    # Convert colour space
    if cfg.diff_colorspace == "lab":
        scan_f = _to_lab(scan)
        r1_f   = _to_lab(r1)
        r2_f   = _to_lab(r2)
    else:
        scan_f = scan.astype(np.float32)
        r1_f   = r1.astype(np.float32)
        r2_f   = r2.astype(np.float32)

    # Local normalisation: subtract local mean per channel per image
    if cfg.local_norm_radius > 0:
        ksize  = _odd(cfg.local_norm_radius)
        scan_f = _local_subtract_mean(scan_f, ksize)
        r1_f   = _local_subtract_mean(r1_f,   ksize)
        r2_f   = _local_subtract_mean(r2_f,   ksize)

    # Per-pixel L2 distance across colour channels
    d_scan_r1 = _l2(scan_f - r1_f)   # scan vs r1
    d_scan_r2 = _l2(scan_f - r2_f)   # scan vs r2
    d_r1_r2   = _l2(r1_f   - r2_f)   # agreement between restorations

    # Defect condition
    defect = (
        (d_scan_r1 > cfg.threshold) &
        (d_scan_r2 > cfg.threshold) &
        (d_r1_r2   < cfg.agreement_threshold)
    )

    # Remove isolated noise pixels
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.morph_open_kernel, cfg.morph_open_kernel),
    )
    mask = cv2.morphologyEx(
        defect.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        kernel,
    )

    coverage = float((mask > 0).mean())
    return mask, coverage


# ---------------------------------------------------------------------------
# Soft / confidence-weighted variant
# ---------------------------------------------------------------------------

def compute_soft_pseudo_label(
    scan: np.ndarray,
    r1:   np.ndarray,
    r2:   np.ndarray,
    cfg:  Optional[PseudoLabelConfig] = None,
) -> np.ndarray:
    """
    Returns a float32 H×W confidence map in [0, 1].

    Higher values = more confident this is a defect pixel.
    Useful for soft-label training or visualisation.
    """
    if cfg is None:
        cfg = PseudoLabelConfig()

    if cfg.diff_colorspace == "lab":
        scan_f = _to_lab(scan)
        r1_f   = _to_lab(r1)
        r2_f   = _to_lab(r2)
    else:
        scan_f = scan.astype(np.float32)
        r1_f   = r1.astype(np.float32)
        r2_f   = r2.astype(np.float32)

    if cfg.local_norm_radius > 0:
        ksize  = _odd(cfg.local_norm_radius)
        scan_f = _local_subtract_mean(scan_f, ksize)
        r1_f   = _local_subtract_mean(r1_f,   ksize)
        r2_f   = _local_subtract_mean(r2_f,   ksize)

    d_scan_r1 = _l2(scan_f - r1_f)
    d_scan_r2 = _l2(scan_f - r2_f)
    d_r1_r2   = _l2(r1_f   - r2_f)

    # Scan divergence score: geometric mean of both scan-vs-restored diffs
    scan_divergence = np.sqrt(
        np.clip(d_scan_r1 / (cfg.threshold + 1e-6), 0, None) *
        np.clip(d_scan_r2 / (cfg.threshold + 1e-6), 0, None)
    )

    # Agreement score: 1 when r1≈r2, 0 when they disagree
    agreement = np.clip(
        1.0 - d_r1_r2 / (cfg.agreement_threshold + 1e-6),
        0.0, 1.0,
    )

    confidence = np.clip(scan_divergence * agreement, 0.0, 1.0).astype(np.float32)
    return confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_lab(bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR → float32 Lab (L in 0-100, a/b in ±128)."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _l2(diff: np.ndarray) -> np.ndarray:
    """H×W×C → H×W, L2 norm across channel axis."""
    return np.sqrt(np.sum(diff ** 2, axis=2))


def _local_subtract_mean(img: np.ndarray, ksize: int) -> np.ndarray:
    """Subtract local mean per channel to remove spatially smooth offsets."""
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = img[:, :, c] - cv2.blur(img[:, :, c], (ksize, ksize))
    return out


def _odd(n: int) -> int:
    """Ensure n is odd (required by some OpenCV functions)."""
    return n if n % 2 == 1 else n + 1
