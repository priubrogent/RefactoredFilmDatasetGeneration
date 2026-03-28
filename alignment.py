"""
Spatial alignment and defect mask generation.

Faithful to the original NewTriAlign/alignment_utils.py that produced
good results, with the mask logic from main_align.py.

All images are expected as float32 BGR in [0, 1] range.
"""

import cv2
import numpy as np


def align_two_images(tgt: np.ndarray, src: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Align src to tgt using ECC with an affine motion model.

    Matches the original NewTriAlign/alignment_utils.py exactly:
      - INTER_LINEAR for resize
      - INTER_CUBIC for warp
      - 5000 iterations, eps 1e-6
      - confidence threshold 0.5

    Args:
        tgt: Target / reference image (float32 BGR, 0–1).
        src: Source image to warp onto tgt (float32 BGR, 0–1).

    Returns:
        (aligned, M): aligned is float32 BGR same size as tgt; M is 2×3 affine.
        Returns (None, None) on ECC failure.
    """
    src_gray = cv2.cvtColor((src * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor((tgt * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    if src.shape[:2] != tgt.shape[:2]:
        src_resized = cv2.resize(src, (tgt.shape[1], tgt.shape[0]), interpolation=cv2.INTER_LINEAR)
        src_gray    = cv2.resize(src_gray, (tgt.shape[1], tgt.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        src_resized = src

    M = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)

    try:
        cc, M = cv2.findTransformECC(src_gray, tgt_gray, M, cv2.MOTION_AFFINE, criteria, None, 1)

        if cc < 0.5:
            return cv2.resize(src, (tgt.shape[1], tgt.shape[0]), interpolation=cv2.INTER_LINEAR), None

        h, w = tgt.shape[:2]
        aligned = cv2.warpAffine(src_resized, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        return aligned, M

    except cv2.error as e:
        print(f"ECC failed: {e}")
        return None, None


def align_three_images(
    ref: np.ndarray,
    img2: np.ndarray,
    img3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Align img2 and img3 independently to ref."""
    img2_aligned, M2 = align_two_images(ref, img2)
    img3_aligned, M3 = align_two_images(ref, img3)
    return img2_aligned, img3_aligned, M2, M3


def generate_gradient_difference_mask(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compute Sobel gradient magnitude difference (img1 – img2).

    Large positive values → edges present in img1 but absent in img2.
    Matches NewTriAlign/alignment_utils.py exactly.

    Returns float32 H×W map (unbounded, can be negative).
    """
    img1_gray = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img2_gray = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def _grad_mag(gray):
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    return _grad_mag(img1_gray) - _grad_mag(img2_gray)


def compute_defect_mask(
    diff_scan_vs_r1: np.ndarray,
    diff_scan_vs_r2: np.ndarray,
    threshold: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute defect mask from two gradient-difference maps.

    Two outputs:
      mask_raw  — raw product map (diff1 * diff2), float32, used for
                  visualisation / soft weighting.
      mask      — binary uint8 mask (255 = defect) thresholded from mask_raw.

    Matches the original main_align.py logic:
        defects_scan = diff_mask_scan_1 * diff_mask_scan_2

    Returns:
        (mask_uint8, mask_raw_float32)
    """
    mask_raw = diff_scan_vs_r1 * diff_scan_vs_r2          # float32 product
    mask_bin = (mask_raw > threshold).astype(np.uint8) * 255
    return mask_bin, mask_raw.astype(np.float32)
