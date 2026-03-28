import numpy as np
from scipy.linalg import solve

CROP  = 150
ALPHA = 1e-3


def compute_color_matrix(reference: np.ndarray, source: np.ndarray) -> np.ndarray:
    """
    Compute 3x4 projective colour matrix H so that  source @ H ≈ reference.
    Both arrays are uint8 RGB (H, W, 3). Uses centre rows (crop 150px top/bottom).
    Returns H of shape (3, 4).
    """
    ref_f = reference[CROP:-CROP].astype(np.float64) / 255.0
    src_f = source[CROP:-CROP].astype(np.float64) / 255.0
    M = ref_f.shape[0] * ref_f.shape[1]
    I_s = src_f.reshape(M, 3)
    I_r = np.hstack([ref_f.reshape(M, 3), np.ones((M, 1))])
    ATA = I_s.T @ I_s
    ATA.flat[::ATA.shape[0] + 1] += ALPHA
    return solve(ATA, I_s.T @ I_r, assume_a="pos")


def apply_color_matrix(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply 3x4 colour matrix H to a full uint8 RGB image. Returns uint8 RGB same size."""
    h, w, _ = img.shape
    pix = img.reshape(-1, 3).astype(np.float64) / 255.0
    result = np.clip(pix @ H, 0.0, 1.0)[:, :3]
    return (result.reshape(h, w, 3) * 255).round().astype(np.uint8)


def color_transfer(scan: np.ndarray, restored: np.ndarray) -> np.ndarray:
    """
    Transfer scan's colours onto the restored image.
    Both are uint8 RGB arrays. Returns cropped (H-2*CROP, W, 3) uint8 result.
    """
    H = compute_color_matrix(scan, restored)
    return apply_color_matrix(restored[CROP:-CROP], H)
