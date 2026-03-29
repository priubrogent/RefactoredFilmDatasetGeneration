"""
Dataset classes for defect-detection training.

Two sources are supported:

FilmTripletDataset — Real aligned triplets from pipeline output directories.
    Directory layout expected:
        <pipeline_dir>/segment*/
            scan/          *.png  — dirty scan frames
            restored_1/    *.png  — spatially aligned restoration 1
            restored_2/    *.png  — spatially aligned restoration 2
    Labels are generated on-the-fly as pseudo-labels via pseudo_labels.py.

SyntheticFilmDataset — Paired synthetic dataset (dirty input / clean target).
    Directory layout expected:
        <synthetic_dir>/train/ or test/
            input/   *.png
            target/  *.png
    A fake "R2" is created by lightly augmenting the target, so the model
    receives the same 9-channel format as the real data.
    The ground-truth label is |input - target| > threshold.
"""

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pseudo_labels import PseudoLabelConfig, compute_pseudo_label


# ---------------------------------------------------------------------------
# Real triplet dataset
# ---------------------------------------------------------------------------

class FilmTripletDataset(Dataset):
    """
    Loads (scan, r1, r2) patches from pipeline output directories and
    generates pseudo-labels on-the-fly.

    Args:
        pipeline_dirs:      List of pipeline output roots (one per movie).
        patch_size:         Square crop size in pixels.
        patches_per_frame:  How many random crops to draw per frame per epoch.
        pl_cfg:             Pseudo-label configuration.
        min_coverage:       Skip frames whose pseudo-label coverage < this.
        max_coverage:       Skip frames whose pseudo-label coverage > this.
        augment:            Apply random flips / rotations.
        seed:               RNG seed for frame list shuffling.
    """

    def __init__(
        self,
        pipeline_dirs:     list[str],
        patch_size:        int                        = 256,
        patches_per_frame: int                        = 8,
        pl_cfg:            Optional[PseudoLabelConfig] = None,
        min_coverage:      float                      = 0.0005,
        max_coverage:      float                      = 0.20,
        augment:           bool                       = True,
        seed:              int                        = 42,
    ):
        self.patch_size        = patch_size
        self.patches_per_frame = patches_per_frame
        self.pl_cfg            = pl_cfg or PseudoLabelConfig()
        self.min_coverage      = min_coverage
        self.max_coverage      = max_coverage
        self.augment           = augment

        self.triplets: list[tuple[str, str, str]] = []

        for base in pipeline_dirs:
            base_p = Path(base)
            if not base_p.exists():
                print(f"  [warn] pipeline dir not found, skipping: {base}")
                continue
            for seg in sorted(base_p.glob("segment*")):
                scan_d = seg / "scan"
                r1_d   = seg / "restored_1"
                r2_d   = seg / "restored_2"
                if not (scan_d.exists() and r1_d.exists() and r2_d.exists()):
                    continue
                for f in sorted(scan_d.glob("*.png")):
                    r1p = r1_d / f.name
                    r2p = r2_d / f.name
                    if r1p.exists() and r2p.exists():
                        self.triplets.append((str(f), str(r1p), str(r2p)))

        rng = random.Random(seed)
        rng.shuffle(self.triplets)
        print(
            f"[FilmTripletDataset] {len(self.triplets)} triplets "
            f"from {len(pipeline_dirs)} dir(s)"
        )

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read each frame ONCE and return all patches_per_frame crops together.
        Shape: (N, 9, H, W) and (N, 1, H, W).
        run_epoch flattens these back to (B*N, ...) after collation.
        """
        scan_p, r1_p, r2_p = self.triplets[idx]

        scan = cv2.imread(scan_p)
        r1   = cv2.imread(r1_p)
        r2   = cv2.imread(r2_p)

        if scan is None or r1 is None or r2 is None:
            return self._zeros()

        mask, coverage = compute_pseudo_label(scan, r1, r2, self.pl_cfg)

        if not (self.min_coverage <= coverage <= self.max_coverage):
            return self._zeros()

        inp_patches, mask_patches = [], []
        for _ in range(self.patches_per_frame):
            s, r1_, r2_, m = _random_crop(scan, r1, r2, mask, self.patch_size)
            if self.augment:
                s, r1_, r2_, m = _geometric_augment(s, r1_, r2_, m)
            inp_t, mask_t = _to_tensors(s, r1_, r2_, m)
            inp_patches.append(inp_t)
            mask_patches.append(mask_t)

        return torch.stack(inp_patches), torch.stack(mask_patches)  # (N,9,H,W), (N,1,H,W)

    def _zeros(self) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.patch_size
        n = self.patches_per_frame
        return torch.zeros(n, 9, p, p), torch.zeros(n, 1, p, p)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticFilmDataset(Dataset):
    """
    Paired synthetic (dirty input / clean target) dataset.

    A fake R2 is created by lightly perturbing the target so the model
    sees the same 9-channel [scan | r1 | r2] format as the real data.
    Ground-truth mask = |input - target| > label_threshold per pixel.

    Args:
        synthetic_dir:    Root dir containing train/ and test/ subdirs.
        split:            "train" or "test".
        patch_size:       Square crop size.
        patches_per_frame: Random crops per image per epoch.
        label_threshold:  Pixel diff (0-255 scale) to binarise GT label.
        augment:          Apply geometric augmentation.
        seed:             RNG seed.
    """

    def __init__(
        self,
        synthetic_dir:     str,
        split:             str   = "train",
        patch_size:        int   = 256,
        patches_per_frame: int   = 4,
        label_threshold:   float = 15.0,
        augment:           bool  = True,
        seed:              int   = 42,
    ):
        self.patch_size        = patch_size
        self.patches_per_frame = patches_per_frame
        self.label_threshold   = label_threshold
        self.augment           = augment

        inp_dir = Path(synthetic_dir) / split / "input"
        tgt_dir = Path(synthetic_dir) / split / "target"

        self.pairs: list[tuple[str, str]] = []
        if inp_dir.exists() and tgt_dir.exists():
            for f in sorted(inp_dir.glob("*.png")):
                tp = tgt_dir / f.name
                if tp.exists():
                    self.pairs.append((str(f), str(tp)))

        rng = random.Random(seed)
        rng.shuffle(self.pairs)
        print(f"[SyntheticFilmDataset] {len(self.pairs)} pairs ({split})")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read each image ONCE, return all patches_per_frame crops stacked."""
        inp_p, tgt_p = self.pairs[idx]

        inp = cv2.imread(inp_p)
        tgt = cv2.imread(tgt_p)

        if inp is None or tgt is None:
            p, n = self.patch_size, self.patches_per_frame
            return torch.zeros(n, 9, p, p), torch.zeros(n, 1, p, p)

        diff  = np.linalg.norm(inp.astype(np.float32) - tgt.astype(np.float32), axis=2)
        label = (diff > self.label_threshold).astype(np.uint8) * 255
        r2    = _fake_r2(tgt)

        inp_patches, mask_patches = [], []
        for _ in range(self.patches_per_frame):
            i, t, r, l = _random_crop(inp, tgt, r2, label, self.patch_size)
            if self.augment:
                i, t, r, l = _geometric_augment(i, t, r, l)
            inp_t, mask_t = _to_tensors(i, t, r, l)
            inp_patches.append(inp_t)
            mask_patches.append(mask_t)

        return torch.stack(inp_patches), torch.stack(mask_patches)  # (N,9,H,W), (N,1,H,W)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_crop(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pad if necessary then take a random aligned crop of all four arrays."""
    h, w = a.shape[:2]
    p = patch_size

    if h < p or w < p:
        pad_h = max(0, p - h)
        pad_w = max(0, p - w)
        border = cv2.BORDER_REFLECT
        a    = cv2.copyMakeBorder(a,    0, pad_h, 0, pad_w, border)
        b    = cv2.copyMakeBorder(b,    0, pad_h, 0, pad_w, border)
        c    = cv2.copyMakeBorder(c,    0, pad_h, 0, pad_w, border)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, border)
        h, w = a.shape[:2]

    y = random.randint(0, h - p)
    x = random.randint(0, w - p)

    return (
        a[y:y+p, x:x+p],
        b[y:y+p, x:x+p],
        c[y:y+p, x:x+p],
        mask[y:y+p, x:x+p],
    )


def _geometric_augment(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identical random geometric augmentation applied to all four arrays."""
    # Horizontal flip
    if random.random() > 0.5:
        a    = cv2.flip(a,    1)
        b    = cv2.flip(b,    1)
        c    = cv2.flip(c,    1)
        mask = cv2.flip(mask, 1)
    # Vertical flip
    if random.random() > 0.5:
        a    = cv2.flip(a,    0)
        b    = cv2.flip(b,    0)
        c    = cv2.flip(c,    0)
        mask = cv2.flip(mask, 0)
    # 90° multiples
    if random.random() > 0.5:
        k    = random.choice([1, 2, 3])
        a    = np.rot90(a,    k).copy()
        b    = np.rot90(b,    k).copy()
        c    = np.rot90(c,    k).copy()
        mask = np.rot90(mask, k).copy()
    return a, b, c, mask


def _to_tensors(
    scan: np.ndarray,
    r1:   np.ndarray,
    r2:   np.ndarray,
    mask: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack BGR images into a 9-channel float tensor and binarise the mask."""
    inp = np.concatenate([
        scan.astype(np.float32) / 255.0,
        r1.astype(np.float32)   / 255.0,
        r2.astype(np.float32)   / 255.0,
    ], axis=2)  # H×W×9

    inp_t  = torch.from_numpy(inp).permute(2, 0, 1)                        # 9×H×W
    mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)  # 1×H×W
    return inp_t, mask_t


def load_frame_as_patches(
    scan_p:     str,
    r1_p:       str,
    r2_p:       str,
    pl_cfg:     "PseudoLabelConfig",
    patch_size: int,
    n_crops:    int = 3,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Load a specific aligned triplet and return n_crops evenly-spaced patches.

    Crops are taken along the diagonal (top-left → bottom-right) so each crop
    shows a different region of the frame.

    Returns:
        (inputs, targets): tensors of shape (n_crops, 9, P, P) and (n_crops, 1, P, P)
        None if any image fails to load.
    """
    scan = cv2.imread(scan_p)
    r1   = cv2.imread(r1_p)
    r2   = cv2.imread(r2_p)
    if scan is None or r1 is None or r2 is None:
        return None

    mask, _ = compute_pseudo_label(scan, r1, r2, pl_cfg)

    h, w = scan.shape[:2]
    p    = patch_size
    ys   = np.linspace(0, max(0, h - p), n_crops, dtype=int)
    xs   = np.linspace(0, max(0, w - p), n_crops, dtype=int)

    inps, masks = [], []
    for y, x in zip(ys, xs):
        s  = scan[y:y+p, x:x+p]
        r1_ = r1[y:y+p, x:x+p]
        r2_ = r2[y:y+p, x:x+p]
        m  = mask[y:y+p, x:x+p]
        # Pad if frame is smaller than patch size
        if s.shape[0] < p or s.shape[1] < p:
            ph = max(0, p - s.shape[0])
            pw = max(0, p - s.shape[1])
            bd = cv2.BORDER_REFLECT
            s   = cv2.copyMakeBorder(s,   0, ph, 0, pw, bd)
            r1_ = cv2.copyMakeBorder(r1_, 0, ph, 0, pw, bd)
            r2_ = cv2.copyMakeBorder(r2_, 0, ph, 0, pw, bd)
            m   = cv2.copyMakeBorder(m,   0, ph, 0, pw, bd)
        inp_t, mask_t = _to_tensors(s, r1_, r2_, m)
        inps.append(inp_t)
        masks.append(mask_t)

    return torch.stack(inps), torch.stack(masks)


def _fake_r2(tgt: np.ndarray) -> np.ndarray:
    """Lightly perturb the clean target to use as a fake second restoration."""
    r2 = tgt.astype(np.float32)
    r2 = r2 * (1.0 + np.random.uniform(-0.04, 0.04))      # brightness jitter
    r2 = r2 + np.random.normal(0, 1.5, r2.shape)           # tiny noise
    return np.clip(r2, 0, 255).astype(np.uint8)
