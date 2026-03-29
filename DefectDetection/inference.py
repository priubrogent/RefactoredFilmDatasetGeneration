"""
Inference — run the trained defect detection model on pipeline output.

For each segment in a pipeline output directory the script reads the
aligned (scan, restored_1, restored_2) triplets, runs the UNet, and saves
the predicted mask alongside the existing heuristic mask so you can
compare them side-by-side.

Output files written into each segment directory:
    mask_model/          predicted binary masks  (uint8 PNG, 255=defect)
    mask_model_soft/     soft probability maps   (uint8 PNG, 0-255)

Usage:
    # Single movie
    python inference.py --checkpoint checkpoints/best_phase2.pt \\
                        --pipeline-dir /data/.../shining_new_test

    # Compare model vs heuristic on one segment
    python inference.py --checkpoint checkpoints/best_phase2.pt \\
                        --pipeline-dir /data/.../shining_new_test \\
                        --segment segment000000-001440 --visualise

    # Multiple movies
    python inference.py --checkpoint checkpoints/best_phase2.pt \\
                        --pipeline-dir /data/.../shining_new_test \\
                        --pipeline-dir /data/.../blade_runner_output
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model import UNet, build_model


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, cfg_model: dict, device: torch.device) -> UNet:
    model = build_model(cfg_model).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)  # handle both wrapped and raw state-dicts
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_frame(
    model:  UNet,
    scan:   np.ndarray,   # uint8 BGR
    r1:     np.ndarray,
    r2:     np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap:   int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model on a full-resolution frame using tiled inference.

    Returns:
        mask_bin  — uint8 H×W binary mask (0 or 255)
        mask_soft — uint8 H×W probability map (0-255)
    """
    h, w = scan.shape[:2]

    # Build 9-channel float32 image
    inp = np.concatenate([
        scan.astype(np.float32) / 255.0,
        r1.astype(np.float32)   / 255.0,
        r2.astype(np.float32)   / 255.0,
    ], axis=2)  # H×W×9

    # If frame is small enough, process in one shot
    if h <= tile_size and w <= tile_size:
        t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
        logit = model(t)[0, 0]
        prob  = torch.sigmoid(logit).cpu().numpy()
    else:
        prob = _tiled_predict(inp, model, device, tile_size, overlap)

    mask_soft = (prob * 255).clip(0, 255).astype(np.uint8)
    mask_bin  = (prob > 0.5).astype(np.uint8) * 255
    return mask_bin, mask_soft


def _tiled_predict(
    inp:       np.ndarray,   # H×W×9, float32
    model:     UNet,
    device:    torch.device,
    tile_size: int,
    overlap:   int,
) -> np.ndarray:
    """Sliding-window inference with linear blending in overlap regions."""
    h, w, _ = inp.shape
    prob_acc = np.zeros((h, w), dtype=np.float32)
    weight   = np.zeros((h, w), dtype=np.float32)

    stride = tile_size - overlap

    ys = list(range(0, h - tile_size + 1, stride))
    if ys[-1] + tile_size < h:
        ys.append(h - tile_size)

    xs = list(range(0, w - tile_size + 1, stride))
    if xs[-1] + tile_size < w:
        xs.append(w - tile_size)

    # Hann window for smooth blending
    win1d = np.hanning(tile_size).astype(np.float32)
    win2d = np.outer(win1d, win1d)

    for y in ys:
        for x in xs:
            tile = inp[y:y+tile_size, x:x+tile_size]
            t    = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            prob_acc[y:y+tile_size, x:x+tile_size] += p * win2d
            weight[y:y+tile_size, x:x+tile_size]   += win2d

    return prob_acc / (weight + 1e-6)


# ---------------------------------------------------------------------------
# Segment processing
# ---------------------------------------------------------------------------

def process_segment(
    seg_dir:  Path,
    model:    UNet,
    device:   torch.device,
    tile_size: int,
    overlap:   int,
) -> int:
    scan_dir = seg_dir / "scan"
    r1_dir   = seg_dir / "restored_1"
    r2_dir   = seg_dir / "restored_2"

    if not (scan_dir.exists() and r1_dir.exists() and r2_dir.exists()):
        return 0

    out_bin  = seg_dir / "mask_model"
    out_soft = seg_dir / "mask_model_soft"
    out_bin.mkdir(exist_ok=True)
    out_soft.mkdir(exist_ok=True)

    frames = sorted(scan_dir.glob("*.png"))
    for fpath in frames:
        scan = cv2.imread(str(fpath))
        r1   = cv2.imread(str(r1_dir / fpath.name))
        r2   = cv2.imread(str(r2_dir / fpath.name))

        if scan is None or r1 is None or r2 is None:
            continue

        mask_bin, mask_soft = predict_frame(model, scan, r1, r2, device, tile_size, overlap)
        cv2.imwrite(str(out_bin  / fpath.name), mask_bin)
        cv2.imwrite(str(out_soft / fpath.name), mask_soft)

    return len(frames)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_comparison(seg_dir: Path, n_frames: int = 3) -> None:
    """Save side-by-side comparison of heuristic vs model mask for n frames."""
    import matplotlib.pyplot as plt

    scan_dir      = seg_dir / "scan"
    heuristic_dir = seg_dir / "mask"
    model_dir     = seg_dir / "mask_model"
    soft_dir      = seg_dir / "mask_model_soft"
    vis_dir       = seg_dir / "debug" / "mask_model_comparison"
    vis_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(scan_dir.glob("*.png"))
    if not frames:
        return

    idxs   = np.linspace(0, len(frames) - 1, min(n_frames, len(frames)), dtype=int)
    for i in idxs:
        fname = frames[i].name
        scan  = cv2.cvtColor(cv2.imread(str(scan_dir / fname)), cv2.COLOR_BGR2RGB)

        h_mask = cv2.imread(str(heuristic_dir / fname), cv2.IMREAD_GRAYSCALE) if (heuristic_dir / fname).exists() else None
        m_mask = cv2.imread(str(model_dir      / fname), cv2.IMREAD_GRAYSCALE) if (model_dir      / fname).exists() else None
        m_soft = cv2.imread(str(soft_dir       / fname), cv2.IMREAD_GRAYSCALE) if (soft_dir        / fname).exists() else None

        ncols = 2 + (h_mask is not None) + (m_soft is not None)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
        fig.suptitle(f"Mask comparison — {fname}", fontsize=12)

        col = 0
        axes[col].imshow(scan); axes[col].set_title("scan"); axes[col].axis("off"); col += 1

        if h_mask is not None:
            ov = scan.copy(); ov[h_mask > 127] = [255, 80, 80]
            pct = 100 * (h_mask > 127).mean()
            axes[col].imshow(ov); axes[col].set_title(f"heuristic ({pct:.2f}%)"); axes[col].axis("off"); col += 1

        if m_mask is not None:
            ov = scan.copy(); ov[m_mask > 127] = [80, 255, 80]
            pct = 100 * (m_mask > 127).mean()
            axes[col].imshow(ov); axes[col].set_title(f"model ({pct:.2f}%)"); axes[col].axis("off"); col += 1

        if m_soft is not None:
            axes[col].imshow(m_soft, cmap="plasma"); axes[col].set_title("model soft"); axes[col].axis("off")

        out_path = vis_dir / f"comparison_{Path(fname).stem}.png"
        plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run defect detection model on pipeline output")
    parser.add_argument("--checkpoint",   required=True,  help="Path to model checkpoint (.pt)")
    parser.add_argument("--pipeline-dir", action="append", dest="pipeline_dirs", default=[],
                        help="Pipeline output dir (can repeat for multiple movies)")
    parser.add_argument("--config",       default="config.yaml",
                        help="Config file (used for model architecture)")
    parser.add_argument("--segment",      default=None,
                        help="Process only this segment subfolder name")
    parser.add_argument("--tile-size",    type=int, default=512)
    parser.add_argument("--overlap",      type=int, default=64)
    parser.add_argument("--visualise",    action="store_true",
                        help="Save side-by-side comparison images")
    parser.add_argument("--device",       default="auto")
    args = parser.parse_args()

    if not args.pipeline_dirs:
        parser.error("Provide at least one --pipeline-dir")

    # Load config for model architecture
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg_model = cfg.get("model", {})
    else:
        print(f"[warn] config not found at {args.config}, using defaults")
        cfg_model = {}

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    model = load_model(args.checkpoint, cfg_model, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    total_frames = 0
    for pdir in args.pipeline_dirs:
        base = Path(pdir)
        if not base.exists():
            print(f"[warn] not found: {pdir}")
            continue

        segments = (
            [base / args.segment]
            if args.segment
            else sorted(base.glob("segment*"))
        )

        for seg in segments:
            if not seg.is_dir():
                continue
            n = process_segment(seg, model, device, args.tile_size, args.overlap)
            print(f"  {seg.name}: {n} frames")
            total_frames += n

            if args.visualise:
                visualise_comparison(seg)

    print(f"\nDone. Total frames processed: {total_frames}")


if __name__ == "__main__":
    main()
