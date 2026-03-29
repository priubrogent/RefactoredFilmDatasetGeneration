"""
Training script for the film defect detection UNet.

Two-phase training:
  Phase 1 — supervised on the synthetic FilmRestoration dataset (clean labels).
             Skipped if synthetic_dir is null in config.
  Phase 2 — self-supervised on real pipeline triplets (pseudo-labels).

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --phase 2          # skip phase 1
    python train.py --config config.yaml --resume checkpoint.pt
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset

import wandb

from dataset import FilmTripletDataset, SyntheticFilmDataset
from model import UNet, build_model, count_parameters
from pseudo_labels import PseudoLabelConfig


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    flat_p = probs.view(-1)
    flat_t = targets.view(-1)
    intersection = (flat_p * flat_t).sum()
    return 1.0 - (2.0 * intersection + eps) / (flat_p.sum() + flat_t.sum() + eps)


def tv_loss(logits: torch.Tensor) -> torch.Tensor:
    """Total variation — encourages spatially smooth predictions."""
    probs = torch.sigmoid(logits)
    dh = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).mean()
    dw = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1]).mean()
    return dh + dw


def combined_loss(
    logits:     torch.Tensor,
    targets:    torch.Tensor,
    bce_fn:     nn.BCEWithLogitsLoss,
    bce_w:      float,
    dice_w:     float,
    tv_w:       float,
) -> tuple[torch.Tensor, dict]:
    bce  = bce_fn(logits, targets)
    dice = dice_loss(logits, targets)
    tv   = tv_loss(logits)
    total = bce_w * bce + dice_w * dice + tv_w * tv
    return total, {"bce": bce.item(), "dice": dice.item(), "tv": tv.item()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    thresh:  float = 0.5,
) -> dict:
    preds = (torch.sigmoid(logits) > thresh).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)
    coverage  = preds.mean().item()

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "iou":       iou,
        "pred_coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Image grid for wandb logging
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_log_grid(
    inputs:  torch.Tensor,   # B×9×H×W
    targets: torch.Tensor,   # B×1×H×W
    logits:  torch.Tensor,   # B×1×H×W
    n:       int = 4,
) -> "wandb.Image":
    import cv2

    n = min(n, inputs.shape[0])
    rows = []
    for i in range(n):
        scan = (inputs[i, :3].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        r1   = (inputs[i, 3:6].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gt   = (targets[i, 0].cpu().numpy() * 255).astype(np.uint8)
        pred = (torch.sigmoid(logits[i, 0]).cpu().numpy() * 255).astype(np.uint8)

        # Convert mask → colour overlays on scan
        gt_overlay   = _red_overlay(scan, gt)
        pred_overlay = _red_overlay(scan, pred)

        gt_rgb   = cv2.cvtColor(gt_overlay,   cv2.COLOR_BGR2RGB)
        pred_rgb = cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB)
        scan_rgb = cv2.cvtColor(scan,         cv2.COLOR_BGR2RGB)
        r1_rgb   = cv2.cvtColor(r1,           cv2.COLOR_BGR2RGB)

        rows.append(np.concatenate([scan_rgb, r1_rgb, gt_rgb, pred_rgb], axis=1))

    grid = np.concatenate(rows, axis=0)
    return wandb.Image(grid, caption="scan | r1 | pseudo/GT mask | prediction")


def _red_overlay(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay mask (0-255 single channel) as red on bgr image."""
    out = bgr.copy()
    m   = mask > 127
    out[m] = (out[m] * (1 - alpha) + np.array([0, 0, 255]) * alpha).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Real-frame probe — fixed batch logged every epoch in both phases
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_probe_grid(
    inputs:    torch.Tensor,   # B×9×H×W  — real frames
    targets:   torch.Tensor,   # B×1×H×W  — pseudo-labels
    logits:    torch.Tensor,   # B×1×H×W  — current model output
    phase_tag: str,
    epoch:     int,
    n:         int = 4,
) -> "wandb.Image":
    """
    Build a wandb image grid for the fixed real-frame probe.

    Each row (one sample):
        scan  |  r1  |  pseudo-label overlay  |  prediction (soft heatmap)

    Because the same probe frames are logged every epoch across both phases,
    wandb's image history lets you scrub through time and see exactly how
    the model predictions on real film change as training progresses from
    synthetic (phase 1) to real (phase 2).
    """
    import cv2

    n = min(n, inputs.shape[0])
    rows = []
    for i in range(n):
        scan = (inputs[i, :3].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        r1   = (inputs[i, 3:6].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pseudo = (targets[i, 0].cpu().numpy() * 255).astype(np.uint8)
        prob   = torch.sigmoid(logits[i, 0]).cpu().numpy()          # float [0,1]

        pseudo_overlay = _red_overlay(scan, pseudo)
        scan_rgb   = cv2.cvtColor(scan,           cv2.COLOR_BGR2RGB)
        r1_rgb     = cv2.cvtColor(r1,             cv2.COLOR_BGR2RGB)
        pseudo_rgb = cv2.cvtColor(pseudo_overlay, cv2.COLOR_BGR2RGB)

        # Soft prediction as plasma heatmap blended over scan
        heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        pred_blend = cv2.addWeighted(scan, 0.5, heat, 0.5, 0)
        pred_rgb   = cv2.cvtColor(pred_blend, cv2.COLOR_BGR2RGB)

        rows.append(np.concatenate([scan_rgb, r1_rgb, pseudo_rgb, pred_rgb], axis=1))

    grid = np.concatenate(rows, axis=0)
    caption = (
        f"[{phase_tag} · epoch {epoch}]  "
        "scan  |  r1  |  pseudo-label  |  prediction (soft)"
    )
    return wandb.Image(grid, caption=caption)


@torch.no_grad()
def log_probe_to_wandb(
    model:      nn.Module,
    probe:      tuple[torch.Tensor, torch.Tensor],  # (inputs, targets) on CPU
    device:     torch.device,
    phase_tag:  str,
    epoch:      int,
    num_log:    int,
) -> None:
    """Run model on the fixed real-frame probe and log the grid to wandb."""
    inputs, targets = probe
    inputs  = inputs.to(device)
    targets = targets.to(device)

    model.eval()
    logits = model(inputs)

    grid = make_probe_grid(inputs, targets, logits, phase_tag, epoch, n=num_log)
    wandb.log({"probe/real_frames": grid, "epoch": epoch})


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer | None,
    bce_fn:    nn.BCEWithLogitsLoss,
    cfg_train: dict,
    device:    torch.device,
    phase_tag: str,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    training = optimizer is not None
    model.train(training)

    bce_w  = cfg_train.get("bce_weight", 1.0)
    dice_w = cfg_train.get("dice_weight", 1.0)
    tv_w   = cfg_train.get("tv_weight", 0.05)

    totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "tv": 0.0,
              "precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    n = 0

    last_inp, last_tgt, last_log = None, None, None

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for inputs, targets in loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss, loss_parts = combined_loss(logits, targets, bce_fn, bce_w, dice_w, tv_w)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            metrics = compute_metrics(logits, targets)
            bs = inputs.shape[0]
            totals["loss"] += loss.item() * bs
            for k, v in {**loss_parts, **metrics}.items():
                totals[k] = totals.get(k, 0.0) + v * bs
            n += bs

            last_inp = inputs.detach()[:4]
            last_tgt = targets.detach()[:4]
            last_log = logits.detach()[:4]

    for k in totals:
        totals[k] /= max(n, 1)

    return totals, last_inp, last_tgt, last_log


# ---------------------------------------------------------------------------
# Training phase
# ---------------------------------------------------------------------------

def train_phase(
    phase:        int,
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          dict,
    device:       torch.device,
    start_epoch:  int = 0,
    probe:        tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    cfg_train = cfg["training"]
    cfg_wandb = cfg.get("wandb", {})

    epochs = cfg_train[f"phase{phase}_epochs"]
    lr     = cfg_train[f"phase{phase}_lr"]
    tag    = f"phase{phase}"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pos_weight = torch.tensor([cfg_train.get("pos_weight", 15.0)], device=device)
    bce_fn     = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ckpt_dir   = Path(cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every = cfg["checkpoint"].get("save_every", 5)
    log_images_every = cfg_wandb.get("log_images_every", 5)
    num_log    = cfg_wandb.get("num_log_samples", 4)

    best_val_loss = math.inf

    for epoch in range(start_epoch, start_epoch + epochs):
        train_metrics, *_ = run_epoch(
            model, train_loader, optimizer, bce_fn, cfg_train, device, tag
        )
        val_metrics, val_inp, val_tgt, val_log = run_epoch(
            model, val_loader, None, bce_fn, cfg_train, device, tag
        )
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"[{tag}] epoch {epoch+1:3d}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_f1={val_metrics['f1']:.3f}  "
            f"val_iou={val_metrics['iou']:.3f}  "
            f"lr={lr_now:.2e}"
        )

        log_dict = {f"{tag}/train/{k}": v for k, v in train_metrics.items()}
        log_dict.update({f"{tag}/val/{k}": v for k, v in val_metrics.items()})
        log_dict[f"{tag}/lr"] = lr_now
        log_dict["epoch"] = epoch + 1

        if val_inp is not None and (epoch + 1) % log_images_every == 0:
            log_dict[f"{tag}/val_samples"] = make_log_grid(
                val_inp, val_tgt, val_log, n=num_log
            )

        wandb.log(log_dict)

        # Probe: fixed real frames logged every epoch so both phases are
        # comparable in the same wandb panel ("probe/real_frames").
        if probe is not None:
            log_probe_to_wandb(model, probe, device, tag, epoch + 1, num_log)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {"epoch": epoch + 1, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "val_loss": best_val_loss},
                ckpt_dir / f"best_{tag}.pt",
            )

        if (epoch + 1) % save_every == 0:
            torch.save(
                {"epoch": epoch + 1, "model": model.state_dict()},
                ckpt_dir / f"{tag}_epoch{epoch+1:03d}.pt",
            )

    print(f"[{tag}] done. best val loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def make_real_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    cfg_data   = cfg["data"]
    cfg_pl     = cfg.get("pseudo_labels", {})
    cfg_train  = cfg["training"]

    pl_cfg = PseudoLabelConfig.from_dict(cfg_pl)

    full_ds = FilmTripletDataset(
        pipeline_dirs=cfg_data["pipeline_dirs"],
        patch_size=cfg_train["patch_size"],
        patches_per_frame=cfg_train["patches_per_frame"],
        pl_cfg=pl_cfg,
        min_coverage=cfg_pl.get("min_coverage", 0.0005),
        max_coverage=cfg_pl.get("max_coverage", 0.20),
        augment=cfg_train.get("augment", True),
        seed=cfg_data.get("seed", 42),
    )

    n_total = len(full_ds)
    n_val   = max(1, int(n_total * cfg_data.get("val_split", 0.10)))
    n_train = n_total - n_val

    # Deterministic split — first half train, second half val (preserves
    # temporal ordering within each movie so val is from different scenes)
    train_ds = Subset(full_ds, list(range(n_train)))
    val_ds   = Subset(full_ds, list(range(n_train, n_total)))

    bs = cfg_train["phase2_batch_size"]
    nw = cfg_train.get("num_workers", 4)

    ctx = "spawn" if nw > 0 else None
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              multiprocessing_context=ctx)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              multiprocessing_context=ctx)
    print(f"[real data] train={n_train}  val={n_val}")
    return train_loader, val_loader


def make_synthetic_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    cfg_data  = cfg["data"]
    cfg_train = cfg["training"]

    train_ds = SyntheticFilmDataset(
        synthetic_dir=cfg_data["synthetic_dir"],
        split="train",
        patch_size=cfg_train["patch_size"],
        patches_per_frame=cfg_train.get("patches_per_frame", 4),
        label_threshold=cfg_train.get("phase1_label_threshold", 15.0),
        augment=cfg_train.get("augment", True),
        seed=cfg_data.get("seed", 42),
    )
    val_ds = SyntheticFilmDataset(
        synthetic_dir=cfg_data["synthetic_dir"],
        split="test",
        patch_size=cfg_train["patch_size"],
        patches_per_frame=2,
        label_threshold=cfg_train.get("phase1_label_threshold", 15.0),
        augment=False,
        seed=cfg_data.get("seed", 42),
    )

    bs = cfg_train["phase1_batch_size"]
    nw = cfg_train.get("num_workers", 4)

    ctx = "spawn" if nw > 0 else None
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              multiprocessing_context=ctx)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              multiprocessing_context=ctx)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Probe batch builder
# ---------------------------------------------------------------------------

def make_probe_batch(
    cfg:    dict,
    n:      int = 8,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Load N fixed real triplets (no augmentation, deterministic) and return
    them as CPU tensors ready to be passed into log_probe_to_wandb().

    Samples are drawn evenly from across all available triplets so the probe
    covers different scenes / movies.  Returns None if no real data is found.
    """
    cfg_data  = cfg["data"]
    cfg_pl    = cfg.get("pseudo_labels", {})
    cfg_train = cfg["training"]
    pl_cfg    = PseudoLabelConfig.from_dict(cfg_pl)

    # Build a no-augment dataset just to get the triplet list
    ds = FilmTripletDataset(
        pipeline_dirs=cfg_data["pipeline_dirs"],
        patch_size=cfg_train["patch_size"],
        patches_per_frame=1,          # one crop per frame
        pl_cfg=pl_cfg,
        min_coverage=cfg_pl.get("min_coverage", 0.0005),
        max_coverage=cfg_pl.get("max_coverage", 0.20),
        augment=False,
        seed=0,                       # fixed seed → same frames every run
    )

    if len(ds) == 0:
        print("[probe] no real data found — probe disabled")
        return None

    # Evenly spaced indices across the full dataset
    indices = np.linspace(0, len(ds) - 1, n, dtype=int).tolist()

    inputs_list, targets_list = [], []
    for idx in indices:
        inp, tgt = ds[idx]
        inputs_list.append(inp)
        targets_list.append(tgt)

    probe = (torch.stack(inputs_list), torch.stack(targets_list))
    print(f"[probe] loaded {len(inputs_list)} fixed real frames for wandb probe")
    return probe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--phase",   type=int, default=0,
                        help="1 = synthetic only, 2 = real only, 0 = both (default)")
    parser.add_argument("--resume",  default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device",  default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Seed
    seed = cfg["data"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Model
    model = build_model(cfg["model"]).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    # Wandb
    cfg_wandb = cfg.get("wandb", {})
    wandb.init(
        project=cfg_wandb.get("project", "film-defect-detection"),
        entity=cfg_wandb.get("entity") or None,
        config=cfg,
        resume="allow",
        id=wandb.util.generate_id() if not args.resume else None,
    )
    wandb.watch(model, log="gradients", log_freq=100)

    run_phase1 = args.phase in (0, 1)
    run_phase2 = args.phase in (0, 2)

    # Build the fixed real-frame probe once — used in both phases so that
    # "probe/real_frames" in wandb shows a continuous timeline from
    # synthetic pre-training all the way through real fine-tuning.
    probe = make_probe_batch(cfg, n=cfg.get("wandb", {}).get("num_log_samples", 4))

    # --- Phase 1: synthetic ---
    if run_phase1 and cfg["data"].get("synthetic_dir"):
        print("\n=== Phase 1: synthetic pre-training ===")
        train_loader, val_loader = make_synthetic_loaders(cfg)
        train_phase(1, model, train_loader, val_loader, cfg, device, start_epoch, probe)
        # Save after phase 1 for phase 2 warm-start
        torch.save(
            {"epoch": cfg["training"]["phase1_epochs"], "model": model.state_dict()},
            Path(cfg["checkpoint"]["dir"]) / "phase1_final.pt",
        )
    elif run_phase1:
        print("[phase 1] synthetic_dir not set — skipping")

    # --- Phase 2: real data ---
    if run_phase2:
        print("\n=== Phase 2: real-data self-supervised fine-tuning ===")
        train_loader, val_loader = make_real_loaders(cfg)
        train_phase(2, model, train_loader, val_loader, cfg, device, probe=probe)

    wandb.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
