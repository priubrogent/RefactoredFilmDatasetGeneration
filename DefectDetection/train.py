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
from tqdm import tqdm

import wandb

from dataset import FilmTripletDataset, SyntheticFilmDataset, load_frame_as_patches
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
# Probe system — fixed images logged every epoch across both phases
#
# Five panels in wandb, each logged to a stable key so the image history
# slider shows the full training timeline:
#   probe/real_specific   — specific named frames (e.g. 400, 450, 1000),
#                           multiple crops per frame
#   probe/real_train      — random frames from the train portion
#   probe/real_val        — random frames from the val portion
#   probe/synthetic_train — samples from the synthetic train split
#   probe/synthetic_val   — samples from the synthetic test split
#
# Each row in a panel:  scan | r1 | pseudo/GT label | soft prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _probe_grid(
    inputs:    torch.Tensor,
    targets:   torch.Tensor,
    logits:    torch.Tensor,
    phase_tag: str,
    epoch:     int,
    title:     str,
) -> "wandb.Image":
    """One wandb image: every row is one sample (scan|r1|label|soft-pred)."""
    n = inputs.shape[0]
    rows = []
    for i in range(n):
        scan   = (inputs[i, :3].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        r1     = (inputs[i, 3:6].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        label  = (targets[i, 0].cpu().numpy() * 255).astype(np.uint8)
        prob   = torch.sigmoid(logits[i, 0]).cpu().numpy()

        label_ov   = _red_overlay(scan, label)
        heat       = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        pred_blend = cv2.addWeighted(scan, 0.5, heat, 0.5, 0)

        rows.append(np.concatenate([
            cv2.cvtColor(scan,       cv2.COLOR_BGR2RGB),
            cv2.cvtColor(r1,         cv2.COLOR_BGR2RGB),
            cv2.cvtColor(label_ov,   cv2.COLOR_BGR2RGB),
            cv2.cvtColor(pred_blend, cv2.COLOR_BGR2RGB),
        ], axis=1))

    grid = np.concatenate(rows, axis=0)
    caption = f"[{phase_tag} · ep {epoch}] {title}  —  scan | r1 | label | pred"
    return wandb.Image(grid, caption=caption)


@torch.no_grad()
def log_all_probes(
    model:     nn.Module,
    probes:    dict,          # {wandb_key: (inputs CPU, targets CPU)}
    device:    torch.device,
    phase_tag: str,
    epoch:     int,
) -> None:
    """Run the model on every probe panel and log all to wandb in one call."""
    model.eval()
    log_dict = {"epoch": epoch}
    for key, (inputs, targets) in probes.items():
        inputs  = inputs.to(device)
        targets = targets.to(device)
        logits  = model(inputs)
        # title = last part of the key (e.g. "real_specific")
        title = key.split("/")[-1].replace("_", " ")
        log_dict[key] = _probe_grid(inputs, targets, logits, phase_tag, epoch, title)
    wandb.log(log_dict)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer | None,
    bce_fn:     nn.BCEWithLogitsLoss,
    cfg_train:  dict,
    device:     torch.device,
    phase_tag:  str,
    global_step: list[int] | None = None,  # mutable [step] counter for batch logging
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

    mode = "train" if training else "val"
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=f"  {phase_tag} {mode}", leave=False, dynamic_ncols=True)
        for inputs, targets in pbar:
            # Dataset returns (B, N, 9, H, W) — flatten patch dim into batch dim
            if inputs.dim() == 5:
                B, N = inputs.shape[:2]
                inputs  = inputs.view(B * N, *inputs.shape[2:])
                targets = targets.view(B * N, *targets.shape[2:])
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

            pbar.set_postfix(loss=f"{totals['loss']/max(n,1):.4f}",
                             f1=f"{totals.get('f1',0)/max(n,1):.3f}")

            if training and global_step is not None:
                wandb.log({
                    f"{phase_tag}/batch/loss": loss.item(),
                    f"{phase_tag}/batch/bce":  loss_parts["bce"],
                    f"{phase_tag}/batch/dice": loss_parts["dice"],
                    "batch_step": global_step[0],
                })
                global_step[0] += 1

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
    probes:       dict | None = None,
) -> None:
    cfg_train = cfg["training"]
    cfg_wandb = cfg.get("wandb", {})

    epochs = cfg_train[f"phase{phase}_epochs"]
    lr     = cfg_train[f"phase{phase}_lr"]
    tag    = f"phase{phase}"

    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    global_step = [0]  # mutable counter passed into run_epoch for batch-level logging

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
            model, train_loader, optimizer, bce_fn, cfg_train, device, tag, global_step
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

        # Probes: all panels logged every epoch — stable keys give a
        # continuous timeline across both phases in wandb.
        if probes:
            log_all_probes(model, probes, device, tag, epoch + 1)

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
# Probe builder
# ---------------------------------------------------------------------------

def make_all_probes(cfg: dict) -> dict:
    """
    Build all probe panels once at startup. Returns:
        {wandb_key: (inputs_cpu, targets_cpu)}

    Panels built:
        probe/real_specific   — specific frame numbers from config
        probe/real_train      — evenly-spaced frames from the train half
        probe/real_val        — evenly-spaced frames from the val half
        probe/synthetic_train — samples from synthetic train split
        probe/synthetic_val   — samples from synthetic test split
    """
    cfg_data  = cfg["data"]
    cfg_pl    = cfg.get("pseudo_labels", {})
    cfg_train = cfg["training"]
    cfg_wandb = cfg.get("wandb", {})
    pl_cfg    = PseudoLabelConfig.from_dict(cfg_pl)
    patch_size = cfg_train["patch_size"]

    probes: dict = {}

    # ── 1. Specific named frames ───────────────────────────────────────────
    specific_nums  = cfg_wandb.get("probe_specific_frames", [400, 450, 1000])
    crops_per      = cfg_wandb.get("probe_crops_per_specific", 3)

    sp_inps, sp_tgts = [], []
    for frame_num in specific_nums:
        fname = f"{frame_num:06d}.png"
        found = False
        for base in cfg_data["pipeline_dirs"]:
            for seg in sorted(Path(base).glob("segment*")):
                scan_p = str(seg / "scan"       / fname)
                r1_p   = str(seg / "restored_1" / fname)
                r2_p   = str(seg / "restored_2" / fname)
                if Path(scan_p).exists() and Path(r1_p).exists() and Path(r2_p).exists():
                    result = load_frame_as_patches(
                        scan_p, r1_p, r2_p, pl_cfg, patch_size, n_crops=crops_per
                    )
                    if result is not None:
                        inps, tgts = result
                        sp_inps.append(inps)
                        sp_tgts.append(tgts)
                        print(f"  [probe] frame {frame_num}: {crops_per} crops from {seg.name}")
                        found = True
                    break
            if found:
                break
        if not found:
            print(f"  [probe] frame {frame_num}: not found in any segment")

    if sp_inps:
        probes["probe/real_specific"] = (
            torch.cat(sp_inps, dim=0),
            torch.cat(sp_tgts, dim=0),
        )

    # ── 2. Real train / val random frames ──────────────────────────────────
    n_train_probe = cfg_wandb.get("probe_n_real_train", 8)
    n_val_probe   = cfg_wandb.get("probe_n_real_val",   8)

    full_ds = FilmTripletDataset(
        pipeline_dirs=cfg_data["pipeline_dirs"],
        patch_size=patch_size,
        patches_per_frame=1,
        pl_cfg=pl_cfg,
        min_coverage=cfg_pl.get("min_coverage", 0.0005),
        max_coverage=cfg_pl.get("max_coverage", 0.20),
        augment=False,
        seed=1,
    )

    if len(full_ds) > 0:
        val_split = cfg_data.get("val_split", 0.5)
        n_total   = len(full_ds)
        n_tr      = max(1, int(n_total * (1 - val_split)))

        for indices, key, n_probe in [
            (np.linspace(0,    n_tr - 1,         min(n_train_probe, n_tr),             dtype=int), "probe/real_train", n_train_probe),
            (np.linspace(n_tr, n_total - 1,      min(n_val_probe,   n_total - n_tr),   dtype=int), "probe/real_val",   n_val_probe),
        ]:
            batch_inps, batch_tgts = [], []
            for i in indices:
                item = full_ds[int(i)]
                if item is None:
                    continue
                inp, tgt = item          # (1, 9, H, W), (1, 1, H, W)
                batch_inps.append(inp[0])
                batch_tgts.append(tgt[0])
            if batch_inps:
                probes[key] = (torch.stack(batch_inps), torch.stack(batch_tgts))
                print(f"  [probe] {key}: {len(batch_inps)} frames")

    # ── 3. Synthetic train / val ───────────────────────────────────────────
    syn_dir        = cfg_data.get("synthetic_dir")
    n_syn_train    = cfg_wandb.get("probe_n_synthetic_train", 6)
    n_syn_val      = cfg_wandb.get("probe_n_synthetic_val",   6)

    if syn_dir:
        for split, n_probe, key in [
            ("train", n_syn_train, "probe/synthetic_train"),
            ("test",  n_syn_val,   "probe/synthetic_val"),
        ]:
            syn_ds = SyntheticFilmDataset(
                synthetic_dir=syn_dir,
                split=split,
                patch_size=patch_size,
                patches_per_frame=1,
                label_threshold=cfg_train.get("phase1_label_threshold", 15.0),
                augment=False,
                seed=1,
            )
            if len(syn_ds) == 0:
                continue
            indices = np.linspace(0, len(syn_ds) - 1, min(n_probe, len(syn_ds)), dtype=int)
            s_inps, s_tgts = [], []
            for i in indices:
                inp, tgt = syn_ds[int(i)]   # (1, 9, H, W), (1, 1, H, W)
                s_inps.append(inp[0])
                s_tgts.append(tgt[0])
            if s_inps:
                probes[key] = (torch.stack(s_inps), torch.stack(s_tgts))
                print(f"  [probe] {key}: {len(s_inps)} samples")

    print(f"[probes] built {len(probes)} panels: {list(probes.keys())}")
    return probes


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

    # Build all probe panels once — logged every epoch across both phases
    # so wandb's image history shows the full training timeline.
    print("\n[probes] building probe panels...")
    probes = make_all_probes(cfg)

    # --- Phase 1: synthetic ---
    if run_phase1 and cfg["data"].get("synthetic_dir"):
        print("\n=== Phase 1: synthetic pre-training ===")
        train_loader, val_loader = make_synthetic_loaders(cfg)
        train_phase(1, model, train_loader, val_loader, cfg, device, start_epoch, probes)
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
        train_phase(2, model, train_loader, val_loader, cfg, device, probes=probes)

    wandb.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
