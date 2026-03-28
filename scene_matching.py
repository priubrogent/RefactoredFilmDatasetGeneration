"""
Scene-change detection and temporal offset estimation between two video versions.

Used to automatically find how many frames one video version is ahead/behind
another (the offset), even when the offset varies across the film.

Typical usage:
    python scene_matching.py --scan PATH_TO_SCAN --restored PATH_TO_RESTORED
"""

import argparse
import hashlib
import os
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CNN feature extractor
# ---------------------------------------------------------------------------

class CNNFeatureExtractor:
    """
    MobileNetV2-based feature extractor for scene matching.

    Produces 1280-dim L2-normalised feature vectors.  These are far more
    discriminative than histogram features because the network captures
    semantic content (objects, textures, spatial layout) rather than just
    colour statistics — so two different film scenes with similar brightness
    distributions will no longer be confused.

    Requires: torch + torchvision (already in requirements.txt).
    Falls back gracefully to None if not installed.
    """

    def __init__(self, device: str = "cpu"):
        import torch
        import torchvision.models as models
        import torchvision.transforms as T

        self._torch = torch
        self.device = device

        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        # Drop the classifier head — use the 1280-dim global-average-pooled features
        self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        self.backbone.eval().to(device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a 1280-dim L2-normalised feature vector for a BGR frame."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            feat = self.backbone(tensor)          # (1, 1280, 7, 7)
            feat = feat.mean(dim=[2, 3])          # (1, 1280) — global avg pool
        feat = feat.squeeze().cpu().numpy()       # (1280,)
        return feat / (np.linalg.norm(feat) + 1e-8)


def _histogram_features(frame_bgr: np.ndarray, resize: tuple = (160, 90)) -> np.ndarray:
    """39-dim histogram feature (fast fallback when torch is not available)."""
    frame_r = cv2.resize(frame_bgr, resize, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame_r, cv2.COLOR_BGR2HSV)

    hist_gray = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    hist_gray /= hist_gray.sum() + 1e-8
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    hist_h /= hist_h.sum() + 1e-8
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_s /= hist_s.sum() + 1e-8

    h, w = gray.shape
    hh, hw = h // 2, w // 2
    grid_means = [gray[:hh, :hw].mean(), gray[:hh, hw:].mean(),
                  gray[hh:, :hw].mean(), gray[hh:, hw:].mean()]
    edges = cv2.Canny(gray, 50, 150)
    return np.concatenate([hist_gray, hist_h, hist_s, grid_means,
                           [gray.mean(), gray.std(), edges.mean() / 255.0]])


def extract_features_for_frames(
    path: str,
    frame_indices: np.ndarray,
    extractor: CNNFeatureExtractor | None = None,
) -> np.ndarray:
    """
    Extract feature vectors for a specific set of frame indices.

    Much faster than scanning the whole video when you only need a handful
    of scene-change frames.  Uses CNN features if extractor is provided,
    otherwise falls back to 39-dim histogram features.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open '{path}'")

    feat_dim = 1280 if extractor is not None else 39
    feats = []

    for idx in tqdm(frame_indices, desc=f"Features '{Path(path).name}'"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            feats.append(np.zeros(feat_dim))
            continue
        if extractor is not None:
            feats.append(extractor.extract(frame))
        else:
            feats.append(_histogram_features(frame))

    cap.release()
    return np.stack(feats)


# ---------------------------------------------------------------------------
# Scene change detection
# ---------------------------------------------------------------------------

def detect_scene_changes(
    path: str,
    max_frames: int = 1800,
    resize: tuple = (160, 90),
    threshold: float = 30.0,
    max_scenes: int = 20,
    frames_after_cut: int = 5,
    min_scene_duration: int = 10,
    fade_window: int = 5,
    brightness_threshold: float = 20.0,
) -> tuple:
    """
    Detect hard cuts in a video using histogram difference with fade filtering.

    Args:
        path:                Path to the video file.
        max_frames:          Maximum number of frames to scan.
        resize:              (width, height) to resize frames for fast processing.
        threshold:           Histogram diff threshold to flag a cut (higher → fewer cuts).
        max_scenes:          Cap on how many scene changes to return.
        frames_after_cut:    Use this frame after the cut as the representative image.
        min_scene_duration:  Minimum frames between two cuts (filters rapid fades).
        fade_window:         Window size for gradual-transition detection.
        brightness_threshold: Frames below this mean brightness are treated as black.

    Returns:
        (scene_frame_indices, scene_images, diff_scores, all_frame_indices)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video '{path}'")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, max_frames)

    prev_hist = None
    diff_scores, brightness_scores, all_frames, frame_indices = [], [], [], []

    with tqdm(total=frames_to_process, desc=f"Detecting cuts in '{path}'") as pbar:
        frame_idx = 0
        while len(all_frames) < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, resize, interpolation=cv2.INTER_AREA)

            hist = cv2.calcHist([gray_small], [0], None, [64], [0, 256]).flatten()
            hist /= hist.sum() + 1e-8

            brightness_scores.append(gray_small.mean())

            if prev_hist is not None:
                diff_scores.append(np.sum(np.abs(hist - prev_hist)) * 100)
            else:
                diff_scores.append(0.0)

            prev_hist = hist.copy()
            all_frames.append(frame.copy())
            frame_indices.append(frame_idx)
            frame_idx += 1
            pbar.update(1)

    cap.release()

    if not all_frames:
        raise ValueError(f"No frames read from '{path}'")

    diff_scores = np.array(diff_scores)
    brightness_scores = np.array(brightness_scores)
    frame_indices = np.array(frame_indices)

    potential_cuts = np.where(diff_scores > threshold)[0]

    if len(potential_cuts) == 0:
        print(f"  No cuts detected with threshold={threshold}. Using first frame.")
        scene_change_idx = np.array([frames_after_cut])
    else:
        filtered_cuts = []
        for cut_idx in potential_cuts:
            if filtered_cuts and (cut_idx - filtered_cuts[-1]) < min_scene_duration:
                continue

            check_start = max(0, cut_idx - fade_window)
            check_end = min(len(brightness_scores), cut_idx + fade_window + 1)
            is_fade = np.any(brightness_scores[check_start:check_end] < brightness_threshold)

            if not is_fade and cut_idx >= fade_window and cut_idx < len(diff_scores) - fade_window:
                window = diff_scores[max(0, cut_idx - fade_window): min(len(diff_scores), cut_idx + fade_window + 1)]
                if np.sum(window > threshold * 0.6) > fade_window:
                    is_fade = True

            if not is_fade:
                filtered_cuts.append(cut_idx)

        scene_change_idx = np.array(filtered_cuts) if filtered_cuts else np.array([frames_after_cut])
        print(f"  {len(potential_cuts)} raw peaks → {len(scene_change_idx)} hard cuts")

    if len(scene_change_idx) > max_scenes:
        strongest = np.argsort(diff_scores[scene_change_idx])[-max_scenes:]
        scene_change_idx = np.sort(scene_change_idx[strongest])

    print(f"  Final: {len(scene_change_idx)} cuts (threshold={threshold:.1f})")

    representative_indices, representative_images = [], []
    for cut_idx in scene_change_idx:
        repr_idx = min(cut_idx + frames_after_cut, len(all_frames) - 1)
        representative_indices.append(frame_indices[repr_idx])
        representative_images.append(all_frames[repr_idx])

    return np.array(representative_indices), representative_images, diff_scores, frame_indices


# ---------------------------------------------------------------------------
# Per-frame visual signatures
# ---------------------------------------------------------------------------

def video_signatures(path: str, max_frames: int = 1800, resize: tuple = (160, 90)) -> np.ndarray:
    """
    Extract a compact visual feature vector for each frame.

    Features per frame (39-dim):
        - 16-bin grayscale histogram
        - 8-bin hue histogram (HSV)
        - 8-bin saturation histogram (HSV)
        - 4 quadrant mean brightness
        - [global mean, std, edge density]

    Returns:
        np.ndarray of shape (N, 39).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video '{path}'")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, max_frames)

    sigs = []
    with tqdm(total=frames_to_process, desc=f"Signatures '{path}'") as pbar:
        while len(sigs) < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            frame_r = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame_r, cv2.COLOR_BGR2HSV)

            hist_gray = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
            hist_gray /= hist_gray.sum() + 1e-8

            hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
            hist_h /= hist_h.sum() + 1e-8

            hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
            hist_s /= hist_s.sum() + 1e-8

            h, w = gray.shape
            hh, hw = h // 2, w // 2
            grid_means = [
                gray[:hh, :hw].mean(), gray[:hh, hw:].mean(),
                gray[hh:, :hw].mean(), gray[hh:, hw:].mean(),
            ]

            edges = cv2.Canny(gray, 50, 150)
            global_stats = [gray.mean(), gray.std(), edges.mean() / 255.0]

            sigs.append(np.concatenate([hist_gray, hist_h, hist_s, grid_means, global_stats]))
            pbar.update(1)

    cap.release()
    if not sigs:
        raise ValueError(f"No frames read from '{path}'")
    return np.stack(sigs)


# ---------------------------------------------------------------------------
# Scene matching and offset estimation
# ---------------------------------------------------------------------------

def compute_scene_similarity(sigA: np.ndarray, sigB: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between every pair of scene signatures.

    Returns:
        similarity_matrix of shape (len(sigA), len(sigB)).
    """
    sigA_norm = (sigA - sigA.mean(1, keepdims=True)) / (sigA.std(1, keepdims=True) + 1e-8)
    sigB_norm = (sigB - sigB.mean(1, keepdims=True)) / (sigB.std(1, keepdims=True) + 1e-8)

    N_A, N_B = len(sigA), len(sigB)
    sim = np.zeros((N_A, N_B))
    for i in range(N_A):
        for j in range(N_B):
            sim[i, j] = np.dot(sigA_norm[i], sigB_norm[j]) / (
                np.linalg.norm(sigA_norm[i]) * np.linalg.norm(sigB_norm[j]) + 1e-8
            )
    return sim


def _consensus_offset(offsets: list, sims: list, tolerance: int = 2) -> tuple[int, int]:
    """
    RANSAC-style voting: find the offset value with the most inlier support.

    For each candidate offset, count how many others are within ±tolerance
    frames.  Ties broken by total inlier similarity.  The winning offset is
    the similarity-weighted mean of its inliers.

    Returns (best_offset, n_inliers).
    """
    if not offsets:
        return 0, 0

    best_offset, best_count, best_score = offsets[0], 0, 0.0
    for off_candidate in offsets:
        inliers = [(o, s) for o, s in zip(offsets, sims) if abs(o - off_candidate) <= tolerance]
        count = len(inliers)
        score = sum(s for _, s in inliers)
        if count > best_count or (count == best_count and score > best_score):
            best_count = count
            best_score = score
            best_offset = int(round(
                np.average([o for o, _ in inliers], weights=[s for _, s in inliers])
            ))
    return best_offset, best_count


def match_scenes_and_compute_offset(
    scenes_a_idx: np.ndarray,
    sigA: np.ndarray,
    scenes_b_idx: np.ndarray,
    sigB: np.ndarray,
    similarity_threshold: float = 0.7,
    consensus_tolerance: int = 2,
) -> tuple:
    """
    Match cut points between two videos and estimate the temporal offset.

    Two improvements over naive greedy matching:

    1. **Hungarian one-to-one assignment** — each scan scene can only match
       one restored scene and vice-versa.  Eliminates the case where several
       different scan scenes all greedily match the same restored scene
       (a common failure mode with weak features).

    2. **Consensus offset selection** — instead of the raw median (corrupted
       by false-positive matches), finds the offset with the most inlier
       support within ±consensus_tolerance frames, then takes the
       similarity-weighted mean of those inliers.

    Returns:
        (best_offset, confidence, matches, similarity_matrix)
    """
    sim = compute_scene_similarity(sigA, sigB)

    print(f"\n  Similarity matrix: {sim.shape}, mean={sim.mean():.3f}, max={sim.max():.3f}")

    # One-to-one assignment via Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-sim)   # maximise similarity

    matches = [
        (int(i), int(j), float(sim[i, j]))
        for i, j in zip(row_ind, col_ind)
        if sim[i, j] >= similarity_threshold
    ]

    if not matches:
        print(f"  No one-to-one matches above threshold {similarity_threshold}")
        return 0, 0.0, matches, sim

    print(f"  {len(matches)} one-to-one matches (Hungarian):")
    offsets, sims_list = [], []
    for idx_a, idx_b, s in matches:
        off = int(scenes_b_idx[idx_b]) - int(scenes_a_idx[idx_a])
        offsets.append(off)
        sims_list.append(s)
        print(f"    A#{idx_a}(f{scenes_a_idx[idx_a]}) ↔ B#{idx_b}(f{scenes_b_idx[idx_b]})"
              f"  offset={off:+d}  sim={s:.3f}")

    best_offset, n_inliers = _consensus_offset(offsets, sims_list, tolerance=consensus_tolerance)

    total_sim  = sum(sims_list)
    inlier_sim = sum(s for o, s in zip(offsets, sims_list)
                     if abs(o - best_offset) <= consensus_tolerance)
    confidence = inlier_sim / (total_sim + 1e-8)

    print(f"\n  Consensus offset: {best_offset:+d} frames  "
          f"({n_inliers}/{len(matches)} inliers, confidence={confidence:.3f})")

    return best_offset, confidence, matches, sim


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def visualize_scene_changes(
    scenes_a_idx, scenes_a_imgs,
    scenes_b_idx, scenes_b_imgs,
    video_a_name, video_b_name,
    diff_a, diff_b,
    output_path: str = "scene_changes_debug.png",
):
    max_scenes = max(len(scenes_a_imgs), len(scenes_b_imgs))
    fig = plt.figure(figsize=(18, 4 + 2.5 * max_scenes))
    gs = GridSpec(max_scenes + 1, 2, figure=fig, hspace=0.4, wspace=0.15,
                  height_ratios=[1.5] + [1] * max_scenes)
    fig.suptitle("Detected Scene Changes", fontsize=16, fontweight="bold")

    for col, (diff, idxs, imgs, name) in enumerate([
        (diff_a, scenes_a_idx, scenes_a_imgs, video_a_name),
        (diff_b, scenes_b_idx, scenes_b_imgs, video_b_name),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(diff, "b-", linewidth=0.5, alpha=0.7)
        for idx in idxs:
            if idx < len(diff):
                ax.axvline(x=idx, color="r", alpha=0.6, linewidth=1.5)
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Frame"); ax.set_ylabel("Diff score"); ax.grid(True, alpha=0.3)

        for i in range(max_scenes):
            ax2 = fig.add_subplot(gs[i + 1, col])
            if i < len(imgs):
                ax2.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
                ax2.set_title(f"Cut #{i+1} – Frame {idxs[i]}", fontsize=9)
            else:
                ax2.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax2.transAxes)
            ax2.axis("off")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def visualize_similarity_matrix(sim, scenes_a_idx, scenes_b_idx, matches,
                                output_path: str = "similarity_matrix.png"):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim, cmap="viridis", aspect="auto")
    if matches:
        ax.scatter([m[1] for m in matches], [m[0] for m in matches],
                   color="red", s=200, facecolors="none", edgecolors="red", linewidths=2)
    plt.colorbar(im, ax=ax, label="Similarity")
    ax.set_xlabel("Video B scene"); ax.set_ylabel("Video A scene")
    ax.set_title("Scene Similarity Matrix", fontsize=14, fontweight="bold")
    if len(scenes_a_idx) <= 20:
        ax.set_yticks(range(len(scenes_a_idx)))
        ax.set_yticklabels([f"{i}(f{scenes_a_idx[i]})" for i in range(len(scenes_a_idx))], fontsize=8)
    if len(scenes_b_idx) <= 20:
        ax.set_xticks(range(len(scenes_b_idx)))
        ax.set_xticklabels([f"{i}(f{scenes_b_idx[i]})" for i in range(len(scenes_b_idx))], fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def visualize_matched_scenes(matches, scenes_a_idx, scenes_a_imgs,
                              scenes_b_idx, scenes_b_imgs,
                              video_a_name, video_b_name,
                              output_path: str = "matched_scenes.png"):
    if not matches:
        print("No matches to visualize")
        return

    fig = plt.figure(figsize=(16, 2.5 * len(matches)))
    gs = GridSpec(len(matches), 3, figure=fig, hspace=0.3, wspace=0.1, width_ratios=[1, 1, 0.15])
    fig.suptitle("Matched Scene Pairs", fontsize=16, fontweight="bold")

    for i, (idx_a, idx_b, sim) in enumerate(matches):
        offset = int(scenes_b_idx[idx_b]) - int(scenes_a_idx[idx_a])

        ax_a = fig.add_subplot(gs[i, 0])
        ax_a.imshow(cv2.cvtColor(scenes_a_imgs[idx_a], cv2.COLOR_BGR2RGB))
        ax_a.set_title(f"{video_a_name}\nScene #{idx_a} @ f{scenes_a_idx[idx_a]}", fontsize=9)
        ax_a.axis("off")

        ax_b = fig.add_subplot(gs[i, 1])
        ax_b.imshow(cv2.cvtColor(scenes_b_imgs[idx_b], cv2.COLOR_BGR2RGB))
        ax_b.set_title(f"{video_b_name}\nScene #{idx_b} @ f{scenes_b_idx[idx_b]}", fontsize=9)
        ax_b.axis("off")

        ax_info = fig.add_subplot(gs[i, 2])
        ax_info.axis("off")
        ax_info.text(0.1, 0.5, f"#{i+1}\nsim={sim:.3f}\noffset={offset:+d}",
                     fontsize=9, va="center",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: str, video_path: str, tag: str, **params) -> str:
    """
    Build a deterministic cache filename from the video path + tag + params.
    Uses a short hash of (absolute path + params) so different runs with
    different settings don't share a cache.
    """
    key = str(Path(video_path).resolve()) + str(sorted(params.items()))
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    stem = Path(video_path).stem
    return os.path.join(cache_dir, f"{stem}__{tag}__{h}.pkl")


def _load_cache(path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(path: str, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Cached → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect scene cuts in two videos and estimate temporal offset between them."
    )
    parser.add_argument("--scan", required=True, help="Path to scan (unrestored) video")
    parser.add_argument("--restored", required=True, help="Path to restored video")
    parser.add_argument("--max-frames", type=int, default=1800,
                        help="Max frames to scan for cuts (default: 1800 = ~60s at 30fps)")
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="Cut detection threshold (default: 30)")
    parser.add_argument("--max-scenes", type=int, default=20,
                        help="Max scene cuts to use (default: 20)")
    parser.add_argument("--sim-threshold", type=float, default=0.5,
                        help="Minimum similarity to accept a scene match (default: 0.5)")
    parser.add_argument("--output-dir", default=".", help="Directory to save visualization PNGs")
    parser.add_argument("--features", choices=["cnn", "histogram"], default="cnn",
                        help="Feature type for scene similarity (default: cnn). "
                             "cnn uses MobileNetV2 — much more discriminative than histograms. "
                             "Falls back to histogram automatically if torch is not installed.")
    parser.add_argument("--cache-dir", default=".scene_cache",
                        help="Directory for caching scene detection and feature results "
                             "so re-runs skip the expensive computation (default: .scene_cache). "
                             "Delete this directory or use --no-cache to force recompute.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore and overwrite any existing cached results.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir,  exist_ok=True)
    out = args.output_dir

    scene_params = dict(max_frames=args.max_frames, threshold=args.threshold,
                        max_scenes=args.max_scenes)

    print("=" * 70)
    print("STEP 1 — Detecting scene cuts")
    print("=" * 70)

    def _get_scenes(video_path):
        cache_file = _cache_path(args.cache_dir, video_path, "scenes", **scene_params)
        if not args.no_cache:
            cached = _load_cache(cache_file)
            if cached is not None:
                print(f"  [cache hit] {Path(video_path).name}")
                return cached
        result = detect_scene_changes(
            video_path, max_frames=args.max_frames,
            threshold=args.threshold, max_scenes=args.max_scenes,
        )
        _save_cache(cache_file, result)
        return result

    try:
        scenes_a_idx, scenes_a_imgs, diff_a, _ = _get_scenes(args.scan)
        scenes_b_idx, scenes_b_imgs, diff_b, _ = _get_scenes(args.restored)
    except (IOError, ValueError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print(f"\nScan:     {len(scenes_a_idx)} cuts at frames {list(scenes_a_idx)}")
    print(f"Restored: {len(scenes_b_idx)} cuts at frames {list(scenes_b_idx)}")

    visualize_scene_changes(
        scenes_a_idx, scenes_a_imgs,
        scenes_b_idx, scenes_b_imgs,
        args.scan, args.restored,
        diff_a, diff_b,
        output_path=f"{out}/scene_changes_debug.png",
    )

    print("\n" + "=" * 70)
    print("STEP 2 — Computing scene features")
    print("=" * 70)

    extractor = None
    if args.features == "cnn":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            extractor = CNNFeatureExtractor(device=device)
            print(f"  Using CNN features (MobileNetV2, device={device})")
        except ImportError:
            print("  torch/torchvision not found — falling back to histogram features")

    if extractor is None:
        print("  Using histogram features (39-dim)")

    feat_params = dict(**scene_params, features=args.features)

    def _get_features(video_path, frame_indices):
        cache_file = _cache_path(args.cache_dir, video_path, "features", **feat_params)
        if not args.no_cache:
            cached = _load_cache(cache_file)
            if cached is not None:
                print(f"  [cache hit] features for {Path(video_path).name}")
                return cached
        result = extract_features_for_frames(video_path, frame_indices, extractor=extractor)
        _save_cache(cache_file, result)
        return result

    sigA = _get_features(args.scan,     scenes_a_idx)
    sigB = _get_features(args.restored, scenes_b_idx)

    print("\n" + "=" * 70)
    print("STEP 3 — Matching scenes and computing offset")
    print("=" * 70)

    offset, confidence, matches, sim = match_scenes_and_compute_offset(
        scenes_a_idx, sigA, scenes_b_idx, sigB,
        similarity_threshold=args.sim_threshold,
    )

    visualize_similarity_matrix(sim, scenes_a_idx, scenes_b_idx, matches,
                                output_path=f"{out}/similarity_matrix.png")
    visualize_matched_scenes(matches, scenes_a_idx, scenes_a_imgs,
                              scenes_b_idx, scenes_b_imgs,
                              args.scan, args.restored,
                              output_path=f"{out}/matched_scenes.png")

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Best offset (restored vs scan): {offset:+d} frames")
    print(f"Confidence: {confidence:.3f}  (matched scenes: {len(matches)})")
    if offset > 0:
        print(f"→ Restored starts {offset} frames AFTER the scan.")
    elif offset < 0:
        print(f"→ Restored starts {-offset} frames BEFORE the scan.")
    else:
        print("→ Videos appear to be already aligned.")

    if matches:
        offsets = [int(scenes_b_idx[m[1]]) - int(scenes_a_idx[m[0]]) for m in matches]
        print(f"\nOffset distribution: median={np.median(offsets):.0f}  "
              f"mean={np.mean(offsets):.1f}  std={np.std(offsets):.1f}  "
              f"range=[{np.min(offsets)}, {np.max(offsets)}]")

    print("\nTips:")
    print(f"  Too few cuts?  Lower --threshold (currently {args.threshold})")
    print(f"  No matches?    Lower --sim-threshold (currently {args.sim_threshold})")
    print("=" * 70)


if __name__ == "__main__":
    main()
