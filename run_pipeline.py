"""
Film restoration dataset pipeline.

For each segment of the film (default: 1 minute):
  1. Run CNN-based scene matching to estimate temporal offset for that segment.
  2. For every Nth frame in the segment (controlled by `stride`):
     a. Extract scan frame.
     b. Extract restored_1 and restored_2 at (scan_frame + segment_offset).
     c. Spatially align restored_1 and restored_2 to scan via ECC.
     d. Compute gradient-difference defect mask.
     e. Save all outputs into segment{START:06d}-{END:06d}/ subfolders.

Usage:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --start-frame 5000 --max-segments 3
"""

import argparse
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from alignment import align_two_images, generate_gradient_difference_mask, compute_defect_mask
from color_library import compute_color_matrix, apply_color_matrix
from video_io import extract_frame, get_video_info
from scene_matching import (
    CNNFeatureExtractor,
    detect_scene_changes,
    extract_features_for_frames,
    match_scenes_and_compute_offset,
    visualize_scene_changes,
    visualize_similarity_matrix,
    visualize_matched_scenes,
    _cache_path,
    _load_cache,
    _save_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save_image(arr: np.ndarray, path: str):
    """Save a numpy array as PNG.  arr can be uint8 or float32 [0,1]."""
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, arr)


def _segment_dir(output_dir: str, start: int, end: int, cfg: dict) -> str:
    d = os.path.join(output_dir, f"segment{start:06d}-{end:06d}")
    subs = ["scan", "restored_1", "restored_1_original",
            "restored_2", "restored_2_original", "mask", "mask_raw", "debug"]
    if cfg.get("save_failures", True):
        subs.append("debug/alignment_failures")
    if cfg.get("save_corrected", False):
        subs += ["scan_corrected_r1", "scan_corrected_r2"]
    if cfg.get("save_inpainted", False):
        subs.append("inpainted")
    for sub in subs:
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d


def _dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, kernel)


def _load_lama(device=None):
    """Load LaMa inpainter. Returns None if not installed."""
    try:
        from simple_lama_inpainting import SimpleLama
        import torch
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return SimpleLama(device=device)
    except ImportError:
        print("[warn] simple-lama-inpainting not installed — --save-inpainted disabled.")
        return None


# ---------------------------------------------------------------------------
# Per-segment temporal offset estimation
# ---------------------------------------------------------------------------

def estimate_segment_offset(
    scan_path: str,
    restored_path: str,
    seg_start: int,
    seg_end: int,
    cfg: dict,
    extractor,
    fallback_offset: int,
    debug_dir: str = None,
) -> int:
    """
    Run scene matching within [seg_start, seg_end) of both videos and return
    the best temporal offset (restored = scan + offset).

    Uses caching so re-runs are fast.
    """
    sc = cfg["scene_matching"]
    cache_dir  = sc.get("cache_dir", ".scene_cache")
    use_cache  = sc.get("use_cache", True)
    max_frames = sc.get("max_frames", 1440)
    threshold  = sc.get("threshold", 30.0)
    max_scenes = sc.get("max_scenes", 20)
    fac        = sc.get("frames_after_cut", 5)
    sim_thr    = sc.get("sim_threshold", 0.50)
    tol        = sc.get("consensus_tolerance", 3)

    scene_params = dict(
        max_frames=max_frames, threshold=threshold,
        max_scenes=max_scenes, seg_start=seg_start,
    )

    def _get_scenes(video_path, start_frame):
        cache_file = _cache_path(cache_dir, video_path, "scenes_seg", **{**scene_params, "vp": video_path})
        if use_cache:
            cached = _load_cache(cache_file)
            if cached is not None:
                print(f"    [cache] scene detection {Path(video_path).name} seg={start_frame}")
                return cached
        result = _detect_scene_changes_windowed(video_path, start_frame, max_frames, threshold, max_scenes, fac)
        _save_cache(cache_file, result)
        return result

    scenes_a_idx, scenes_a_imgs, diff_a = _get_scenes(scan_path, seg_start)
    # For the restored video the window is offset by fallback_offset
    restored_start = seg_start + fallback_offset
    scenes_b_idx_local, scenes_b_imgs, diff_b = _get_scenes(restored_path, max(0, restored_start))

    # scenes_b_idx_local are frame numbers relative to restored video start;
    # convert back to scan-relative by subtracting fallback_offset so offsets
    # are in scan-frame space.
    scenes_b_idx = scenes_b_idx_local  # absolute restored frame numbers

    feat_params = dict(**scene_params, features=sc.get("features", "cnn"))

    def _get_features(video_path, frame_indices):
        cache_file = _cache_path(cache_dir, video_path, "feats_seg",
                                 **{**feat_params, "vp": video_path, "frames": str(frame_indices[:3].tolist())})
        if use_cache:
            cached = _load_cache(cache_file)
            if cached is not None:
                return cached
        result = extract_features_for_frames(video_path, frame_indices, extractor=extractor)
        _save_cache(cache_file, result)
        return result

    if len(scenes_a_idx) == 0 or len(scenes_b_idx) == 0:
        print(f"    No scene cuts found in segment — keeping offset {fallback_offset:+d}")
        return fallback_offset

    sigA = _get_features(scan_path, scenes_a_idx)
    sigB = _get_features(restored_path, scenes_b_idx)

    offset, confidence, matches, sim = match_scenes_and_compute_offset(
        scenes_a_idx, sigA, scenes_b_idx, sigB,
        similarity_threshold=sim_thr,
        consensus_tolerance=tol,
    )

    # --- Save debug visualizations into segment's debug/ folder ---
    if debug_dir is not None:
        scan_name     = Path(scan_path).stem
        restored_name = Path(restored_path).stem
        try:
            visualize_scene_changes(
                scenes_a_idx, scenes_a_imgs,
                scenes_b_idx, scenes_b_imgs,
                scan_name, restored_name,
                diff_a, diff_b,
                output_path=os.path.join(debug_dir, "scene_changes.png"),
            )
            visualize_similarity_matrix(
                sim, scenes_a_idx, scenes_b_idx, matches,
                output_path=os.path.join(debug_dir, "similarity_matrix.png"),
            )
            visualize_matched_scenes(
                matches, scenes_a_idx, scenes_a_imgs,
                scenes_b_idx, scenes_b_imgs,
                scan_name, restored_name,
                output_path=os.path.join(debug_dir, "matched_scenes.png"),
            )
        except Exception as e:
            print(f"    [warn] debug visualization failed: {e}")

    if not matches:
        print(f"    No matches — keeping offset {fallback_offset:+d}")
        return fallback_offset

    print(f"    Segment offset={offset:+d}  confidence={confidence:.3f}  ({len(matches)} matches)")
    return offset


def _detect_scene_changes_windowed(
    video_path: str,
    start_frame: int,
    max_frames: int,
    threshold: float,
    max_scenes: int,
    frames_after_cut: int,
) -> tuple:
    """
    Detect scene changes within a window starting at start_frame.
    Returns (representative_indices, representative_images, diff_scores)
    where indices are absolute frame numbers in the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, min(start_frame, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_to_read = min(max_frames, total - start_frame)
    resize = (160, 90)
    min_scene_duration = 10
    fade_window = 5
    brightness_threshold = 20.0

    prev_hist = None
    diff_scores = []
    all_frames = []
    frame_indices = []

    for i in range(frames_to_read):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, resize, interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([gray_small], [0], None, [64], [0, 256]).flatten()
        hist /= hist.sum() + 1e-8
        diff_scores.append(np.sum(np.abs(hist - prev_hist)) * 100 if prev_hist is not None else 0.0)
        prev_hist = hist.copy()
        all_frames.append(frame.copy())
        frame_indices.append(start_frame + i)

    cap.release()

    if not all_frames:
        return np.array([]), [], np.array([])

    diff_scores = np.array(diff_scores)
    brightness_scores = np.array([
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), resize).mean()
        for f in all_frames
    ])

    potential_cuts = np.where(diff_scores > threshold)[0]
    filtered_cuts = []
    for cut_idx in potential_cuts:
        if filtered_cuts and (cut_idx - filtered_cuts[-1]) < min_scene_duration:
            continue
        cs = max(0, cut_idx - fade_window)
        ce = min(len(brightness_scores), cut_idx + fade_window + 1)
        if np.any(brightness_scores[cs:ce] < brightness_threshold):
            continue
        if cut_idx >= fade_window and cut_idx < len(diff_scores) - fade_window:
            window = diff_scores[max(0, cut_idx - fade_window): min(len(diff_scores), cut_idx + fade_window + 1)]
            if np.sum(window > threshold * 0.6) > fade_window:
                continue
        filtered_cuts.append(cut_idx)

    if not filtered_cuts:
        # No cuts — use start frame as sole representative
        repr_idx = min(frames_after_cut, len(all_frames) - 1)
        return np.array([frame_indices[repr_idx]]), [all_frames[repr_idx]], diff_scores

    scene_change_idx = np.array(filtered_cuts)
    if len(scene_change_idx) > max_scenes:
        strongest = np.argsort(diff_scores[scene_change_idx])[-max_scenes:]
        scene_change_idx = np.sort(scene_change_idx[strongest])

    representative_indices, representative_images = [], []
    for cut_idx in scene_change_idx:
        repr_local = min(cut_idx + frames_after_cut, len(all_frames) - 1)
        representative_indices.append(frame_indices[repr_local])
        representative_images.append(all_frames[repr_local])

    return np.array(representative_indices), representative_images, diff_scores


# ---------------------------------------------------------------------------
# Worker process initializer — loads heavy models once per process
# ---------------------------------------------------------------------------

# Module-level global so worker functions can access it without pickling
_worker_lama = None


def _worker_init(save_inpainted: bool, threads_per_worker: int):
    """Called once when each worker process starts."""
    global _worker_lama

    # Cap OpenCV's internal thread pool so N workers × M threads stays bounded.
    cv2.setNumThreads(threads_per_worker)

    # Same cap for OpenBLAS / numpy (used by colour transfer and scipy).
    os.environ["OMP_NUM_THREADS"]        = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"]   = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"]        = str(threads_per_worker)

    if save_inpainted:
        # Force CPU — each worker loading LaMa on GPU would exhaust VRAM fast.
        import torch
        _worker_lama = _load_lama(device=torch.device("cpu"))


def _frame_task(kwargs: dict) -> bool:
    """Top-level picklable wrapper — required for ProcessPoolExecutor."""
    return process_frame(**kwargs, lama=_worker_lama)


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

def process_frame(
    frame_num: int,
    offset: int,
    scan_path: str,
    r1_path: str,
    r2_path: str,
    seg_dir: str,
    mask_threshold: float,
    mask_morph: int = 3,
    save_failures: bool = True,
    save_corrected: bool = False,
    save_inpainted: bool = False,
    inpaint_dilation: int = 3,
    lama=None,
):
    """Extract, align, mask, and save one frame triplet."""
    tag          = f"{frame_num:06d}.png"
    failures_dir = os.path.join(seg_dir, "debug", "alignment_failures")

    # --- Extract frames ---
    scan_raw = extract_frame(scan_path, frame_num, 0)
    r1_raw   = extract_frame(r1_path,   frame_num, offset)
    r2_raw   = extract_frame(r2_path,   frame_num, offset)

    if scan_raw is None:
        print(f"    [skip] scan frame {frame_num} unavailable")
        return False
    if r1_raw is None or r2_raw is None:
        print(f"    [skip] frame {frame_num}: restored frame unavailable at offset {offset:+d}")
        return False

    # --- Convert to float32 [0,1] ---
    scan_f = scan_raw.astype(np.float32) / 255.0
    r1_f   = r1_raw.astype(np.float32)   / 255.0
    r2_f   = r2_raw.astype(np.float32)   / 255.0

    # --- Spatial alignment ---
    # Falls back to plain resize if ECC fails (NaN / non-convergence).
    # Failed frames are saved to debug/alignment_failures/ for inspection.
    h, w = scan_f.shape[:2]
    r1_failed = r2_failed = False

    r1_aligned, _ = align_two_images(scan_f, r1_f)
    if r1_aligned is None:
        print(f"    [fallback] frame {frame_num}: r1 ECC failed")
        r1_aligned = cv2.resize(r1_f, (w, h), interpolation=cv2.INTER_LINEAR)
        r1_failed = True

    r2_aligned, _ = align_two_images(scan_f, r2_f)
    if r2_aligned is None:
        print(f"    [fallback] frame {frame_num}: r2 ECC failed")
        r2_aligned = cv2.resize(r2_f, (w, h), interpolation=cv2.INTER_LINEAR)
        r2_failed = True

    # Save failure evidence: scan + whichever restored(s) failed to align
    if save_failures and (r1_failed or r2_failed):
        _save_image(scan_raw, os.path.join(failures_dir, f"{frame_num:06d}_scan.png"))
        if r1_failed:
            _save_image(r1_raw, os.path.join(failures_dir, f"{frame_num:06d}_r1.png"))
        if r2_failed:
            _save_image(r2_raw, os.path.join(failures_dir, f"{frame_num:06d}_r2.png"))

    # --- Defect mask ---
    diff1 = generate_gradient_difference_mask(scan_f, r1_aligned)
    diff2 = generate_gradient_difference_mask(scan_f, r2_aligned)
    mask, mask_raw = compute_defect_mask(diff1, diff2, threshold=mask_threshold, morph_kernel=mask_morph)

    # --- Normalise mask_raw for saving (map to [0,1]) ---
    mr_min, mr_max = mask_raw.min(), mask_raw.max()
    mask_raw_vis = ((mask_raw - mr_min) / (mr_max - mr_min)).astype(np.float32) if mr_max > mr_min else np.zeros_like(mask_raw)

    # --- Core saves ---
    _save_image(scan_raw,    os.path.join(seg_dir, "scan",                tag))
    _save_image(r1_raw,      os.path.join(seg_dir, "restored_1_original", tag))
    _save_image(r2_raw,      os.path.join(seg_dir, "restored_2_original", tag))
    _save_image(r1_aligned,  os.path.join(seg_dir, "restored_1",          tag))
    _save_image(r2_aligned,  os.path.join(seg_dir, "restored_2",          tag))
    _save_image(mask,        os.path.join(seg_dir, "mask",                tag))
    _save_image(mask_raw_vis, os.path.join(seg_dir, "mask_raw",           tag))

    # --- Optional: colour-corrected scan (scan content, restored colour grade) ---
    if save_corrected:
        scan_rgb = cv2.cvtColor(scan_raw, cv2.COLOR_BGR2RGB)
        r1_rgb   = cv2.cvtColor((r1_aligned * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        r2_rgb   = cv2.cvtColor((r2_aligned * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        H1 = compute_color_matrix(reference=r1_rgb, source=scan_rgb)
        H2 = compute_color_matrix(reference=r2_rgb, source=scan_rgb)
        corrected_r1 = cv2.cvtColor(apply_color_matrix(scan_rgb, H1), cv2.COLOR_RGB2BGR)
        corrected_r2 = cv2.cvtColor(apply_color_matrix(scan_rgb, H2), cv2.COLOR_RGB2BGR)
        _save_image(corrected_r1, os.path.join(seg_dir, "scan_corrected_r1", tag))
        _save_image(corrected_r2, os.path.join(seg_dir, "scan_corrected_r2", tag))

    # --- Optional: LaMa inpainting ---
    if save_inpainted and lama is not None:
        from PIL import Image
        dilated       = _dilate_mask(mask, px=inpaint_dilation)
        scan_pil      = Image.fromarray(cv2.cvtColor(scan_raw, cv2.COLOR_BGR2RGB))
        mask_pil      = Image.fromarray(dilated)
        inpainted     = lama(scan_pil, mask_pil)
        inpainted_bgr = cv2.cvtColor(np.array(inpainted), cv2.COLOR_RGB2BGR)
        _save_image(inpainted_bgr, os.path.join(seg_dir, "inpainted", tag))

    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cfg: dict, cli_overrides: dict):
    # --- Apply CLI overrides ---
    if cli_overrides.get("start_frame") is not None:
        cfg["start_frame"] = cli_overrides["start_frame"]
    if cli_overrides.get("max_segments") is not None:
        cfg["max_segments"] = cli_overrides["max_segments"]

    scan_path = cfg["videos"]["scan"]
    r1_path   = cfg["videos"]["restored_1"]
    r2_path   = cfg["videos"]["restored_2"]
    out_dir   = cfg.get("output_dir", "pipeline_output")
    os.makedirs(out_dir, exist_ok=True)

    start_frame    = cfg.get("start_frame", 0)
    segment_frames = cfg.get("segment_frames", 1440)
    stride         = cfg.get("stride", 25)
    max_segments   = cfg.get("max_segments", None)
    mask_threshold   = cfg["mask"]["threshold"]
    mask_morph       = cfg["mask"].get("morph_kernel", 3)
    save_failures    = cfg.get("save_failures",  True)
    save_corrected   = cfg.get("save_corrected", False)
    save_inpainted   = cfg.get("save_inpainted", False)
    inpaint_dilation = cfg.get("inpaint_dilation", 3)
    workers          = cfg.get("workers", max(1, multiprocessing.cpu_count() - 1))
    max_cores        = cfg.get("max_cores", None)
    if max_cores is not None:
        threads_per_worker = max(1, max_cores // workers)
    else:
        threads_per_worker = max(1, multiprocessing.cpu_count() // workers)

    # --- Video info ---
    print("=" * 60)
    print("Video info")
    print("=" * 60)
    scan_info = get_video_info(scan_path)
    print(f"  Scan:       {scan_info['name']}  "
          f"{scan_info['total_frames']} frames  {scan_info['fps']:.3f} fps")
    r1_info = get_video_info(r1_path)
    print(f"  Restored 1: {r1_info['name']}  {r1_info['total_frames']} frames")
    r2_info = get_video_info(r2_path)
    print(f"  Restored 2: {r2_info['name']}  {r2_info['total_frames']} frames")

    total_scan_frames = scan_info["total_frames"]

    # --- CNN extractor ---
    sc = cfg["scene_matching"]
    extractor = None
    if sc.get("features", "cnn") == "cnn":
        try:
            import torch
            device_cfg = cfg.get("device", "auto")
            if device_cfg == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = device_cfg
            extractor = CNNFeatureExtractor(device=device)
            print(f"\nCNN extractor ready (MobileNetV2, device={device})")
        except ImportError:
            print("\ntorch not found — falling back to histogram features")

    # --- Build segment list ---
    segments = []
    seg_start = start_frame
    while seg_start < total_scan_frames:
        seg_end = min(seg_start + segment_frames, total_scan_frames)
        segments.append((seg_start, seg_end))
        seg_start = seg_end
        if max_segments and len(segments) >= max_segments:
            break

    frames_per_seg = len(range(0, segment_frames, stride))
    total_frames_est = frames_per_seg * len(segments)

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Segments:        {len(segments)}  (frames {start_frame} → {segments[-1][1]})")
    print(f"  Segment length:  {segment_frames} frames  (~{segment_frames / scan_info['fps']:.0f}s)")
    print(f"  Stride:          every {stride} frames  (~{frames_per_seg} frames/segment)")
    print(f"  Frames total:    ~{total_frames_est}")
    print(f"  Output dir:      {out_dir}")
    print()
    print(f"  Scene matching:  {sc.get('features', 'cnn').upper()} features  "
          f"(sim_threshold={sc.get('sim_threshold', 0.5)}, "
          f"tolerance=±{sc.get('consensus_tolerance', 3)})")
    print(f"  Mask threshold:  {mask_threshold}  (morph_kernel={mask_morph})")
    print()
    print(f"  Save failures:   {save_failures}")
    print(f"  Save corrected:  {save_corrected}")
    print(f"  Save inpainted:  {save_inpainted}"
          + (f"  (dilation={inpaint_dilation}px)" if save_inpainted else ""))
    print(f"  Workers:         {workers}  ({threads_per_worker} threads/worker)")
    print("=" * 60 + "\n")

    # Carry forward the offset between segments
    current_offset = 0

    # Create the executor ONCE — workers are reused across all segments.
    # This avoids spawning a fresh pool per segment and leaving zombies behind.
    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(save_inpainted, threads_per_worker),
    )

    try:
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            print("=" * 60)
            print(f"Segment {seg_idx + 1}/{len(segments)}  "
                  f"frames {seg_start}–{seg_end}  (offset so far: {current_offset:+d})")
            print("=" * 60)

            # --- 1. Temporal offset for this segment ---
            print("  Step 1: scene matching for temporal offset...")
            seg_dir = _segment_dir(out_dir, seg_start, seg_end, cfg)
            current_offset = estimate_segment_offset(
                scan_path, r1_path,
                seg_start, seg_end,
                cfg, extractor,
                fallback_offset=current_offset,
                debug_dir=os.path.join(seg_dir, "debug"),
            )

            # --- 2. Frame-level processing (parallel) ---
            frame_nums = list(range(seg_start, seg_end, stride))
            print(f"  Step 2: processing {len(frame_nums)} frames "
                  f"(every {stride} frames, offset={current_offset:+d}, workers={workers})...")

            tasks = [dict(
                frame_num=fn,
                offset=current_offset,
                scan_path=scan_path,
                r1_path=r1_path,
                r2_path=r2_path,
                seg_dir=seg_dir,
                mask_threshold=mask_threshold,
                mask_morph=mask_morph,
                save_failures=save_failures,
                save_corrected=save_corrected,
                save_inpainted=save_inpainted,
                inpaint_dilation=inpaint_dilation,
            ) for fn in frame_nums]

            saved = 0
            futures = {executor.submit(_frame_task, t): t["frame_num"] for t in tasks}
            with tqdm(total=len(futures), desc=f"  seg {seg_start}-{seg_end}") as pbar:
                for fut in as_completed(futures):
                    try:
                        if fut.result():
                            saved += 1
                    except Exception as e:
                        fn = futures[fut]
                        print(f"\n    [error] frame {fn}: {e}")
                    pbar.update(1)

            print(f"  Saved {saved}/{len(frame_nums)} frames → {seg_dir}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted — shutting down workers...")
    finally:
        # Cancel pending futures and terminate worker processes cleanly.
        executor.shutdown(wait=False, cancel_futures=True)
        # Hard-kill any workers that are still alive (e.g. stuck in ECC).
        import multiprocessing as _mp
        for child in _mp.active_children():
            child.terminate()
        print("Workers stopped.")

    print("Pipeline complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Must be set before any CUDA/torch usage in the main process.
    # 'spawn' starts workers fresh (no inherited CUDA context), which is
    # required when torch is imported in the parent process.
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Film restoration dataset pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="Override start_frame from config")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Override max_segments from config (useful for testing)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = _load_config(args.config)
    run(cfg, {"start_frame": args.start_frame, "max_segments": args.max_segments})


if __name__ == "__main__":
    main()
