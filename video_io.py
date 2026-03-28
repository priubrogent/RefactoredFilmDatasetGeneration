"""
Video I/O utilities.
"""

import cv2
import numpy as np
from pathlib import Path


def extract_frame(video_path: str, frame_number: int, offset: int = 0) -> np.ndarray | None:
    """
    Extract a frame from a video at (frame_number + offset).

    Args:
        video_path: Path to the video file.
        frame_number: Base frame number.
        offset: Offset added to frame_number (can be negative).

    Returns:
        Frame as uint8 BGR numpy array, or None on failure.
    """
    target = frame_number + offset
    if target < 0:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[video_io] Cannot open: {video_path}")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if target >= total:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def get_video_info(video_path: str) -> dict:
    """Return basic metadata for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    info = {
        "path": str(Path(video_path).resolve()),
        "name": Path(video_path).name,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_s"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0.0
    cap.release()
    return info
