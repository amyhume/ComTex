from comtex.utils import *
import subprocess
from pathlib import Path
import cv2
import numpy as np

def compute_flicker(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Cannot open video {video_path}")
        return None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur_s = frame_count / fps

    if dur_s > 10800:
        return None, None, None, None

    ret, prev_frame = cap.read()
    if not ret:
        print(f"Warning: Cannot read first frame of {video_path}")
        return None, None, None, None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flickers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))
        flickers.append(diff.mean())

        prev_gray = gray

    cap.release()
    if len(flickers) == 0:
        return None, None, None, None

    flickers = np.array(flickers, dtype=np.float32)
    time_s = np.arange(len(flickers)) / fps
    mean_flicker = float(np.mean(flickers))
    std_flicker = float(np.std(flickers))

    if '.npz' in output_path:
        np.savez_compressed(output_path, flicker=flickers, time_s=time_s)
    else:
        print(f"Couldn't save file: must be .npz format. Returning SF series still")
    
    return mean_flicker, std_flicker, flickers, time_s

def extract_frames_ffmpeg(video_path, frames_dir, fps=1):
    """
    Robustly extract 1 frame per second using ffmpeg.
    Skips if frames already exist.
    """

    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Resume / skip-if-exists
    if any(frames_dir.glob("frame_*.jpg")):
        print(f"[ffmpeg] Frames already exist, skipping: {frames_dir}")
        return

    output_pattern = str(frames_dir / "frame_%06d.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-fflags", "+genpts",     # FIX broken timestamps
        "-i", video_path,
        "-vf", f"fps={str(fps)}",           # RELIABLE 1 Hz sampling
        "-vsync", "vfr",
        "-q:v", "2",
        output_pattern
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[ffmpeg ERROR] {video_path}")
        print(e.stderr.strip())
        raise
