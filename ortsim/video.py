"""MP4 writing and the text overlay shared by both backends."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from . import REPO_ROOT

DEFAULT_VIDEO_DIR = REPO_ROOT / "videos"


def default_video_path(policy: str, scene: str, backend: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_VIDEO_DIR / f"{stamp}_{scene.replace('/', '-')}_{policy}_{backend}.mp4"


class VideoWriter:
    def __init__(self, path: Path, fps: float, overlay: bool = True):
        import imageio.v2 as imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.overlay = overlay
        self._writer = imageio.get_writer(
            str(self.path), fps=fps, codec="libx264", quality=8, macro_block_size=8,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        self.frames = 0
        print(f"[video] recording -> {self.path}", flush=True)

    def write(self, rgb: np.ndarray, lines: list[str] | None = None) -> None:
        frame = np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)
        if self.overlay and lines:
            frame = draw_lines(frame, lines)
        self._writer.append_data(frame)
        self.frames += 1

    def close(self) -> None:
        self._writer.close()
        print(f"[video] saved {self.frames} frames -> {self.path}", flush=True)


def draw_lines(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2

    h = frame.shape[0]
    scale = max(0.5, h / 1080 * 0.9)
    thick = max(1, int(round(scale * 2)))
    y = int(40 * scale)
    for line in lines:
        cv2.putText(frame, line, (int(20 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(frame, line, (int(20 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += int(38 * scale)
    return frame
