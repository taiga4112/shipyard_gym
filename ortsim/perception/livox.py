"""Livox MID-360 ray pattern, consumed as a rolling window over the recorded scan sequence.

Each sensor update takes the next
``samples`` rows of the recorded (theta, phi) sequence, casting every ``downsample``-th
one, so the rosette sweeps the way the hardware does. Rays are rigidly attached to the
mount (``ray_alignment="base"``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..mathutils import quat_to_matrix


class RollingLivoxPattern:
    def __init__(
        self,
        pattern_path: Path,
        mount_pos,
        mount_rot_wxyz,
        samples: int,
        downsample: int = 1,
        max_distance: float = 20.0,
        rolling_window_start: int = 0,
    ):
        sequence = np.load(pattern_path)  # (M, 2): theta, phi [rad]
        theta, phi = sequence[:, 0], sequence[:, 1]
        dirs = np.stack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)], axis=1)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs @ quat_to_matrix(np.asarray(mount_rot_wxyz, dtype=np.float64)).T

        stride = min(int(samples), len(dirs))
        self.num_windows = len(dirs) // stride
        windows = dirs[: self.num_windows * stride].reshape(self.num_windows, stride, 3)
        self.windows = np.ascontiguousarray(windows[:, :: max(int(downsample), 1), :])
        self.mount_pos = np.asarray(mount_pos, dtype=np.float64)
        self.max_distance = float(max_distance)
        self._start = rolling_window_start // stride - 1
        self.index = self._start

    @property
    def num_rays(self) -> int:
        return self.windows.shape[1]

    def reset(self) -> None:
        # The real prisms keep turning across a reset; training never rewinds the window.
        pass

    def next_rays(self, base_pos: np.ndarray, base_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Advance one window. Returns (origin_w (3,), directions_w (R, 3))."""
        self.index = (self.index + 1) % self.num_windows
        rot = quat_to_matrix(base_quat)
        origin = base_pos + rot @ self.mount_pos
        return origin, self.windows[self.index] @ rot.T
