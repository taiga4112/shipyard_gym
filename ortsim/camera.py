"""Follow camera with view presets and a spring arm that keeps walls out of the way.

The eye sits at a preset offset in the robot's (smoothed) yaw frame. Every frame a ray
is cast from the look-at point toward the eye against the scene; if a wall is in
between, the arm shortens to just in front of it at once, and grows back slowly once
the view is clear, so recordings do not jitter.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .mathutils import yaw_of

# Name -> eye offset in the yaw frame [m] (x forward, y left, z up). "rear_right" is
# replaced by the robot config's follow_offset when given.
DEFAULT_PRESETS = {
    "rear_right": (-2.3, -1.6, 1.2),
    "rear_left": (-2.3, 1.6, 1.2),
    "behind": (-2.8, 0.0, 1.0),
    "side_right": (0.0, -2.8, 0.8),
    "top": (-0.8, 0.0, 3.5),
}

Raycast = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


class FollowCamera:
    def __init__(
        self,
        presets: dict[str, tuple[float, float, float]],
        start: str | None = None,
        target_offset=(0.0, 0.0, 0.0),
        smoothing: float = 0.08,
        min_arm: float = 0.6,
        wall_margin: float = 0.2,
        regrow: float = 0.03,
    ):
        self.presets = {k: np.asarray(v, dtype=np.float64) for k, v in presets.items()}
        self.names = list(self.presets)
        self.index = self.names.index(start) if start else 0
        self.offset = self.presets[self.names[self.index]].copy()
        self.target_offset = np.asarray(target_offset, dtype=np.float64)
        self.smoothing = smoothing
        self.min_arm = min_arm
        self.wall_margin = wall_margin
        self.regrow = regrow
        self._target: np.ndarray | None = None
        self._yaw = 0.0
        self._arm = 1.0  # fraction of the preset offset currently in use
        self.zoom = 1.0  # user zoom, applied before the wall check

    @classmethod
    def from_config(cls, cam_cfg: dict, start: str | None = None) -> "FollowCamera":
        presets = dict(DEFAULT_PRESETS)
        if "follow_offset" in cam_cfg:
            presets["rear_right"] = tuple(cam_cfg["follow_offset"])
        presets.update({k: tuple(v) for k, v in (cam_cfg.get("presets") or {}).items()})
        if start is not None and start not in presets:
            raise ValueError(f"unknown camera view {start!r}; choose from {list(presets)}")
        return cls(presets, start, cam_cfg.get("target_offset", (0.0, 0.0, 0.0)))

    @property
    def view(self) -> str:
        return self.names[self.index]

    def cycle(self) -> str:
        self.index = (self.index + 1) % len(self.names)
        print(f"[camera] {self.view}", flush=True)
        return self.view

    def reset(self) -> None:
        self._target = None
        self._arm = 1.0

    def update(self, base_pos: np.ndarray, base_quat: np.ndarray, raycast: Raycast | None = None):
        """Returns (eye, target) in world coordinates."""
        yaw = yaw_of(base_quat)
        target = base_pos + self.target_offset
        goal_offset = self.presets[self.view]
        if self._target is None:
            self._target, self._yaw = target.copy(), yaw
            self.offset = goal_offset.copy()
        else:
            a = self.smoothing
            self._target += a * (target - self._target)
            self._yaw += a * np.arctan2(np.sin(yaw - self._yaw), np.cos(yaw - self._yaw))
            self.offset += a * (goal_offset - self.offset)  # glide between presets

        c, s = np.cos(self._yaw), np.sin(self._yaw)
        ox, oy, oz = self.offset
        arm = np.array([c * ox - s * oy, s * ox + c * oy, oz]) * self.zoom
        length = float(np.linalg.norm(arm))

        allowed = 1.0
        if raycast is not None and length > 1e-6:
            direction = arm / length
            hit = raycast(self._target, direction[None, :], length + self.wall_margin)[0]
            if np.isfinite(hit).all():
                dist = float(np.linalg.norm(hit - self._target))
                allowed = max(self.min_arm, dist - self.wall_margin) / length
        allowed = min(allowed, 1.0)
        # Snap in when blocked, ease back out when clear.
        self._arm = allowed if allowed < self._arm else self._arm + self.regrow * (allowed - self._arm)
        return self._target + arm * self._arm, self._target.copy()
