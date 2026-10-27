"""Fall detection and automatic respawn ("reborn").

A fall is: body tilted past ``tilt_deg``, root closer than ``min_height`` to the
ground straight below it, or no ground below it at all (off the edge of a scene), held
for ``hold_s``; or a non-finite state (sim blew up), immediately. Ground height comes from a downward raycast against the scene, so steps
and slopes do not read as falls.

Respawn targets:
  start -- the scene's spawn pose
  last  -- where the robot was upright ``rewind_s`` before the fall started, so a
           demo can retry an obstacle; repeated falls rewind further back.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .mathutils import quat_rotate_inverse, yaw_of
from .runner import RobotState

MODES = ("start", "last", "off")


@dataclass
class Pose:
    x: float
    y: float
    ground_z: float
    yaw: float


class Reborn:
    def __init__(
        self,
        mode: str,
        step_dt: float,
        tilt_deg: float = 60.0,
        min_height: float = 0.25,
        hold_s: float = 0.5,
        rewind_s: float = 2.0,
        grace_s: float = 1.0,
    ):
        if mode not in MODES:
            raise ValueError(f"reborn mode must be one of {MODES}")
        self.mode = mode
        self.step_dt = step_dt
        self.cos_tilt = np.cos(np.deg2rad(tilt_deg))
        self.min_height = min_height
        self.hold_steps = max(1, round(hold_s / step_dt))
        self.rewind_steps = max(1, round(rewind_s / step_dt))
        self.grace_steps = round(grace_s / step_dt)
        self.history: deque[tuple[int, Pose]] = deque(maxlen=self.rewind_steps * 8)
        self.count = 0
        self._bad = 0
        self._age = 0
        self._clock = 0
        self._quick_failures = 0

    def respawned(self, manual: bool = False) -> None:
        """Call after every reset so the next fall is judged from scratch."""
        quick = not manual and self._age < self.grace_steps + round(3.0 / self.step_dt)
        self._quick_failures = self._quick_failures + 1 if quick else 0
        self._bad = 0
        self._age = 0

    def update(self, state: RobotState, ground_z: float) -> tuple[str, Pose | None] | None:
        """Returns (reason, pose) when a respawn is due; pose None means the scene spawn."""
        if self.mode == "off":
            return None
        self._age += 1
        self._clock += 1
        finite = all(np.isfinite(v).all() for v in (state.base_pos, state.base_quat, state.joint_pos))
        if not finite:
            return "non-finite state", self._target()

        up = -quat_rotate_inverse(state.base_quat, np.array([0.0, 0.0, -1.0]))[2]
        height = state.base_pos[2] - ground_z if np.isfinite(ground_z) else np.inf
        reason = None
        if up < self.cos_tilt:
            reason = f"tilt {np.degrees(np.arccos(np.clip(up, -1, 1))):.0f} deg"
        elif height < self.min_height:
            reason = f"base {height:.2f} m above ground"
        elif not np.isfinite(ground_z):
            # Nothing below at all: off the edge of the scene, falling for good.
            reason = "no ground below"

        if reason is None:
            self._bad = 0
            if np.isfinite(ground_z):
                x, y = (float(v) for v in state.base_pos[:2])
                self.history.append((self._clock, Pose(x, y, float(ground_z), yaw_of(state.base_quat))))
            return None
        if self._age <= self.grace_steps:
            return None
        self._bad += 1
        if self._bad < self.hold_steps:
            return None
        return reason, self._target()

    def _target(self) -> Pose | None:
        self.count += 1
        if self.mode == "start":
            return None
        # Rewind from when the fall started (not from now); each quick re-fall at the
        # same spot rewinds one more ``rewind_s`` further back.
        target = self._clock - self._bad - self.rewind_steps * (1 + self._quick_failures)
        candidates = [pose for t, pose in self.history if t <= target]
        return candidates[-1] if candidates else None
