"""Velocity command driven by held keys, shared by both backends.

Holding a key accelerates that axis toward its limit; releasing it lets the command
ease back to the baseline -- zero, or the cruise command given with ``--command`` --
so the robot slows down and stops on its own instead of walking on forever.
"""

from __future__ import annotations

import numpy as np

from .config import CommandSpec

AXES = ("lin_vel_x", "lin_vel_y", "ang_vel_z")

# key name -> (axis, direction). Names follow carb (Isaac); the MuJoCo viewer maps its
# GLFW codes onto the same names, so both backends share one layout.
KEYMAP = {
    "W": (0, +1), "UP": (0, +1),
    "S": (0, -1), "DOWN": (0, -1),
    "A": (1, +1),
    "D": (1, -1),
    "Q": (2, +1), "LEFT": (2, +1),
    "E": (2, -1), "RIGHT": (2, -1),
}

HELP = """\
Keyboard (hold to move, release to slow down and stop)
  W / S  or  Up / Down      forward / brake
  A / D                     sideways left / right
  Q / E  or  Left / Right   turn left / right
  X                         stop: also clears the --command cruise speed
  R                         reset robot
  C                         next camera view (rear_right / rear_left / behind / side_right / top)
  V                         start / stop video recording
  M                         show / hide height-map markers
  H                         print this help
"""


class VelocityCommand:
    def __init__(self, spec: CommandSpec):
        self.spec = spec
        self._lo = np.array([spec.limits[a][0] for a in AXES])
        self._hi = np.array([spec.limits[a][1] for a in AXES])
        self._accel = np.array(spec.accel, dtype=np.float64)
        self._decel = np.array(spec.decel, dtype=np.float64)
        self.baseline = np.clip(np.array(spec.initial, dtype=np.float64), self._lo, self._hi)
        self.value = self.baseline.copy()
        self._printed = self.value.copy()
        self._settled = True

    def set(self, value) -> None:
        """Cruise command: what the command returns to when no key is held."""
        self.baseline = np.clip(np.asarray(value, dtype=np.float64), self._lo, self._hi)
        self.value = self.baseline.copy()

    def stop(self) -> None:
        self.baseline[:] = 0.0
        print("[cmd] stop", flush=True)

    @staticmethod
    def directions(held: set[str]) -> np.ndarray:
        d = np.zeros(3)
        for name in held:
            if name in KEYMAP:
                axis, sign = KEYMAP[name]
                d[axis] += sign
        return np.sign(d)

    def update(self, held: set[str], dt: float) -> None:
        """Advance one control step given the keys currently held down."""
        d = self.directions(held)
        target = np.where(d > 0, self._hi, np.where(d < 0, self._lo, self.baseline))
        delta = target - self.value
        # Shedding speed (release or reversing) uses the decel rate, gaining it the accel rate.
        slowing = delta * self.value < 0
        rate = np.where(slowing | (d == 0), self._decel, self._accel)
        step = np.clip(delta, -rate * dt, rate * dt)
        self.value = self.value + step
        self._settled = bool(np.all(np.abs(step) < 1e-9))

    def update_target(self, target, dt: float) -> None:
        """Advance one control step toward an analog target (remote controller).

        Same ramp as the keys: gaining speed uses the accel rate, shedding it the decel rate.
        """
        target = np.clip(np.asarray(target, dtype=np.float64), self._lo, self._hi)
        delta = target - self.value
        slowing = (np.abs(target) < np.abs(self.value)) | (target * self.value < 0)
        rate = np.where(slowing, self._decel, self._accel)
        step = np.clip(delta, -rate * dt, rate * dt)
        self.value = self.value + step
        self._settled = bool(np.all(np.abs(step) < 1e-9))

    def maybe_report(self) -> None:
        """Print while ramping (every 0.1) and once when the command settles."""
        moved = np.max(np.abs(self.value - self._printed))
        if moved >= 0.1 - 1e-9 or (self._settled and moved > 1e-6):
            self._printed = self.value.copy()
            self.report()

    def out_of_training_range(self) -> list[str]:
        out = []
        for i, a in enumerate(AXES):
            lo, hi = self.spec.trained_ranges[a]
            if not (lo - 1e-6 <= self.value[i] <= hi + 1e-6):
                out.append(a)
        return out

    def describe(self) -> str:
        vx, vy, wz = self.value
        return f"vx={vx:+.2f} m/s  vy={vy:+.2f} m/s  yaw={wz:+.2f} rad/s"

    def report(self) -> None:
        ood = self.out_of_training_range()
        note = f"   (outside trained range: {', '.join(ood)})" if ood else ""
        print(f"[cmd] {self.describe()}{note}", flush=True)
