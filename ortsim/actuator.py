"""Explicit PD with the torque-speed envelope of ``UnitreeActuator``.

Computed on the host every physics step and applied as joint effort, identically in
both backends, so the motor model does not depend on what each simulator offers.
"""

from __future__ import annotations

import numpy as np

from .config import PolicySpec


class TorqueSpeedPD:
    def __init__(self, spec: PolicySpec):
        motors = [spec.robot.motors[j] for j in spec.joint_names]
        self.kp = spec.stiffness.copy()
        self.kd = spec.damping.copy()
        self.y1 = np.array([m.Y1 for m in motors])
        self.y2 = np.array([m.Y1 if m.Y2 is None else m.Y2 for m in motors])
        self.x1 = np.array([m.X1 for m in motors])
        self.x2 = np.array([m.X2 for m in motors])
        self.fs = np.array([m.Fs for m in motors])
        self.fd = np.array([m.Fd for m in motors])
        self.va = np.array([m.Va for m in motors])

    def __call__(self, q_des: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        tau = self.kp * (q_des - q) - self.kd * qd
        same_direction = qd * tau > 0
        max_tau = np.where(same_direction, self.y1, self.y2)
        speed = np.abs(qd)
        span = np.where(self.x2 > self.x1, self.x2 - self.x1, 1.0)  # X2 == X1: no falloff band
        falloff = np.clip(max_tau - max_tau / span * (speed - self.x1), 0.0, None)
        max_tau = np.where(speed < self.x1, max_tau, falloff)
        tau = np.clip(tau, -max_tau, max_tau)
        return tau - (self.fs * np.tanh(qd / self.va) + self.fd * qd)
