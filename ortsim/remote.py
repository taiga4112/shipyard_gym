"""Remote command port: lets an outside process (robot_controller's bridge) drive the robot.

One UDP socket on localhost (``--remote PORT``). The bridge sends JSON datagrams

    {"t": "cmd", "vx": 0.5, "vy": 0.0, "wz": 0.2}     velocity target (m/s, rad/s), ~50 Hz
    {"t": "reset"}                                     respawn the robot

and ortsim answers the last sender with

    {"t": "state", ...}                                robot state, every control step
    b"JPG0" + jpeg bytes                               head-camera frame (one datagram)

A target that is not refreshed for ``timeout`` seconds falls back to zero, so a crashed
bridge leaves the robot standing. While remote targets are fresh they replace the held
keys; the command still ramps with the policy's accel/decel limits.
"""

from __future__ import annotations

import json
import socket
import time

import numpy as np

FRAME_MAGIC = b"JPG0"
MAX_DATAGRAM = 65000


class RemotePort:
    def __init__(self, port: int, host: str = "127.0.0.1", timeout: float = 0.3):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.timeout = timeout
        self.peer: tuple[str, int] | None = None
        self.target = np.zeros(3)
        self._last_cmd = -1.0
        self.reset_requested = False
        print(f"[remote] listening on udp://{host}:{port}", flush=True)

    @property
    def active(self) -> bool:
        return self._last_cmd >= 0 and time.monotonic() - self._last_cmd < self.timeout

    def poll(self) -> None:
        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                return
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            self.peer = addr
            kind = msg.get("t")
            if kind == "cmd":
                self.target = np.array([float(msg.get(k, 0.0)) for k in ("vx", "vy", "wz")])
                self._last_cmd = time.monotonic()
            elif kind == "reset":
                self.reset_requested = True

    def command_target(self) -> np.ndarray | None:
        """Velocity target while the bridge is driving, else None (keyboard has control)."""
        return self.target if self.active else None

    def send_state(self, state, command, step_count: int, step_dt: float, policy: str, reborn_count: int) -> None:
        if self.peer is None:
            return
        q = state.base_quat
        w, x, y, z = q
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        from .mathutils import quat_rotate_inverse

        v_body = quat_rotate_inverse(q, state.lin_vel_w)
        w_body = quat_rotate_inverse(q, state.ang_vel_w)
        spec = command.spec
        msg = {
            "t": "state",
            "policy": policy,
            "time": round(step_count * step_dt, 3),
            "pos": [round(float(v), 3) for v in state.base_pos],
            "rpy": [round(float(v), 4) for v in (roll, pitch, yaw)],
            "vel": [round(float(v_body[0]), 3), round(float(v_body[1]), 3), round(float(w_body[2]), 3)],
            "cmd": [round(float(v), 3) for v in command.value],
            "limits": {
                "vx": list(spec.limits["lin_vel_x"]),
                "vy": list(spec.limits["lin_vel_y"]),
                "wz": list(spec.limits["ang_vel_z"]),
            },
            "remote": self.active,
            "reborn": reborn_count,
        }
        self._send(json.dumps(msg).encode())

    def send_frame(self, jpeg: bytes) -> None:
        if self.peer is None or len(jpeg) + len(FRAME_MAGIC) > MAX_DATAGRAM:
            return
        self._send(FRAME_MAGIC + jpeg)

    def _send(self, data: bytes) -> None:
        try:
            self.sock.sendto(data, self.peer)
        except OSError:
            pass  # bridge gone; keep simulating

    def close(self) -> None:
        self.sock.close()
