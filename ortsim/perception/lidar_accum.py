"""Height map accumulated from two body-fixed LiDARs on a world-aligned 0.1 m grid.

NumPy reimplementation of the height map the A2 distill2 policy was trained with. Kept
identical:

* rays: Isaac Lab ``LidarPatternCfg`` (vertical ``linspace(v0, v1, channels)``, 360 deg
  horizontal without the duplicate end), rotated by the mount quaternion then by the
  full base orientation (``ray_alignment="base"``), cast up to ``max_distance``;
* a hit counts if its distance from the **base origin** (the sensor's ``pos_w``, not the
  LiDAR) is within ``[r_min, r_max]`` and, optionally, it is at most ``zmax`` above the base;
* hits are written to world cells ``floor(v / res + phase)``, highest hit per cell per
  step, blended with an asymmetric EMA (``ema_up`` when rising, ``ema_down`` when
  falling, a fresh cell takes the new value); the grid follows the robot in whole cells
  and forgets what leaves its ``+-half`` square;
* output: a yaw-aligned 17 x 11 node window (x inner), value ``(base_z - z - offset) * scale``,
  0 where unseen, followed by the 187 unseen flags.

The write-position noise (per-episode bias plus per-step jitter, mimicking localisation
drift) is reproduced with ``pose_noise`` (1.0 = training); 0 gives a clean map.
"""

from __future__ import annotations

import numpy as np

from ..mathutils import quat_to_matrix, yaw_of


def lidar_pattern(vertical_fov, horizontal_fov, channels: int, horizontal_res: float) -> np.ndarray:
    """Isaac Lab ``lidar_pattern`` ray directions in the sensor frame, (N, 3)."""
    v = np.deg2rad(np.linspace(vertical_fov[0], vertical_fov[1], channels))
    n_h = int(np.ceil((horizontal_fov[1] - horizontal_fov[0]) / horizontal_res)) + 1
    h = np.linspace(horizontal_fov[0], horizontal_fov[1], n_h)
    if abs(abs(horizontal_fov[0] - horizontal_fov[1]) - 360.0) < 1e-6:
        h = h[:-1]
    va, ha = np.meshgrid(v, np.deg2rad(h), indexing="ij")
    return np.stack([np.cos(va) * np.cos(ha), np.cos(va) * np.sin(ha), np.sin(va)], axis=-1).reshape(-1, 3)


class LidarAccumMap:
    def __init__(self, p: dict, seed: int = 0):
        self.res = float(p.get("resolution", 0.1))
        self.phase = float(p.get("phase", 0.25))
        self.half_cells = int(round(float(p.get("follow_half", 6.0)) / self.res))
        self.g = 2 * self.half_cells + 1
        self.offset = float(p.get("offset", 0.0))
        self.scale = float(p.get("value_scale", 0.2))
        self.r_min, self.r_max = float(p.get("r_min", 0.15)), float(p.get("r_max", 6.0))
        self.max_distance = float(p.get("max_distance", self.r_max))
        self.zmax = p.get("zmax")
        self.ema_up = float(p.get("ema_up", 1.0))
        self.ema_down = float(p.get("ema_down", 1.0))
        self.noise = float(p.get("pose_noise", 0.0))
        self.rng = np.random.default_rng(seed)

        base = lidar_pattern(p.get("vertical_fov", (-7.0, 52.0)), p.get("horizontal_fov", (-180.0, 180.0)),
                             int(p.get("channels", 21)), float(p.get("horizontal_res", 3.0)))
        starts, dirs = [], []
        for m in p["mounts"]:
            r = quat_to_matrix(np.asarray(m["rot_wxyz"], dtype=float))
            dirs.append(base @ r.T)
            starts.append(np.broadcast_to(np.asarray(m["pos"], dtype=float), base.shape))
        self.ray_starts_b = np.concatenate(starts)
        self.ray_dirs_b = np.concatenate(dirs)

        wx, wy = p.get("window_x", (-0.8, 0.8)), p.get("window_y", (-0.5, 0.5))
        nx = int(round((wx[1] - wx[0]) / self.res)) + 1
        ny = int(round((wy[1] - wy[0]) / self.res)) + 1
        gy, gx = np.meshgrid(np.arange(ny) * self.res + wy[0], np.arange(nx) * self.res + wx[0], indexing="ij")
        self.cell_xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
        self.num_cells = len(self.cell_xy)
        self.reset()

    def reset(self) -> None:
        self.h = np.full((self.g, self.g), np.nan)   # [y, x], world z
        self.anchor: np.ndarray | None = None        # world cell of the grid centre
        s = self.noise
        self.bias_xy = self.rng.normal(0.0, 0.03 * s, 2) if s > 0 else np.zeros(2)
        self.bias_yaw = self.rng.normal(0.0, np.deg2rad(2.0) * s) if s > 0 else 0.0
        self.unseen = np.ones(self.num_cells, dtype=bool)
        self.last_z = np.full(self.num_cells, np.nan)

    def _world_cell(self, v: np.ndarray) -> np.ndarray:
        return np.floor(v / self.res + self.phase).astype(np.int64)

    def _follow(self, base_xy: np.ndarray) -> None:
        cur = self._world_cell(base_xy)
        if self.anchor is None:
            self.anchor = cur
            return
        dx, dy = cur - self.anchor
        if dx == 0 and dy == 0:
            return
        g = self.g
        new = np.full_like(self.h, np.nan)
        sy, sx = slice(max(dy, 0), g + min(dy, 0)), slice(max(dx, 0), g + min(dx, 0))
        ty, tx = slice(max(-dy, 0), g + min(-dy, 0)), slice(max(-dx, 0), g + min(-dx, 0))
        if sy.start < sy.stop and sx.start < sx.stop:
            new[ty, tx] = self.h[sy, sx]
        self.h = new
        self.anchor = cur

    def rays(self, base_pos: np.ndarray, base_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = quat_to_matrix(base_quat)
        return base_pos + self.ray_starts_b @ r.T, self.ray_dirs_b @ r.T

    def update(self, hits: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
        """Merge this step's ray hits ((N, 3), inf = miss) and return the 374-dim observation."""
        self._follow(base_pos[:2])
        rng = np.linalg.norm(hits - base_pos, axis=1)
        ok = np.isfinite(hits).all(axis=1) & (rng >= self.r_min) & (rng <= self.r_max)
        if self.zmax is not None:
            ok &= hits[:, 2] <= base_pos[2] + float(self.zmax)
        if self.noise > 0:
            dyaw = self.bias_yaw + self.rng.normal(0.0, np.deg2rad(0.2) * self.noise)
            dxy = self.bias_xy + self.rng.normal(0.0, 0.005 * self.noise, 2)
        else:
            dyaw, dxy = 0.0, np.zeros(2)
        h = hits[ok]
        rel = h[:, :2] - base_pos[:2]
        c, s = np.cos(dyaw), np.sin(dyaw)
        wx = base_pos[0] + dxy[0] + c * rel[:, 0] - s * rel[:, 1]
        wy = base_pos[1] + dxy[1] + s * rel[:, 0] + c * rel[:, 1]
        gx = self._world_cell(wx) - self.anchor[0] + self.half_cells
        gy = self._world_cell(wy) - self.anchor[1] + self.half_cells
        inside = (gx >= 0) & (gx < self.g) & (gy >= 0) & (gy < self.g)
        flat = gy[inside] * self.g + gx[inside]
        new = np.full(self.g * self.g, -np.inf)
        np.maximum.at(new, flat, h[inside, 2])
        new = new.reshape(self.g, self.g)
        wrote = np.isfinite(new)
        old = self.h
        a = np.where(new >= old, self.ema_up, self.ema_down)
        blended = np.where(np.isfinite(old), a * new + (1.0 - a) * old, new)
        self.h = np.where(wrote, blended, old)

        yaw = yaw_of(base_quat)
        c, s = np.cos(yaw), np.sin(yaw)
        qx = base_pos[0] + c * self.cell_xy[:, 0] - s * self.cell_xy[:, 1]
        qy = base_pos[1] + s * self.cell_xy[:, 0] + c * self.cell_xy[:, 1]
        cx = np.clip(self._world_cell(qx) - self.anchor[0] + self.half_cells, 0, self.g - 1)
        cy = np.clip(self._world_cell(qy) - self.anchor[1] + self.half_cells, 0, self.g - 1)
        z = self.h[cy, cx]
        self.unseen = ~np.isfinite(z)
        self.last_z = z
        value = np.where(self.unseen, 0.0, (base_pos[2] - z - self.offset) * self.scale)
        return np.concatenate([value, self.unseen.astype(np.float64)])

    def cell_positions_w(self, base_pos: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
        """World positions of the window cells at the height the map holds, for debug drawing."""
        yaw = yaw_of(base_quat)
        c, s = np.cos(yaw), np.sin(yaw)
        x, y = self.cell_xy[:, 0], self.cell_xy[:, 1]
        z = np.where(self.unseen, base_pos[2] - 0.35, self.last_z)
        return np.stack([base_pos[0] + c * x - s * y, base_pos[1] + s * x + c * y, z], axis=-1)
