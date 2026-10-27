"""Body-centred height grid built from LiDAR returns.

NumPy port of ``tsubame_isaac...mdp.lidar_elevation_map.LidarElevationMap`` with the
training-time noise model left out (a demo wants the clean sensor). Kept identical:

* grid layout: ``idx = ix * num_y + iy``, x forward, centred on the root;
* value per cell: ``root_z - hit_z - offset``, highest return wins (an ``amin``);
* cells without a return this step keep their held value;
* motion compensation: the held grid is resampled (bilinear, zero padding, like
  ``grid_sample(align_corners=True)``) onto the current pose before new returns merge,
  with the vertical move added back to measured cells and ``flat_fill`` elsewhere.
"""

from __future__ import annotations

import numpy as np

from ..mathutils import yaw_of

_UNOBSERVED = 1.0e4


class LidarElevationMap:
    def __init__(
        self,
        resolution: float,
        size: tuple[float, float],
        offset: float,
        flat_fill: float,
        scanner_offset_xy: tuple[float, float] = (0.0, 0.0),
        motion_compensation: bool = True,
    ):
        self.resolution = float(resolution)
        self.offset = float(offset)
        self.flat_fill = float(flat_fill)
        self.motion_compensation = bool(motion_compensation)
        self.num_x = round(size[0] / resolution) + 1
        self.num_y = round(size[1] / resolution) + 1
        self.num_cells = self.num_x * self.num_y
        self.x0 = -size[0] / 2 + scanner_offset_xy[0]
        self.y0 = -size[1] / 2 + scanner_offset_xy[1]
        cx = np.linspace(self.x0, self.x0 + size[0], self.num_x)
        cy = np.linspace(self.y0, self.y0 + size[1], self.num_y)
        gx, gy = np.meshgrid(cx, cy, indexing="ij")
        self.cell_xy = np.stack([gx.ravel(), gy.ravel()], axis=-1)
        self.reset()

    def reset(self) -> None:
        self.hold = np.full(self.num_cells, self.flat_fill)
        self.measured = np.zeros(self.num_cells, dtype=bool)
        self.unobserved = np.ones(self.num_cells, dtype=bool)
        self._pose_valid = False
        self._prev_xy = np.zeros(2)
        self._prev_yaw = 0.0
        self._prev_z = 0.0

    def _bilinear(self, images: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        """Sample (C, num_x, num_y) at fractional indices with zero padding."""
        x0 = np.floor(fx).astype(np.int64)
        y0 = np.floor(fy).astype(np.int64)
        wx, wy = fx - x0, fy - y0
        out = np.zeros((images.shape[0], fx.size))
        for dx, dy, w in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)), (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < self.num_x) & (yi >= 0) & (yi < self.num_y)
            vals = images[:, np.clip(xi, 0, self.num_x - 1), np.clip(yi, 0, self.num_y - 1)]
            out += w * vals * ok
        return out

    def _advance_hold(self, xy: np.ndarray, z: float, yaw: float) -> None:
        if not self.motion_compensation:
            return
        if self._pose_valid:
            c, s = np.cos(self._prev_yaw), np.sin(self._prev_yaw)
            d = xy - self._prev_xy
            ux, uy = c * d[0] + s * d[1], -s * d[0] + c * d[1]
            dyaw = yaw - self._prev_yaw
            cd, sd = np.cos(dyaw), np.sin(dyaw)
            cell_x, cell_y = self.cell_xy[:, 0], self.cell_xy[:, 1]
            qx = cd * cell_x - sd * cell_y + ux
            qy = sd * cell_x + cd * cell_y + uy
            src = np.stack([self.hold, self.measured.astype(np.float64), np.ones(self.num_cells)])
            sampled = self._bilinear(
                src.reshape(3, self.num_x, self.num_y),
                (qx - self.x0) / self.resolution,
                (qy - self.y0) / self.resolution,
            )
            inside = sampled[2] > 0.999
            measured = inside & (sampled[1] > 0.5)
            self.hold = np.where(measured, sampled[0] + (z - self._prev_z), self.flat_fill)
            self.measured = measured
        self._prev_xy = xy.copy()
        self._prev_yaw = yaw
        self._prev_z = z
        self._pose_valid = True

    def update(self, hits_w: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
        """Merge one sweep. ``hits_w`` is (R, 3), non-finite rows are misses."""
        yaw = yaw_of(root_quat)
        self._advance_hold(root_pos[:2], float(root_pos[2]), yaw)

        finite = np.isfinite(hits_w).all(axis=-1)
        hits = np.where(finite[:, None], hits_w, 0.0)
        rel = hits - root_pos
        c, s = np.cos(yaw), np.sin(yaw)
        bx = c * rel[:, 0] + s * rel[:, 1]
        by = -s * rel[:, 0] + c * rel[:, 1]
        ix = np.round((bx - self.x0) / self.resolution).astype(np.int64)
        iy = np.round((by - self.y0) / self.resolution).astype(np.int64)
        valid = (ix >= 0) & (ix < self.num_x) & (iy >= 0) & (iy < self.num_y) & finite

        heights = root_pos[2] - hits[:, 2] - self.offset
        grid = np.full(self.num_cells, _UNOBSERVED)
        np.minimum.at(grid, ix[valid] * self.num_y + iy[valid], heights[valid])

        self.unobserved = grid >= _UNOBSERVED * 0.5
        grid = np.where(self.unobserved, self.hold, grid)
        self.hold = grid
        self.measured |= ~self.unobserved
        return grid.copy()

    def cell_positions_w(self, values: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
        """World positions of the cells as the map sees them, for debug drawing."""
        yaw = yaw_of(root_quat)
        c, s = np.cos(yaw), np.sin(yaw)
        x, y = self.cell_xy[:, 0], self.cell_xy[:, 1]
        return np.stack(
            [root_pos[0] + c * x - s * y, root_pos[1] + s * x + c * y, root_pos[2] - values - self.offset], axis=-1
        )
