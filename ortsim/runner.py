"""Backend-independent control loop: state -> observation -> ONNX -> PD torques."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .actuator import TorqueSpeedPD
from .assets import resolve
from .command import VelocityCommand
from .config import ObsTerm, PolicySpec
from .mathutils import quat_rotate_inverse, yaw_of
from .perception.elevation_map import LidarElevationMap
from .perception.lidar_accum import LidarAccumMap
from .perception.livox import RollingLivoxPattern
from .policy import OnnxPolicy


@dataclass
class RobotState:
    base_pos: np.ndarray
    base_quat: np.ndarray
    """(w, x, y, z), world frame"""
    lin_vel_w: np.ndarray
    ang_vel_w: np.ndarray
    joint_pos: np.ndarray
    """policy joint order"""
    joint_vel: np.ndarray


class Backend(Protocol):
    def get_state(self) -> RobotState: ...
    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]: ...
    def apply_joint_torque(self, tau: np.ndarray) -> None: ...
    def physics_step(self) -> None: ...
    def raycast(self, origin: np.ndarray, directions: np.ndarray, max_distance: float) -> np.ndarray: ...
    def raycast_points(self, starts: np.ndarray, directions: np.ndarray, max_distance: float) -> np.ndarray: ...
    def reset_robot(self, pose=None) -> None: ...
    def set_joint_position_target(self, q: np.ndarray) -> None: ...


def _grid_scan_layout(p: dict) -> dict:
    """Ray offsets (yaw frame) of an Isaac Lab GridPatternCfg, optionally k x k rays per cell.

    Points follow ``ordering="xy"`` (x inner, y outer); the sub-rays of one cell are
    consecutive, offset by ``((arange(k) + 0.5) / k - 0.5) * resolution`` in x and y.
    """
    res = float(p["resolution"])
    sx, sy = p["size"]
    nx, ny = int(round(sx / res)) + 1, int(round(sy / res)) + 1
    xs = np.linspace(-sx / 2, sx / 2, nx) + p.get("center_ahead_x", 0.0)
    ys = np.linspace(-sy / 2, sy / 2, ny) + p.get("center_left_y", 0.0)
    gx, gy = np.meshgrid(xs, ys, indexing="xy" if p.get("ordering", "xy") == "xy" else "ij")
    points = np.stack([gx.ravel(), gy.ravel()], axis=1)
    k = int(p.get("sub", 1))
    o = ((np.arange(k) + 0.5) / k - 0.5) * res
    ox, oy = np.meshgrid(o, o, indexing="ij")
    sub = np.stack([ox.ravel(), oy.ravel()], axis=1)
    local = (points[:, None, :] + sub[None]).reshape(-1, 2)
    return {"local_xy": local, "cell_xy": points, "sub": k, "offset": float(p.get("offset", 0.0)),
            "ray_up": float(p.get("ray_up", 20.0)), "max_distance": float(p.get("max_distance", 1.0e6))}


class _History:
    def __init__(self, length: int):
        self.length = length
        self.buffer: list[np.ndarray] = []

    def push(self, x: np.ndarray) -> np.ndarray:
        if self.length <= 1:
            return x
        if not self.buffer:
            self.buffer = [x] * self.length
        else:
            self.buffer = self.buffer[1:] + [x]
        return np.concatenate(self.buffer)

    def clear(self) -> None:
        self.buffer = []


class PolicyRunner:
    def __init__(self, spec: PolicySpec, backend: Backend):
        self.spec = spec
        self.backend = backend
        self.policy = OnnxPolicy(spec.onnx)
        self.actuator = TorqueSpeedPD(spec)
        self.command = VelocityCommand(spec.command)
        self.last_action = np.zeros(len(spec.joint_names))
        self.q_des = spec.default_joint_pos.copy()
        self._histories = {t.name: _History(t.history_length) for t in spec.obs_terms}
        self._joint_index = {j: i for i, j in enumerate(spec.joint_names)}
        self.sensors: dict[str, RollingLivoxPattern] = {}
        self.maps: dict[str, LidarElevationMap] = {}
        self.last_hits: np.ndarray | None = None
        self.last_map: np.ndarray | None = None
        self.grids: dict[str, dict] = {}
        self.accum: dict[str, LidarAccumMap] = {}
        self.last_grid: tuple[ObsTerm, np.ndarray] | None = None
        """(term, per-cell value before clip) of the latest grid height scan, for drawing"""
        self._build_perception()
        self.step_count = 0
        self.episode_steps = 0
        """control steps since the last reset (gait phase clock)"""

        if self.policy.action_dim != len(spec.joint_names):
            raise ValueError(f"ONNX outputs {self.policy.action_dim} actions, config has {len(spec.joint_names)} joints")

    # ----------------------------------------------------------------------------------

    def _build_perception(self) -> None:
        for term in self.spec.obs_terms:
            if term.params.get("type") == "grid_height_scan":
                self.grids[term.name] = _grid_scan_layout(term.params)
            if term.params.get("type") == "lidar_accum_map":
                self.accum[term.name] = LidarAccumMap(term.params)
            if term.params.get("type") != "lidar_elevation_map":
                continue
            p = term.params
            sensor_name = p["sensor"]
            if sensor_name not in self.sensors:
                s = self.spec.sensors[sensor_name]
                if s.get("type") != "rolling_livox":
                    raise NotImplementedError(f"sensor type {s.get('type')!r}")
                self.sensors[sensor_name] = RollingLivoxPattern(
                    resolve(s["pattern"]),
                    s["mount_pos"],
                    s["mount_rot_wxyz"],
                    s["samples_per_step"],
                    s.get("downsample", 1),
                    s.get("max_distance", 20.0),
                    s.get("rolling_window_start", 0),
                )
            self.maps[term.name] = LidarElevationMap(
                p["resolution"],
                tuple(p["size"]),
                p["offset"],
                p["flat_fill"],
                tuple(p.get("scanner_offset_xy", (0.0, 0.0))),
                p.get("motion_compensation", True),
            )

    def _joint_order(self, term: ObsTerm) -> list[int]:
        names = (term.params.get("params") or {}).get("asset_cfg", {}) or {}
        names = names.get("joint_names") if isinstance(names, dict) else None
        names = names or self.spec.robot.joints
        return [self._joint_index[n] for n in names]

    def _term_value(self, term: ObsTerm, state: RobotState) -> np.ndarray:
        q = state.base_quat
        name = term.params.get("type", term.name)
        if name == "base_lin_vel":
            return quat_rotate_inverse(q, state.lin_vel_w)
        if name == "base_ang_vel":
            return quat_rotate_inverse(q, state.ang_vel_w)
        if name == "projected_gravity":
            return quat_rotate_inverse(q, np.array([0.0, 0.0, -1.0]))
        if name in ("velocity_commands", "generated_commands", "keyboard_velocity_commands"):
            return self.command.value.copy()
        if name == "joint_pos_rel":
            idx = self._joint_order(term)
            return state.joint_pos[idx] - self.spec.default_joint_pos[idx]
        if name == "joint_vel_rel":
            return state.joint_vel[self._joint_order(term)]
        if name == "last_action":
            return self.last_action.copy()
        if name == "gait_phase":
            p = term.params
            if np.linalg.norm(self.command.value) < p.get("stand_threshold", 0.1):
                return np.zeros(2)
            g = (self.episode_steps * self.spec.step_dt % p["period"]) / p["period"]
            return np.array([np.sin(2 * np.pi * g), np.cos(2 * np.pi * g)])
        if name == "grid_height_scan":
            return self._grid_height_scan(term, state)
        if name == "lidar_accum_map":
            amap = self.accum[term.name]
            starts, dirs = amap.rays(state.base_pos, state.base_quat)
            hits = self.backend.raycast_points(starts, dirs, amap.max_distance)
            self.last_hits = hits
            return amap.update(hits, state.base_pos, state.base_quat)
        if name == "lidar_elevation_map":
            sensor = self.sensors[term.params["sensor"]]
            origin, dirs = sensor.next_rays(state.base_pos, state.base_quat)
            hits = self.backend.raycast(origin, dirs, sensor.max_distance)
            self.last_hits = hits
            self.last_map = self.maps[term.name].update(hits, state.base_pos, state.base_quat)
            return self.last_map
        raise NotImplementedError(f"observation term {name!r} is not implemented in ortsim.runner")

    def _grid_height_scan(self, term: ObsTerm, state: RobotState) -> np.ndarray:
        """Isaac Lab ``height_scan`` from a yaw-aligned top-down RayCaster grid.

        ``base_z - hit_z - offset`` per ray; with ``sub > 1`` each cell keeps the smallest
        value over its sub x sub rays (the highest point), ignoring misses, and a cell with no
        hit at all reads -inf (clipped by the term's clip).
        """
        g = self.grids[term.name]
        yaw = float(np.arctan2(2 * (state.base_quat[0] * state.base_quat[3] + state.base_quat[1] * state.base_quat[2]),
                               1 - 2 * (state.base_quat[2] ** 2 + state.base_quat[3] ** 2)))
        c, s = np.cos(yaw), np.sin(yaw)
        local = g["local_xy"]
        starts = np.empty((len(local), 3))
        starts[:, 0] = state.base_pos[0] + c * local[:, 0] - s * local[:, 1]
        starts[:, 1] = state.base_pos[1] + s * local[:, 0] + c * local[:, 1]
        scene = getattr(self.backend, "scene", None)
        ray_up = g["ray_up"]
        if scene is not None and scene.height_scan_ray_up is not None:
            ray_up = min(ray_up, float(scene.height_scan_ray_up))
        starts[:, 2] = state.base_pos[2] + ray_up
        dirs = np.broadcast_to(np.array([0.0, 0.0, -1.0]), starts.shape)
        hits = self.backend.raycast_points(starts, dirs, g["max_distance"])
        values = state.base_pos[2] - hits[:, 2] - g["offset"]
        values = np.where(np.isfinite(values), values, np.inf).reshape(-1, g["sub"] ** 2)
        cell = values.min(axis=1)
        self.last_hits = hits
        cell = np.where(np.isfinite(cell), cell, -np.inf)
        self.last_grid = (term, cell)
        return cell

    def map_points(self, state: RobotState) -> tuple[np.ndarray, np.ndarray] | None:
        """World positions of the height-map cells the policy saw, for debug drawing.

        Returns ``(points, flagged)``: for a LiDAR elevation map ``flagged`` marks cells held
        from earlier scans; for an accumulated LiDAR map, cells never seen; for a grid height scan it marks cells that hit nothing or were
        clipped (the point is drawn at the clipped height the policy actually got).
        """
        if self.maps and self.last_map is not None:
            hmap = next(iter(self.maps.values()))
            return hmap.cell_positions_w(self.last_map, state.base_pos, state.base_quat), hmap.unobserved
        if self.accum:
            amap = next(iter(self.accum.values()))
            return amap.cell_positions_w(state.base_pos, state.base_quat), amap.unseen
        if self.last_grid is None:
            return None
        term, cell = self.last_grid
        g = self.grids[term.name]
        seen = np.clip(cell, *term.clip) if term.clip is not None else np.where(np.isfinite(cell), cell, 0.0)
        flagged = seen != cell
        yaw = yaw_of(state.base_quat)
        c, s = np.cos(yaw), np.sin(yaw)
        x, y = g["cell_xy"][:, 0], g["cell_xy"][:, 1]
        pts = np.stack([state.base_pos[0] + c * x - s * y, state.base_pos[1] + s * x + c * y,
                        state.base_pos[2] - seen - g["offset"]], axis=-1)
        return pts, flagged

    def compute_obs(self, state: RobotState) -> np.ndarray:
        parts = []
        for term in self.spec.obs_terms:
            x = np.asarray(self._term_value(term, state), dtype=np.float64)
            if term.clip is not None:
                x = np.clip(x, term.clip[0], term.clip[1])
            x = x * term.scale
            parts.append(self._histories[term.name].push(x))
        obs = np.concatenate(parts)
        if obs.shape[0] != self.policy.obs_dim:
            sizes = ", ".join(f"{t.name}={p.shape[0]}" for t, p in zip(self.spec.obs_terms, parts))
            raise ValueError(f"observation has {obs.shape[0]} dims but the ONNX expects {self.policy.obs_dim} ({sizes})")
        return obs

    # ----------------------------------------------------------------------------------

    def reset(self, pose=None) -> None:
        """Respawn the robot (``pose``: an ``ortsim.reborn.Pose``, None = scene spawn).

        ``step_count`` keeps running: it is the session clock used for --duration and
        the video overlay.
        """
        self.backend.reset_robot(pose)
        self.policy.reset()
        self.last_action[:] = 0.0
        self.q_des = self.spec.default_joint_pos.copy()
        for h in self._histories.values():
            h.clear()
        for m in self.maps.values():
            m.reset()
        for m in self.accum.values():
            m.reset()
        self.episode_steps = 0

    def ground_below(self, state: RobotState) -> float:
        """Scene height straight below the root (inf if nothing within 5 m)."""
        hit = self.backend.raycast(state.base_pos, np.array([[0.0, 0.0, -1.0]]), 5.0)[0]
        return float(hit[2]) if np.isfinite(hit).all() else float("inf")

    def step(self) -> RobotState:
        """One control step (``decimation`` physics steps). Returns the state the policy saw."""
        state = self.backend.get_state()
        obs = self.compute_obs(state)
        action = self.policy(obs)
        self.last_action = action
        q_des = self.spec.action_offset + self.spec.action_scale * action
        if self.spec.action_clip is not None:
            q_des = np.clip(q_des, self.spec.action_clip[:, 0], self.spec.action_clip[:, 1])
        self.q_des = q_des
        if self.spec.robot.implicit_pd:
            self.backend.set_joint_position_target(q_des)
            for _ in range(self.spec.decimation):
                self.backend.physics_step()
        else:
            for _ in range(self.spec.decimation):
                q, qd = self.backend.get_joint_state()
                self.backend.apply_joint_torque(self.actuator(q_des, q, qd))
                self.backend.physics_step()
        self.step_count += 1
        self.episode_steps += 1
        return state
