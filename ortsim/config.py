"""Robot / policy / scene configs (``configs/**.yaml``) and what the runner derives from them.

Everything the runner needs about a policy comes from three places:

* ``deploy.yaml`` exported next to the ONNX at training time: observation terms with
  their scales, action scale, PD gains, default pose -- in the *articulation* joint
  order of the training sim;
* the policy config: which files, which robot, and whatever ``deploy.yaml`` does not
  carry (perception terms, sim timing, command ranges);
* the robot config: URDF, joint order, motor model.

``PolicySpec`` resolves all of that into per-joint arrays in *policy* order, so the
backends never touch deploy.yaml index maps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import REPO_ROOT
from .assets import resolve
from .yamlio import load_yaml

CONFIG_DIR = REPO_ROOT / "configs"


def _match_value(name: str, table: dict, default=None):
    """Value for ``name`` from a {regex: value} table (first full match wins)."""
    for pattern, value in table.items():
        if re.fullmatch(pattern, name):
            return value
    return default


def _motor_table(groups: list[dict], joints: list[str]) -> dict[str, MotorModel]:
    """``motors`` list ({joints: [regex...], X1: ..., armature: ...}) -> per-joint models."""
    table = {}
    for group in groups:
        model = MotorModel(**{k: v for k, v in group.items() if k != "joints"})
        for j in joints:
            if any(re.fullmatch(p, j) for p in group["joints"]):
                table[j] = model
    return table


# --------------------------------------------------------------------------------------
# Robot
# --------------------------------------------------------------------------------------


@dataclass
class MotorModel:
    """PD actuator with a torque-speed envelope (``UnitreeActuator`` in training).

    Torque limit is ``Y1`` (same direction as the velocity) / ``Y2`` (opposing) below
    ``X1`` and falls linearly to zero at ``X2``. ``Fs``/``Fd``/``Va`` are the actuator's
    own friction model; ``armature`` and ``friction`` are written to the physics joint.
    """

    X1: float = 1e9
    X2: float = 1e9
    Y1: float = 1e9
    Y2: float | None = None
    Fs: float = 0.0
    Fd: float = 0.0
    Va: float = 0.01
    armature: float = 0.0
    friction: float = 0.0
    velocity_limit: float | None = None
    """Joint speed limit given to the simulator (implicit PD only)."""


@dataclass
class RobotSpec:
    name: str
    urdf: Path | None
    """Training URDF; None when MuJoCo uses a vendor MJCF (``mujoco.mjcf``) and Isaac a USD."""
    base_body: str
    joints: list[str]
    """Articulation joint order of the training sim (what deploy.yaml indexes into)."""
    init_height: float
    init_joint_pos: dict[str, float]
    motors: dict[str, MotorModel]
    isaac: dict
    mujoco: dict
    camera: dict
    raw: dict
    implicit_pd: bool = False
    """Training solved the PD inside the simulator (Isaac Lab ImplicitActuator): the runner
    then sends joint position targets and each backend runs its own implicit PD, clipped at
    ``Y1``, instead of ortsim computing torques."""

    @classmethod
    def load(cls, name: str) -> "RobotSpec":
        raw = load_yaml(CONFIG_DIR / "robots" / f"{name}.yaml")
        joints = list(raw["joints"])
        motors = _motor_table(raw["motors"], joints)
        missing = [j for j in joints if j not in motors]
        if missing:
            raise ValueError(f"robot {name}: no motor model for joints {missing}")
        init = raw.get("init", {})
        return cls(
            name=name,
            urdf=resolve(raw["urdf"]) if raw.get("urdf") else None,
            base_body=raw["base_body"],
            joints=joints,
            init_height=float(init.get("height", 0.5)),
            init_joint_pos={j: float(_match_value(j, init.get("joint_pos", {}), 0.0)) for j in joints},
            motors=motors,
            isaac=raw.get("isaac", {}),
            mujoco=raw.get("mujoco", {}),
            camera=raw.get("camera", {}),
            raw=raw,
            implicit_pd=bool(raw.get("implicit_pd", False)),
        )


# --------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------


@dataclass
class ObsTerm:
    name: str
    scale: np.ndarray | float
    clip: tuple[float, float] | None
    history_length: int
    params: dict = field(default_factory=dict)


@dataclass
class CommandSpec:
    trained_ranges: dict[str, tuple[float, float]]
    limits: dict[str, tuple[float, float]]
    initial: tuple[float, float, float]
    accel: tuple[float, float, float]
    """While a key is held: how fast the command ramps toward its limit (per s)."""
    decel: tuple[float, float, float]
    """After release: how fast it eases back to the cruise / zero command (per s)."""


@dataclass
class PolicySpec:
    name: str
    description: str
    robot: RobotSpec
    onnx: Path
    sim_dt: float
    decimation: int
    joint_names: list[str]
    """Policy (action) joint order."""
    default_joint_pos: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    action_scale: np.ndarray
    action_offset: np.ndarray
    action_clip: np.ndarray | None
    obs_terms: list[ObsTerm]
    sensors: dict
    command: CommandSpec
    fall_tilt: float
    """tilt from upright [rad] counted as a fall (training's ``bad_orientation``)"""
    raw: dict

    @property
    def step_dt(self) -> float:
        return self.sim_dt * self.decimation

    @classmethod
    def load(cls, name: str) -> "PolicySpec":
        raw = load_yaml(CONFIG_DIR / "policies" / f"{name}.yaml")
        robot = RobotSpec.load(raw["robot"])
        if "deploy" in raw["files"]:
            deploy = load_yaml(resolve(raw["files"]["deploy"]))
        elif "deploy_inline" in raw:
            # Runs exported without deploy.yaml: import_policy rebuilt it from env.yaml.
            deploy = raw["deploy_inline"]
        else:
            raise ValueError(f"policy {name}: needs files.deploy or deploy_inline")
        onnx = resolve(raw["files"]["onnx"])
        if raw.get("motors"):
            # The motor parameters this policy was trained with win over the robot defaults.
            robot.motors.update(_motor_table(raw["motors"], robot.joints))

        art = robot.joints
        n = len(art)

        def per_art(values, what: str) -> dict[str, float]:
            values = list(np.broadcast_to(np.asarray(values, dtype=np.float64), (n,)))
            if len(values) != n:
                raise ValueError(f"deploy.yaml {what} has {len(values)} entries, robot has {n} joints")
            return dict(zip(art, values))

        default_by_name = per_art(deploy["default_joint_pos"], "default_joint_pos")
        kp, kd = deploy["stiffness"], deploy["damping"]
        if raw.get("deploy_gain_order", "articulation") == "sdk":
            # unitree_rl_lab's export_deploy_cfg writes stiffness / damping in SDK joint order
            # (stiffness[joint_ids_map] = articulation values) but default_joint_pos in
            # articulation order; joint_ids_map[i] is the SDK index of articulation joint i.
            ids = deploy["joint_ids_map"]
            kp, kd = [kp[i] for i in ids], [kd[i] for i in ids]
        kp_by_name = per_art(kp, "stiffness")
        kd_by_name = per_art(kd, "damping")

        if len(deploy["actions"]) != 1:
            raise NotImplementedError("only a single joint-position action term is supported")
        (action_name, action), = deploy["actions"].items()
        joint_names = _resolve_joint_names(action.get("joint_names"), art)
        idx = [art.index(j) for j in joint_names]
        if action.get("joint_ids") is not None and list(action["joint_ids"]) != idx:
            raise ValueError(
                f"deploy.yaml joint_ids {action['joint_ids']} disagree with the robot config's joint "
                f"order (expected {idx}); check configs/robots/{robot.name}.yaml 'joints'."
            )
        m = len(joint_names)

        def per_policy(value) -> np.ndarray:
            return np.broadcast_to(np.asarray(value, dtype=np.float64), (m,)).copy()

        default = np.array([default_by_name[j] for j in joint_names])
        act_cfg = raw.get("action", {})
        # deploy.yaml writes offset 0.0 even when the action used ``use_default_offset``,
        # so the policy config states which one training used.
        offset = default.copy() if act_cfg.get("offset", "default") == "default" else per_policy(act_cfg["offset"])
        clip = action.get("clip")
        clip = np.asarray(clip, dtype=np.float64) if clip is not None else None

        obs_cfg = raw.get("observation", {}) or {}
        replace = obs_cfg.get("replace_terms", {}) or {}
        obs_terms = []
        for term_name, term in deploy["observations"].items():
            # replace_terms says how to compute a deploy.yaml term the runner cannot infer from
            # its name (e.g. a LiDAR height map); deploy.yaml keeps its scale and clip.
            obs_terms.append(_obs_term(term_name, {**term, **replace.get(term_name, {})}))
        for extra in obs_cfg.get("extra_terms", []) or []:
            obs_terms.append(_obs_term(extra["name"], extra))

        sim = raw.get("sim", {})
        sim_dt = float(sim.get("dt", 0.005))
        decimation = int(sim.get("decimation", round(deploy["step_dt"] / sim_dt)))
        if abs(sim_dt * decimation - deploy["step_dt"]) > 1e-6:
            raise ValueError(f"sim dt {sim_dt} x decimation {decimation} != deploy step_dt {deploy['step_dt']}")

        cmd = raw.get("command", {})
        axes = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        trained = {a: tuple(cmd.get("trained_ranges", {}).get(a, (0.0, 0.0))) for a in axes}
        limits = {a: tuple(cmd.get("limits", {}).get(a, trained[a])) for a in axes}
        command = CommandSpec(
            trained_ranges=trained,
            limits=limits,
            initial=tuple(cmd.get("initial", (0.0, 0.0, 0.0))),
            accel=tuple(cmd.get("accel", (0.6, 0.6, 1.2))),
            decel=tuple(cmd.get("decel", (1.0, 1.0, 2.0))),
        )
        term = raw.get("termination", {})

        return cls(
            name=name,
            description=raw.get("description", ""),
            robot=robot,
            onnx=onnx,
            sim_dt=sim_dt,
            decimation=decimation,
            joint_names=joint_names,
            default_joint_pos=default,
            stiffness=np.array([kp_by_name[j] for j in joint_names]),
            damping=np.array([kd_by_name[j] for j in joint_names]),
            action_scale=per_policy(action.get("scale", 1.0)),
            action_offset=offset,
            action_clip=clip,
            obs_terms=obs_terms,
            sensors=raw.get("sensors", {}),
            command=command,
            fall_tilt=float(np.deg2rad(term.get("tilt_limit_deg", 60.0))),
            raw=raw,
        )


def _resolve_joint_names(names, art: list[str]) -> list[str]:
    if not names:
        return list(art)
    out: list[str] = []
    for pattern in names:
        matched = [j for j in art if re.fullmatch(pattern, j)]
        if not matched:
            raise ValueError(f"action joint pattern {pattern!r} matches no robot joint")
        out.extend(j for j in matched if j not in out)
    return out


def _obs_term(name: str, term: dict) -> ObsTerm:
    scale = term.get("scale", 1.0)
    scale = 1.0 if scale is None else (np.asarray(scale, dtype=np.float64) if isinstance(scale, list) else float(scale))
    clip = term.get("clip")
    return ObsTerm(
        name=name,
        scale=scale,
        clip=tuple(clip) if clip is not None else None,
        history_length=int(term.get("history_length", 1) or 1),
        params={k: v for k, v in term.items() if k not in ("scale", "clip", "history_length", "name")},
    )


# --------------------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------------------


@dataclass
class SceneSpec:
    id: str
    usd: Path | None
    """None for the built-in flat ground."""
    spawn_xy: tuple[float, float]
    spawn_yaw: float
    """radians"""
    spawn_z: float
    collision_exclude: list[str]
    friction: float
    description: str = ""
    use_scene_lights: bool = False
    """Isaac: keep the lights stored in the USD instead of the built-in light rig."""
    camera: str | None = None
    """Initial follow-camera view (a preset name, see ortsim.camera)."""
    up_axis: str | None = None
    """Override the USD's upAxis metadata (a file declaring Z while holding Y-up data)."""
    collision: str = "auto"
    """auto: use the USD's own colliders (UsdPhysics.CollisionAPI prims, visible or not) when
    it has any, else give every visible mesh a triangle-mesh collider. authored / generate
    force one of the two."""
    height_scan_ray_up: float | None = None
    """Indoor scenes: start top-down height-scan rays this far above the base instead of the
    training height (20 m), so a deck, roof or overhang above the robot is not read as the
    ground. The value is still base_z - hit_z - offset, as in training."""
    mujoco: dict = None
    """MuJoCo collision: {collision: convex | hfield, hfield_resolution, hfield_cap, crop,
    planar_thickness}. planar_thickness (metres, default 0) extrudes planar triangles
    into collision prisms in convex mode, extending below floors without filling holes.
    convex (default) decomposes non-convex meshes with CoACD; hfield rasterises the collision
    geometry from above into a height field -- the only workable option for large scans,
    exact for floors, steps and walls but filling in anything under an overhang (surfaces
    above ``hfield_cap`` are ignored, so a lower level can be kept walkable)."""

    @classmethod
    def load(cls, scene_id: str) -> "SceneSpec":
        if scene_id in ("flat", "builtin/flat"):
            return cls("builtin/flat", None, (0.0, 0.0), 0.0, 0.0, [], 1.0, "flat ground plane")
        if "/" not in scene_id:
            raise ValueError(f"scene id must look like '<set>/<scene>' (got {scene_id!r}); see configs/scenes/")
        set_name, scene_name = scene_id.split("/", 1)
        raw = load_yaml(CONFIG_DIR / "scenes" / f"{set_name}.yaml")
        if scene_name not in raw["scenes"]:
            raise KeyError(f"scene {scene_name!r} not in configs/scenes/{set_name}.yaml")
        merged = {**raw.get("defaults", {}), **(raw["scenes"][scene_name] or {})}
        spawn = {**raw.get("defaults", {}).get("spawn", {}), **merged.get("spawn", {})}
        return cls(
            id=scene_id,
            usd=resolve(merged["usd"]),
            spawn_xy=tuple(spawn.get("pos", (0.0, 0.0))[:2]),
            spawn_yaw=float(np.deg2rad(spawn.get("yaw_deg", 0.0))),
            spawn_z=float(spawn.get("ground_z", 0.0)),
            collision_exclude=list(merged.get("collision_exclude", [])),
            friction=float(merged.get("friction", 1.0)),
            description=merged.get("description", ""),
            use_scene_lights=bool(merged.get("use_scene_lights", False)),
            camera=merged.get("camera"),
            up_axis=merged.get("up_axis"),
            collision=merged.get("collision", "auto"),
            mujoco=dict(merged.get("mujoco") or {}),
            height_scan_ray_up=merged.get("height_scan_ray_up"),
        )

    def with_spawn(self, x: float, y: float, yaw_deg: float | None = None, ground_z: float | None = None) -> "SceneSpec":
        """A copy starting at (x, y); yaw and ground height default to the scene's."""
        return replace(self, spawn_xy=(x, y),
                       spawn_yaw=self.spawn_yaw if yaw_deg is None else float(np.deg2rad(yaw_deg)),
                       spawn_z=self.spawn_z if ground_z is None else ground_z)

    def is_excluded(self, prim_path: str) -> bool:
        leaf = prim_path.rstrip("/").split("/")[-1]
        return any(re.fullmatch(p, leaf) or re.fullmatch(p, prim_path) for p in self.collision_exclude)


def list_configs(kind: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / kind).glob("*.yaml"))
