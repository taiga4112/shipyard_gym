"""MuJoCo backend: robot from the training URDF (or a vendor MJCF), scene from the same USD meshes."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from .. import REPO_ROOT
from ..assets import resolve
from ..config import PolicySpec, SceneSpec
from ..mathutils import quat_from_yaw, quat_to_matrix
from ..runner import RobotState

SCENE_GROUP = 2  # scene visuals (rays are cast with warp against the collision geometry)
ROBOT_VISUAL_GROUP = 1
ROBOT_COLLISION_GROUP = 3  # hidden by default


def _scene_cache(scene: SceneSpec) -> Path:
    from ..tools.build_mujoco_scene import cache_path

    out = cache_path(scene)
    if not out.exists():
        cmd = [sys.executable, "-m", "ortsim.tools.build_mujoco_scene", scene.id]
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return out


def _mjcf_robot_spec(policy: PolicySpec) -> mujoco.MjSpec:
    """A vendor MJCF, aligned with what Isaac simulates.

    Isaac's explicit actuators have no joint damping or friction loss and the robot has no
    physics material of its own, so the MJCF's joint damping / frictionloss are replaced by
    the motor config and its contact friction (often a higher-priority foot value) by 1.0.
    Its actuators and keyframes are dropped: ortsim adds one torque motor per joint.
    """
    robot = policy.robot
    spec = mujoco.MjSpec.from_file(str(resolve(robot.mujoco["mjcf"])))
    for act in list(spec.actuators):
        spec.delete(act)
    for key in list(spec.keys):
        spec.delete(key)
    base = spec.body(robot.base_body)
    free = [j for j in base.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
    if free:
        free[0].name = "root"
    else:
        base.add_freejoint(name="root")
    for geom in spec.geoms:
        if geom.contype == 0 and geom.conaffinity == 0:
            geom.group = ROBOT_VISUAL_GROUP
            continue
        geom.group = ROBOT_COLLISION_GROUP
        geom.friction = [1.0, geom.friction[1], geom.friction[2]]
        geom.priority = 0
    for name in policy.joint_names:
        joint = spec.joint(name)
        joint.damping = np.zeros_like(np.asarray(joint.damping, dtype=float))
    _add_motors(spec, policy)
    return spec


def _robot_spec(policy: PolicySpec) -> mujoco.MjSpec:
    robot = policy.robot
    if robot.mujoco.get("mjcf"):
        return _mjcf_robot_spec(policy)
    urdf = robot.urdf
    text = urdf.read_text()
    meshdir = urdf.parent / "meshes"
    ext = f'<mujoco><compiler meshdir="{meshdir}" discardvisual="false" fusestatic="false"/></mujoco>'
    # Vendor URDFs sometimes carry their own <mujoco> block (meshdir etc.); ours replaces it.
    text = re.sub(r"<mujoco>.*?</mujoco>", "", text, flags=re.S)
    text = re.sub(r"(<robot[^>]*>)", lambda m: m.group(1) + ext, text, count=1)
    text = text.replace('filename="meshes/', 'filename="')
    # MuJoCo's URDF parser rejects repeated <material> definitions (common in vendor URDFs);
    # they only colour the visuals, which get a uniform colour below.
    text = re.sub(r"<material\b[^>]*/>|<material\b[^>]*>.*?</material>", "", text, flags=re.S)
    spec = mujoco.MjSpec.from_string(text)

    base = spec.body(robot.base_body)
    base.add_freejoint(name="root")

    capsules = robot.mujoco.get("replace_cylinders_with_capsules", True)
    for geom in spec.geoms:
        if geom.contype == 0 and geom.conaffinity == 0:
            geom.group = ROBOT_VISUAL_GROUP
            geom.rgba = robot.mujoco.get("rgba", [0.6, 0.6, 0.62, 1.0])
            continue
        geom.group = ROBOT_COLLISION_GROUP
        if capsules and geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE

    _add_motors(spec, policy)
    return spec


def _add_motors(spec: mujoco.MjSpec, policy: PolicySpec) -> None:
    """One actuator per policy joint: a torque motor (ortsim computes the PD) or, for
    ``implicit_pd`` robots, a position servo with the policy's gains and torque limit, whose
    damping the implicitfast integrator treats implicitly like PhysX's implicit drive."""
    robot = policy.robot
    for name, kp, kd in zip(policy.joint_names, policy.stiffness, policy.damping):
        motor = robot.motors[name]
        joint = spec.joint(name)
        joint.armature = motor.armature
        joint.frictionloss = motor.friction
        act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.ctrllimited = 0
        if robot.implicit_pd:
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            gain = np.zeros(10)
            gain[0] = kp
            bias = np.zeros(10)
            bias[1], bias[2] = -kp, -kd
            act.gainprm, act.biasprm = gain, bias
            if motor.Y1 < 1e8:
                act.forcelimited = 1
                act.forcerange = [-motor.Y1, motor.Y1]


def _scene_data(scene: SceneSpec):
    data = np.load(_scene_cache(scene))
    return data, json.loads(str(data["info"]))


def _scene_ray_mesh(scene: SceneSpec):
    """The geometry the scene's rays may hit (everything but excluded prims), for warp."""
    from ..raymesh import RayMesh
    from ..scene_mesh import MeshData, merge

    if scene.usd is None:
        s = 100.0
        verts = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float64)
        return RayMesh(verts, np.array([[0, 1, 2], [0, 2, 3]]))
    data, info = _scene_data(scene)
    if "ray" in info:
        meshes = [MeshData(r["path"], data[f"r{i}_verts"], data[f"r{i}_faces"], (0, 0, 0))
                  for i, r in enumerate(info["ray"])]
    else:  # caches from before the ray meshes were stored: the visible, non-excluded meshes
        meshes = [MeshData(v["path"], data[f"v{i}_verts"], data[f"v{i}_faces"], (0, 0, 0))
                  for i, v in enumerate(info["visual"]) if not v.get("excluded")]
    return RayMesh(*merge(meshes))


def _add_scene(spec: mujoco.MjSpec, scene: SceneSpec) -> None:
    world = spec.worldbody
    spec.add_texture(
        name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.55, 0.65, 0.8], rgb2=[0.1, 0.1, 0.15], width=256, height=256,
    )
    for direction, diffuse, shadow in (([0.3, 0.2, -1], 0.7, True), ([-0.3, -0.2, -1], 0.3, False)):
        light = world.add_light(pos=[0, 0, 10], dir=direction, diffuse=[diffuse] * 3, castshadow=shadow)
        if hasattr(mujoco, "mjtLightType"):  # MuJoCo >= 3.3
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        else:
            light.directional = True

    if scene.usd is None:
        # Checker floor so walking reads in videos (a uniform plane hides all motion).
        spec.add_texture(
            name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.42, 0.44, 0.47], rgb2=[0.32, 0.34, 0.37], mark=mujoco.mjtMark.mjMARK_EDGE,
            markrgb=[0.75, 0.75, 0.75], width=512, height=512,
        )
        mat = spec.add_material(name="grid", texrepeat=[1, 1], texuniform=True, reflectance=0.05)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
        world.add_geom(
            name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1], material="grid",
            group=SCENE_GROUP, friction=[scene.friction, 0.005, 0.0001],
        )
        return

    data, info = _scene_data(scene)
    for i, v in enumerate(info["visual"]):
        name = f"vis{i}"
        spec.add_mesh(name=name, uservert=data[f"v{i}_verts"].ravel().tolist(),
                      userface=data[f"v{i}_faces"].ravel().tolist(), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        rgba = [*v["color"], 1.0]
        # Excluded prims stay visible (group 0) but out of the LiDAR's group.
        world.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=rgba,
                       contype=0, conaffinity=0, group=0 if v.get("excluded") else SCENE_GROUP)
    hf = info.get("hfield")
    if hf:
        # Large scans: the collision geometry rasterised from above (see build_mujoco_scene).
        rx, ry = (hf["nx"] - 1) * hf["res"] / 2, (hf["ny"] - 1) * hf["res"] / 2
        spec.add_hfield(name="scene_hfield", size=[rx, ry, hf["z1"] - hf["z0"], 0.5],
                        nrow=hf["ny"], ncol=hf["nx"], userdata=data["hfield"].ravel().tolist())
        world.add_geom(name="scene_hfield", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="scene_hfield",
                       pos=[hf["x0"] + rx, hf["y0"] + ry, hf["z0"]], rgba=[0.8, 0.3, 0.3, 0.4],
                       group=4, friction=[scene.friction, 0.005, 0.0001])
    for c in info["collision"]:
        k = c["key"]
        spec.add_mesh(name=k, uservert=data[f"{k}_verts"].ravel().tolist(),
                      userface=data[f"{k}_faces"].ravel().tolist(), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        # Collision parts are hidden (group 4) and kept out of the LiDAR: rays hit the
        # exact visual triangles instead of the convex pieces.
        world.add_geom(name=k, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=k, rgba=[0.8, 0.3, 0.3, 0.4],
                       group=4, friction=[scene.friction, 0.005, 0.0001])


HEAD_CAMERA = "head_camera"


def _add_head_camera(spec: mujoco.MjSpec, policy: PolicySpec) -> None:
    """A forward-looking camera fixed to the base (``camera.head`` in the robot config).

    ``pos`` is in the base frame (x forward, z up), ``pitch_deg`` tilts it down, ``fovy``
    is the vertical field of view. This is what the remote controller streams.
    """
    head = policy.robot.camera.get("head") or {}
    pos = head.get("pos", [0.3, 0.0, 0.05])
    pitch = np.radians(float(head.get("pitch_deg", 10.0)))
    # MuJoCo cameras look along -z with +y up; turn that to look along base +x, then pitch down.
    look_x = np.array([0.5, 0.5, -0.5, -0.5])
    tilt = np.array([np.cos(pitch / 2), -np.sin(pitch / 2), 0.0, 0.0])  # about camera x, nose down
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, look_x, tilt)
    spec.body(policy.robot.base_body).add_camera(
        name=HEAD_CAMERA, pos=pos, quat=quat, fovy=float(head.get("fovy", 70.0))
    )


class MujocoBackend:
    def __init__(self, policy: PolicySpec, scene: SceneSpec):
        self.policy = policy
        self.scene = scene
        spec = _robot_spec(policy)
        _add_head_camera(spec, policy)
        _add_scene(spec, scene)
        spec.option.timestep = policy.sim_dt
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.visual.global_.offwidth = 1920
        spec.visual.global_.offheight = 1080
        spec.visual.quality.shadowsize = 4096
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.base_body = self.model.body(policy.robot.base_body).id
        root = self.model.joint("root")
        self._qpos_root = root.qposadr[0]
        self._qvel_root = root.dofadr[0]
        joints = [self.model.joint(j) for j in policy.joint_names]
        self._qpos_idx = np.array([j.qposadr[0] for j in joints])
        self._qvel_idx = np.array([j.dofadr[0] for j in joints])
        self._act_idx = np.array([self.model.actuator(j).id for j in policy.joint_names])
        self._ray_mesh = None  # warp mesh, built on first raycast_points()
        self.reset_robot()

    # --- runner interface -------------------------------------------------------------

    def reset_robot(self, pose=None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        robot = self.policy.robot
        qr = self._qpos_root
        if pose is None:
            x, y, z, yaw = *self.scene.spawn_xy, self.scene.spawn_z, self.scene.spawn_yaw
        else:
            x, y, z, yaw = pose.x, pose.y, pose.ground_z, pose.yaw
        self.data.qpos[qr : qr + 3] = [x, y, z + robot.init_height]
        self.data.qpos[qr + 3 : qr + 7] = quat_from_yaw(yaw)
        self.data.qpos[self._qpos_idx] = [robot.init_joint_pos[j] for j in self.policy.joint_names]
        mujoco.mj_forward(self.model, self.data)

    def get_state(self) -> RobotState:
        qr, vr = self._qpos_root, self._qvel_root
        quat = self.data.qpos[qr + 3 : qr + 7].copy()
        quat /= np.linalg.norm(quat)
        ang_local = self.data.qvel[vr + 3 : vr + 6]
        return RobotState(
            base_pos=self.data.qpos[qr : qr + 3].copy(),
            base_quat=quat,
            lin_vel_w=self.data.qvel[vr : vr + 3].copy(),
            ang_vel_w=quat_to_matrix(quat) @ ang_local,
            joint_pos=self.data.qpos[self._qpos_idx].copy(),
            joint_vel=self.data.qvel[self._qvel_idx].copy(),
        )

    def get_joint_state(self):
        return self.data.qpos[self._qpos_idx].copy(), self.data.qvel[self._qvel_idx].copy()

    def set_joint_position_target(self, q: np.ndarray) -> None:
        self.data.ctrl[self._act_idx] = q

    def apply_joint_torque(self, tau: np.ndarray) -> None:
        self.data.ctrl[self._act_idx] = tau

    def physics_step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def raycast_points(self, starts: np.ndarray, directions: np.ndarray, max_distance: float) -> np.ndarray:
        if self._ray_mesh is None:
            self._ray_mesh = _scene_ray_mesh(self.scene)
        return self._ray_mesh.cast(starts, directions, max_distance)

    def raycast(self, origin: np.ndarray, directions: np.ndarray, max_distance: float) -> np.ndarray:
        # The same warp mesh as raycast_points / Isaac: the scene's collision geometry.
        return self.raycast_points(np.broadcast_to(origin, directions.shape), directions, max_distance)
