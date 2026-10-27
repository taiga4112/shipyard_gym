"""Isaac Sim backend, built on Isaac Lab's asset / sim utilities (no manager-based env).

Import only after ``AppLauncher`` has started the app.
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
import omni.usd
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

from ..assets import resolve
from ..config import PolicySpec, SceneSpec
from ..mathutils import quat_from_yaw
from ..runner import RobotState
from ..scene_mesh import extract_meshes, has_authored_collision, merge, stage_correction

GROUND_PATH = "/World/ground"
COLLIDER_PATH = "/World/ground_collider"  # generated, double-sided (scenes without colliders)
ROBOT_PATH = "/World/Robot"


def _spawn_scene(scene: SceneSpec, device: str):
    """Spawn the scene and return the warp mesh the LiDAR casts against."""
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=scene.friction, dynamic_friction=scene.friction, restitution=0.0,
        friction_combine_mode="multiply", restitution_combine_mode="multiply",
    )
    stage = omni.usd.get_context().get_stage()
    if scene.usd is None:
        plane = sim_utils.GroundPlaneCfg(physics_material=material_cfg)
        plane.func(GROUND_PATH, plane)
        s = 100.0
        verts = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
    else:
        src = Usd.Stage.Open(str(scene.usd))  # the file's own metadata: units, up axis, colliders
        correction = stage_correction(src, scene.up_axis)
        authored = scene.collision == "authored" or (scene.collision == "auto" and has_authored_collision(src))
        del src
        prim = stage.DefinePrim(GROUND_PATH, "Xform")
        prim.GetReferences().AddReference(str(scene.usd))
        # Bring cm / Y-up exports to the Z-up metre stage (references are not rescaled).
        if not np.allclose(correction, np.eye(4)):
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(Gf.Matrix4d(*correction.ravel().tolist()))
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                # A scene is static ground; a dynamic body in it would fall away under the robot.
                UsdPhysics.RigidBodyAPI(p).CreateKinematicEnabledAttr().Set(True)
        material_path = "/World/PhysicsMaterials/ground"
        material_cfg.func(material_path, material_cfg)
        if authored:
            # The file's own colliders, hidden ones included (proxies for awkward visuals),
            # rebuilt double-sided: authored scan colliders have downward faces too (b03's inner
            # bottom plate: every face points down, and the robot sank through it).
            meshes = extract_meshes(stage, GROUND_PATH, exclude=scene.is_excluded, collision=True,
                                    correction=np.eye(4))
            for m in meshes:
                UsdPhysics.CollisionAPI(stage.GetPrimAtPath(m.path)).CreateCollisionEnabledAttr().Set(False)
            _add_double_sided_collider(stage, meshes, material_path)
        else:
            meshes = extract_meshes(stage, GROUND_PATH, exclude=scene.is_excluded, correction=np.eye(4))
            _add_double_sided_collider(stage, meshes, material_path)
        if not meshes:
            raise RuntimeError(f"scene {scene.id}: no collision geometry (collision={scene.collision})")
        verts, faces = merge(meshes)
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        print(f"[isaac] scene {scene.id}: {len(meshes)} {'authored' if authored else 'generated'} collision meshes, "
              f"{len(faces)} triangles, extent {np.round(lo, 2).tolist()} .. {np.round(hi, 2).tolist()} m", flush=True)

    if scene.usd is not None and not scene.use_scene_lights:
        # Blender exports carry their own lights (here a dome whose texture is missing);
        # a fixed rig keeps every scene equally lit for demo videos.
        for prim in Usd.PrimRange(stage.GetPrimAtPath(GROUND_PATH)):
            if prim.HasAPI(UsdLux.LightAPI):
                UsdLux.LightAPI(prim).CreateIntensityAttr().Set(0.0)
                UsdGeom.Imageable(prim).MakeInvisible()
    dome = sim_utils.DomeLightCfg(intensity=1200.0, color=(0.85, 0.9, 1.0))
    dome.func("/World/DomeLight", dome)
    sun = sim_utils.DistantLightCfg(intensity=2500.0, angle=2.0, color=(1.0, 0.97, 0.92))
    # Mostly overhead (tilted ~25 deg) so tall walls do not shade the whole course.
    sun.func("/World/SunLight", sun, orientation=(0.976, 0.13, 0.17, 0.0))
    return convert_to_warp_mesh(verts.astype(np.float32), faces.astype(np.int32), device=device)


def _add_double_sided_collider(stage, meshes, material_path: str) -> None:
    """One hidden triangle-mesh collider holding every triangle in both windings.

    Scan / CAD conversions have faces facing either way, and PhysX only pushes a body out
    along a triangle's front side: through a face whose normal points down the robot falls
    (measured: half the plates of the Kure DXF blocks, all of b03's inner bottom plate in
    its authored collider). Doubling the triangles makes the orientation irrelevant.
    """
    verts, faces = merge(meshes)
    both = np.concatenate([faces, faces[:, ::-1]])
    prim = UsdGeom.Mesh.Define(stage, COLLIDER_PATH)
    prim.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in verts])
    prim.CreateFaceVertexCountsAttr([3] * len(both))
    prim.CreateFaceVertexIndicesAttr(both.ravel().astype(int).tolist())
    UsdGeom.Imageable(prim).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr().Set(UsdPhysics.Tokens.none)
    sim_utils.bind_physics_material(COLLIDER_PATH, material_path)


def _robot_cfg(policy: PolicySpec, scene: SceneSpec) -> ArticulationCfg:
    robot = policy.robot
    iso = robot.isaac
    groups: dict[int, tuple] = {}
    for joint, motor in robot.motors.items():
        groups.setdefault(id(motor), (motor, []))[1].append(joint)
    actuators = {}
    kp = dict(zip(policy.joint_names, policy.stiffness))
    kd = dict(zip(policy.joint_names, policy.damping))
    for i, (motor, joints) in enumerate(groups.values()):
        if robot.implicit_pd:
            # As trained: PhysX solves the PD; ortsim only sends position targets.
            actuators[f"motor{i}"] = ImplicitActuatorCfg(
                joint_names_expr=joints,
                stiffness={j: float(kp[j]) for j in joints},
                damping={j: float(kd[j]) for j in joints},
                effort_limit_sim=float(motor.Y1),
                velocity_limit_sim=motor.velocity_limit,
                armature=motor.armature, friction=motor.friction,
            )
            continue
        # Zero gains: the PD and torque-speed envelope are computed by ortsim and sent
        # as effort, so both backends share one motor model.
        actuators[f"motor{i}"] = IdealPDActuatorCfg(
            joint_names_expr=joints, stiffness=0.0, damping=0.0, effort_limit=1.0e6, effort_limit_sim=1.0e6,
            armature=motor.armature, friction=motor.friction,
        )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
        max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=iso.get("enabled_self_collisions", True),
        solver_position_iteration_count=iso.get("solver_position_iteration_count", 4),
        solver_velocity_iteration_count=iso.get("solver_velocity_iteration_count", 0),
    )
    if iso.get("usd"):
        # The exact asset training spawned (UsdFileCfg); the URDF is then only for MuJoCo.
        spawn = sim_utils.UsdFileCfg(
            usd_path=str(resolve(iso["usd"])),
            activate_contact_sensors=False,
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        )
    else:
        spawn = sim_utils.UrdfFileCfg(
            asset_path=str(robot.urdf),
            fix_base=False,
            merge_fixed_joints=iso.get("merge_fixed_joints", False),
            self_collision=iso.get("self_collision", False),
            collider_type=iso.get("collider_type", "convex_hull"),
            replace_cylinders_with_capsules=iso.get("replace_cylinders_with_capsules", True),
            make_instanceable=True,
            activate_contact_sensors=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force", target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        )
    x, y = scene.spawn_xy
    return ArticulationCfg(
        prim_path=ROBOT_PATH,
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(x, y, scene.spawn_z + robot.init_height),
            rot=tuple(quat_from_yaw(scene.spawn_yaw)),
            joint_pos=dict(robot.init_joint_pos),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators=actuators,
    )


class IsaacBackend:
    def __init__(self, policy: PolicySpec, scene: SceneSpec, device: str = "cuda:0"):
        self.policy = policy
        self.scene = scene
        self.device = device
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=policy.sim_dt,
                render_interval=policy.decimation,
                device=device,
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            )
        )
        self.ray_mesh = _spawn_scene(scene, device)
        self.robot = Articulation(_robot_cfg(policy, scene))
        self._extra_init = []

    def add_post_init(self, fn) -> None:
        """Run ``fn`` after ``sim.reset()`` (sensors / markers that need a playing sim)."""
        self._extra_init.append(fn)

    def start(self) -> None:
        self.sim.reset()
        names = list(self.robot.joint_names)
        if names != self.policy.robot.joints:
            print(f"[isaac] warning: articulation joint order {names} != robot config 'joints'", flush=True)
        self._ids = torch.tensor([names.index(j) for j in self.policy.joint_names], device=self.device)
        self._ids_list = self._ids.tolist()
        for fn in self._extra_init:
            fn()
        self.reset_robot()

    # --- runner interface -------------------------------------------------------------

    def reset_robot(self, pose=None) -> None:
        r = self.robot
        root = r.data.default_root_state.clone()
        if pose is not None:
            root[0, :3] = torch.tensor([pose.x, pose.y, pose.ground_z + self.policy.robot.init_height], device=self.device)
            root[0, 3:7] = torch.tensor(quat_from_yaw(pose.yaw), dtype=torch.float32, device=self.device)
        r.write_root_pose_to_sim(root[:, :7])
        r.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
        r.write_joint_state_to_sim(r.data.default_joint_pos.clone(), torch.zeros_like(r.data.default_joint_vel))
        r.set_joint_effort_target(torch.zeros_like(r.data.default_joint_pos))
        r.set_joint_position_target(r.data.default_joint_pos.clone())
        r.write_data_to_sim()
        r.reset()
        self.sim.forward()
        r.update(0.0)

    def get_state(self) -> RobotState:
        d = self.robot.data
        return RobotState(
            base_pos=d.root_pos_w[0].cpu().numpy().astype(np.float64),
            base_quat=d.root_quat_w[0].cpu().numpy().astype(np.float64),
            lin_vel_w=d.root_lin_vel_w[0].cpu().numpy().astype(np.float64),
            ang_vel_w=d.root_ang_vel_w[0].cpu().numpy().astype(np.float64),
            joint_pos=d.joint_pos[0, self._ids].cpu().numpy().astype(np.float64),
            joint_vel=d.joint_vel[0, self._ids].cpu().numpy().astype(np.float64),
        )

    def get_joint_state(self):
        d = self.robot.data
        return (
            d.joint_pos[0, self._ids].cpu().numpy().astype(np.float64),
            d.joint_vel[0, self._ids].cpu().numpy().astype(np.float64),
        )

    def apply_joint_torque(self, tau: np.ndarray) -> None:
        t = torch.as_tensor(tau, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.robot.set_joint_effort_target(t, joint_ids=self._ids_list)
        self.robot.write_data_to_sim()

    def set_joint_position_target(self, q: np.ndarray) -> None:
        t = torch.as_tensor(q, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.robot.set_joint_position_target(t, joint_ids=self._ids_list)
        self.robot.write_data_to_sim()

    def physics_step(self) -> None:
        self.sim.step(render=False)
        self.robot.update(self.policy.sim_dt)

    def raycast(self, origin: np.ndarray, directions: np.ndarray, max_distance: float) -> np.ndarray:
        starts = torch.as_tensor(np.broadcast_to(origin, directions.shape).copy(), dtype=torch.float32, device=self.device)
        dirs = torch.as_tensor(directions, dtype=torch.float32, device=self.device)
        hits = raycast_mesh(starts, dirs, self.ray_mesh, max_dist=max_distance)[0]
        return hits.cpu().numpy().astype(np.float64)

    def raycast_points(self, starts: np.ndarray, directions: np.ndarray, max_distance: float) -> np.ndarray:
        s = torch.as_tensor(np.ascontiguousarray(starts), dtype=torch.float32, device=self.device)
        d = torch.as_tensor(np.ascontiguousarray(directions), dtype=torch.float32, device=self.device)
        return raycast_mesh(s, d, self.ray_mesh, max_dist=max_distance)[0].cpu().numpy().astype(np.float64)

    def render(self) -> None:
        self.sim.render()
