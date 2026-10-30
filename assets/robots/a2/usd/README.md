# A2 robot USD (the asset A2-Terrain-SGS-v1 was trained with)

The USD for spawning the A2 in Isaac Sim / Isaac Lab. The A2-Terrain-SGS-v1 policy was trained and evaluated with this USD.
In this repository it lives in `assets/robots/a2/usd/`.

## Contents

```
a2.usd                         entry file (loads the four files in configuration/; keep them alongside it)
  md5 b4d456bc1dc121e37646283285ed64b9
configuration/a2_base.usd      geometry (visuals and collisions)  md5 c0fc759a787f79f0391cb21c0281ad36
configuration/a2_physics.usd   mass and joints                    md5 1750e987efbedc8cae66c66983b13fde
configuration/a2_robot.usd                                        md5 135527794cfad60506c9bbc3e640f98a
configuration/a2_sensor.usd                                       md5 113fd5fe59144c289ed53df7a1b76f67
```

## How it was made

1. Source URDF: unitree_ros `robots/a2_description/urdf/a2.urdf` (md5 730b8f4c9a835221ea4e842b1e0ebb93).
2. Removed **the camera and LiDAR links (camera_link, front_lidar_link, rear_lidar_link), the joints attaching them,
   and the world link** to make `a2_nosensors.urdf` (md5 4d57fb7a494304ceb48d197bd8c121e5).
   Reason: these links have no inertia, so PhysX assigns each one 1 kg and the robot weighs 43 kg
   (40.071 kg without them; 17 links, 12 joints).
3. Converted `a2_nosensors.urdf` to USD with Isaac Lab's URDF converter (UrdfConverter, 2026-09-16). Main settings:
   fix_base false / make_instanceable true / merge_fixed_joints false / self_collision false / collider_type convex_hull /
   joint_drive force, position (stiffness 40, damping 2; **overridden by the user**: both training and deployment
   apply torques directly instead of using the joint drives).
4. Collisions are the URDF's primitive shapes (29 boxes, 4 cylinders, 4 spheres).

LiDAR mounts (values from the URDF before the links were removed; the user attaches the rays):
```
front_lidar_joint  from base_link         xyz (0.33767, 0, 0.08134)    rpy (1.5708, 0, 1.5708)
rear_lidar_joint   from front_lidar_link  xyz (0, 0.005997, -0.617644)  rpy (0, 3.14159, 0)
```

## Usage

Referenced from `isaac.usd` in `configs/robots/a2.yaml`. Joint PD gains, armature, and effort limits come from the policy config.
