# G1 (29 DoF) robot model

- `g1_29dof_rev_1_0.urdf`: Unitree Robotics' [G1 URDF in unitree_ros](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_rev_1_0.urdf),
  unchanged. The file is byte-identical to that upstream URDF (verified on 2026-10-27;
  the original source revision is not recorded).
  33.341 kg, the same as the USD unitree_rl_lab trains with.
- `meshes/*.STL`: its meshes decimated to <= 6000 faces each (same names). The feet collide
  with spheres; the other links' collision meshes are used as convex hulls by MuJoCo.
- `usd/`: the training USD and its `configuration/` layers. Isaac Sim loads the local entry file
  `usd/g1_29dof_rev_1_0.usd` (see `configs/robots/g1_29dof.yaml`).

Copyright Unitree Robotics, BSD 3-Clause License (see `LICENSE`).
