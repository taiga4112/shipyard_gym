# Go2 robot model

- `go2_description.urdf`: Unitree Robotics' [Go2 URDF in unitree_ros](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf),
  with `package://go2_description/dae/*.dae` mesh paths rewritten to `meshes/*.obj`.
  The file matches that upstream URDF after this path conversion (verified on 2026-10-27;
  the original source revision is not recorded).
  Masses match the USD the policies were trained with (`usd/go2.usd`, 16.087 kg).
- `meshes/*.obj`: the visual `.dae` meshes converted to OBJ (MuJoCo cannot read COLLADA) and
  decimated to <= 6000 faces each. Visual only: collisions in the URDF are boxes, cylinders
  and spheres.
- `usd/`: the training USD and its `configuration/` layers. Isaac Sim loads the local entry file
  `usd/go2.usd` (see `configs/robots/go2.yaml`).

Copyright Unitree Robotics, BSD 3-Clause License (see `LICENSE`).
