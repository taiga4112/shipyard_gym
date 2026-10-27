# A2 robot model

- `a2.xml`: unitree_mujoco `unitree_robots/a2/a2.xml`, unchanged (md5 26d8f9e1ab425d8cbb8e5274a3493c3a,
  the file the A2-Terrain-SGS-v1 policy's spec.json names as its MuJoCo body). 40.071 kg, the same
  as the training USD.
- `meshes/*.STL`: that model's visual meshes decimated to <= 6000 faces each (same names, so
  `a2.xml` is byte-identical). Collisions in `a2.xml` are primitives and are not affected.
- `usd/`: the training USD and its `configuration/` layers. Isaac Sim loads the local entry file
  `usd/a2.usd` (see `configs/robots/a2.yaml`).

Copyright Unitree Robotics, BSD 3-Clause License (see `LICENSE`).
