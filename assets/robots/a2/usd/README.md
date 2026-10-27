# A2 の機体の USD（A2-Terrain-SGS-v1 の学習と同じ物）

Isaac Sim / Isaac Lab で A2 を建てるための機体の USD です。方策 A2-Terrain-SGS-v1 の学習・評価は、この USD で行いました。
このパッケージでは `assets/robots/a2/usd/` に配置しています。

## 入っている物

```
a2.usd                         入口（中身は configuration/ の 4 つを読む。4 つと同じ所に置いたまま使う）
  md5 b4d456bc1dc121e37646283285ed64b9
configuration/a2_base.usd      形（見た目と当たり判定）  md5 c0fc759a787f79f0391cb21c0281ad36
configuration/a2_physics.usd   質量・関節                md5 1750e987efbedc8cae66c66983b13fde
configuration/a2_robot.usd                               md5 135527794cfad60506c9bbc3e640f98a
configuration/a2_sensor.usd                              md5 113fd5fe59144c289ed53df7a1b76f67
```

## 作り方

1. 元の URDF: unitree_ros の `robots/a2_description/urdf/a2.urdf`（md5 730b8f4c9a835221ea4e842b1e0ebb93）
2. そこから **カメラと LiDAR のリンク（camera_link・front_lidar_link・rear_lidar_link）と、それをつなぐ関節、world のリンク** を外して
   `a2_nosensors.urdf`（md5 4d57fb7a494304ceb48d197bd8c121e5）を作った。
   理由: これらのリンクには慣性が書かれておらず、PhysX が 1 つにつき 1 kg を割り当てて機体が 43 kg になるため（外すと 40.071 kg。リンク 17・関節 12）
3. `a2_nosensors.urdf` を Isaac Lab の URDF 変換（UrdfConverter。2026-09-16）で USD にした。主な設定:
   fix_base false / make_instanceable true / merge_fixed_joints false / self_collision false / collider_type convex_hull /
   joint_drive force・position（stiffness 40・damping 2。**使う側で上書きする**。学習も deploy の台本も、関節の駆動は使わずトルクを入れる）
4. 当たり判定は URDF 由来の基本形状（箱 29・円柱 4・球 4）

LiDAR の取り付け（外す前の URDF の値。使う側で光線を付ける）:
```
front_lidar_joint  base_link から xyz (0.33767, 0, 0.08134) rpy (1.5708, 0, 1.5708)
rear_lidar_joint   front_lidar_link から xyz (0, 0.005997, -0.617644) rpy (0, 3.14159, 0)
```

## 使い方

`configs/robots/a2.yaml` の `isaac.usd` から参照します。関節の PD・armature・力の上限は、policy 設定の値を使います。
