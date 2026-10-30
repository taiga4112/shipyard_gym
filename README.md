# Shipyard Gym

Shipyard Gym is a simulation workspace for exploring general-purpose robot behavior in shipyard environments. It provides a place to try robot capabilities against the terrain and structures they may encounter in shipbuilding, inspect how they behave, and identify what needs to improve.

## Purpose and approach

The goal is to explore how general-purpose robots can operate across the varied conditions of a shipyard. The test scenes include plate steps, inverted-L structures, manhole openings, stairs, slopes, and grating, making it possible to examine specific challenges in isolation and repeat experiments under consistent conditions.

The project takes a practical, iterative approach:

- **Start with shipyard conditions.** Use representative obstacles and terrain to guide experiments in robot behavior.
- **Try different robots and policies.** Keep robot, policy, and scene configurations separate so that capabilities can be explored across platforms.
- **Make behavior observable.** Use interactive control, terrain perception visualization, and video recording to inspect successes and failures.
- **Use simulation to inform further development.** Treat results as evidence for the next experiment and eventual hardware validation.

The current implementation focuses on **locomotion and terrain traversal** using pretrained policies for Unitree Go2, A2, and G1. It runs policy inference in Isaac Sim or MuJoCo, with keyboard control and a follow camera. Broader shipyard tasks are part of the project's direction; the functionality documented here is the current locomotion test environment.

## Included policies

| Policy | Robot | Description |
|---|---|---|
| `go2_perceptive_mid360_phase5` | Unitree Go2 | Rough-terrain locomotion over walls and steps; MID-360 LiDAR height map and a GRU |
| `a2_terrain_sgs_v1` | Unitree A2 | Rough-terrain locomotion over stairs, walls, boxes, and gaps; 735-point top-down height map, gait phase, and an MLP |
| `a2_distill2` | Unitree A2 | Distilled policy combining stair and longitudinal stiffener teachers; 17 × 11 height map accumulated from front and rear LiDAR, gait phase, and an MLP |
| `unitree_g1_29dof_velocity_v0` | Unitree G1 (29 DoF) | Flat-ground locomotion using Unitree's pretrained policy; proprioception only, with a five-step observation history |

## Requirements

| Component | Requirements |
|---|---|
| Isaac Sim | Linux and an NVIDIA GPU. Isaac Sim 5.1 and Isaac Lab 0.54, the previously verified combination. Use the Isaac Lab Python environment. |
| MuJoCo | Linux, Windows, or macOS, and [uv](https://docs.astral.sh/uv/), which manages the Python environment and project dependencies. An NVIDIA GPU is not required. |

## Quick start

Run commands from the repository root.

### Isaac Sim

Activate your Isaac Lab Python environment, then run:

```bash
python scripts/run_isaac.py --policy go2_perceptive_mid360_phase5 --scene basic/0_1
```

Startup takes approximately one minute. Once the window opens, click the viewport to enable keyboard input.

### MuJoCo

```bash
uv sync    # Initial environment setup
uv run scripts/run_mujoco.py --policy go2_perceptive_mid360_phase5 --scene basic/0_1
```

Choose another policy from the table above and a scene from the catalog below to explore different combinations. Policy inclusion does not imply that it can traverse every scene.

## Controls

The following controls are shared by Isaac Sim and MuJoCo. Hold a movement key to move; releasing it slows the robot to a stop by default. If `--command` is set, releasing the keys returns to the configured cruise command.

| Key | Action |
|---|---|
| W / Up arrow | Move forward |
| S / Down arrow | Brake |
| A / D | Move sideways left / right |
| Q / E or Left / Right arrow | Turn left / right |
| X | Stop and clear the cruise command |
| R | Reset to the starting position |
| C | Cycle camera views |
| V | Start / stop recording; videos are saved to `videos/` by default |
| M | Toggle the terrain height map used by the policy |
| H | Show help |
| Mouse wheel | Zoom (MuJoCo only) |

In Isaac Sim, Space pauses the simulation. Use X to stop the robot.

## Run options

| Option | Description |
|---|---|
| `--policy <name>` | Select a pretrained policy (required). |
| `--scene <scene>` | Select a scene from the catalog below. Use `builtin/flat` for flat ground. |
| `--command 0.6,0,0` | Set the cruise command: forward velocity (m/s), lateral velocity (m/s), and yaw rate (rad/s). Releasing movement keys returns to this command. |
| `--spawn=14.05,1.3,270` | Override the scene's starting pose: `X,Y[,YAW_DEG[,GROUND_Z]]`. |
| `--camera rear_left` | Set the initial camera view: `rear_right`, `rear_left`, `behind`, `side_right`, or `top`. |
| `--reborn start\|last\|off` | After a fall, respawn at the start, respawn at the pose from about two seconds before the fall, or disable respawning. Default: `start`. |
| `--show-heightmap` | Display the policy's terrain height map. See the color legend below. |
| `--record [file.mp4]` | Start recording immediately, optionally to a specified path. |
| `--duration 20` | Exit after 20 seconds of simulation time. |
| `--headless` | Run without a viewer window, for example for batch recording. |

Height-map markers are green for points observed in the current step and red for retained observations. For policies using top-down measurements, red indicates a missed ray or a point outside the measurement range. For policies using accumulated LiDAR, red can also indicate a cell that has not yet been observed.

### Batch recording

Record a selection of scenes without opening a viewer. Output is saved to `videos/` by default. Extend the scene list using the catalog below.

The loops below use Bash syntax (on Windows, use Git Bash or adapt them for PowerShell). MuJoCo uses EGL for headless rendering on Linux and the platform's default rendering backend on macOS and Windows. If you have explicitly set `MUJOCO_GL`, ensure it is supported on your OS; `egl` is not supported on macOS.

```bash
# Isaac Sim
for scene in basic/0_1 basic/1_1_1 basic/2_1 basic/5_1; do
  python scripts/run_isaac.py --policy go2_perceptive_mid360_phase5 --scene "$scene" \
    --headless --record --duration 20 --command 0.6,0,0
done

# MuJoCo
for scene in basic/0_1 basic/1_1_1 basic/2_1 basic/5_1; do
  uv run scripts/run_mujoco.py --policy go2_perceptive_mid360_phase5 --scene "$scene" \
    --headless --record --duration 20 --command 0.6,0,0
done
```

## Scene catalog

Basic locomotion and terrain traversal scenes are stored in `assets/scenes/basic/` and referenced using `repo:` paths. Scene definitions are maintained in [`configs/scenes/basic.yaml`](configs/scenes/basic.yaml).
Plate steps are narrow vertical plates arranged in a grid. Inverted-L steps add a horizontal plate at the top. Manholes are openings in a wall: *sill* is the height of the opening's lower edge above the ground, and *depth* is the wall thickness.

<!-- scenes:begin -->
<!-- generated by `uv run python -m ortsim.tools.scene_docs`; edit configs/scenes/*.yaml -->

| Scene | Description |
|---|---|
| `basic/0_1` | Flat ground |
| `basic/1_1_1` | Flat, plate steps (10 cm), 1 m apart |
| `basic/1_1_2` | Flat, plate steps (20 cm), 1 m apart |
| `basic/1_1_3_1` | Flat, plate steps (25 cm), 2 m apart |
| `basic/1_1_3_2` | Flat, plate steps (25 cm), 1 m apart |
| `basic/1_1_3_3` | Flat, plate steps (25 cm), 0.6 m apart |
| `basic/1_1_4` | Flat, plate steps (30 cm), 1 m apart |
| `basic/1_1_5_1` | Flat, plate steps (40 cm), 2 m apart |
| `basic/1_1_5_2` | Flat, plate steps (40 cm), 1 m apart |
| `basic/1_1_5_3` | Flat, plate steps (40 cm), 0.6 m apart |
| `basic/1_1_6` | Flat, plate steps (60 cm), 2 m along the course and 1 m across |
| `basic/1_1_6_1` | Flat, plate steps (60 cm), 2 m apart |
| `basic/1_1_6_2` | Flat, plate steps (60 cm), 1 m apart |
| `basic/1_1_6_3` | Flat, plate steps (60 cm), 0.6 m apart |
| `basic/1_1_7` | Flat, plate steps (35 cm), 1 m apart |
| `basic/1_2_1_1` | Flat, inverted-L steps (20 cm), 2 m apart |
| `basic/1_2_1_2` | Flat, inverted-L steps (20 cm), 1 m apart |
| `basic/1_2_1_3` | Flat, inverted-L steps (20 cm), 0.6 m apart |
| `basic/1_2_2` | Flat, inverted-L steps (40 cm) |
| `basic/1_2_3` | Flat, inverted-L steps (60 cm) |
| `basic/1_2_4` | Flat, inverted-L steps (30 cm) |
| `basic/1_2_5` | Flat, inverted-L steps (35 cm) |
| `basic/1_2_6` | Flat, inverted-L steps (25 cm) |
| `basic/1_4_1` | Flat, tilted steps (small) |
| `basic/1_4_2` | Flat, tilted steps (medium) |
| `basic/1_4_3` | Flat, tilted steps (large) |
| `basic/2_1` | Flat, manhole (sill 10 cm, width 80 cm, depth 1 cm) |
| `basic/2_2` | Flat, manhole (sill 30 cm, width 80 cm, depth 1 cm) |
| `basic/2_3` | Flat, manhole (sill 60 cm, width 80 cm, depth 1 cm) |
| `basic/2_4` | Flat, manhole (sill 10 cm, width 60 cm, depth 1 cm) |
| `basic/2_5` | Flat, manhole (sill 60 cm, width 60 cm, depth 1 cm) |
| `basic/2_6` | Flat, manhole (sill 60 cm, width 60 cm, depth 10 cm) |
| `basic/2_7` | Flat, manhole (sill 60 cm, width 60 cm, depth 20 cm) |
| `basic/2_8` | Flat, manhole (sill 10 cm, width 60 cm, depth 20 cm) |
| `basic/2_9` | Flat, manhole (sill 25 cm, width 60 cm, depth 1 cm) |
| `basic/3_1` | Flat, manhole (sill 60 cm, width 60 cm, depth 10 cm) with an inverted-L step on the floor right below the opening |
| `basic/3_2` | Flat, manhole (sill 60 cm, width 60 cm, depth 10 cm) with an adjacent plate |
| `basic/5_1` | Stairs (20 cm tread, 20 cm rise, with risers) |
| `basic/5_2` | Stairs (20 cm tread, 20 cm rise, without risers) |
| `basic/5_3` | Slope (5 deg) |
| `basic/5_4` | Slope (15 deg) |
| `basic/5_5` | Changing gradient |
| `basic/6_1` | Grating |
| `bridge_scanned/ynu` | Environment created from a scan of the structural engineering laboratory at Yokohama National University (YNU), as part of the BRIDGE project |
| `builtin/flat` | Flat ground |

<!-- scenes:end -->

## Repository layout

```text
configs/       Robot, policy, and scene configurations
assets/        Robot models, ONNX policies, and USD scenes (scenes/basic/)
ortsim/        Shared policy runner, simulator backends, and perception utilities
scripts/       Entry points: run_isaac.py and run_mujoco.py
videos/        Recorded videos, when created
```

### Robot assets

Robot assets are grouped under `assets/robots/<robot>/`. Isaac Sim USD files and associated layers are in `usd/`; MuJoCo URDF or MJCF files are in the robot directory; meshes are in `meshes/`. Configurations in `configs/robots/` reference these assets using `repo:` paths. See each robot's `SOURCE.md` for provenance.

### Policy assets

Pretrained models use the path `assets/policies/<policy>/policy.onnx`. Go2 and G1 policies also have a `deploy.yaml` in the same directory for external observation and action settings. A2 observation and action settings are included in `configs/policies/<policy>.yaml`. Policy configurations reference their models using `repo:` paths.

Each policy directory, `assets/scenes/basic/`, and `assets/sensors/` also has a `SOURCE.md` describing provenance and license.

## Interpreting results

- Commands outside a policy's training range can lead to poor locomotion. The console reports a warning when commands exceed that range.
- Isaac Sim and MuJoCo use different contact models, so behavior can differ even with the same policy and scene.
- Simulation results do not establish real-world robot performance. Hardware validation is needed before drawing conclusions about operation in a shipyard.

## License

The code, configurations, the Open Reach Tech policies (`go2_perceptive_mid360_phase5`, `a2_terrain_sgs_v1`, `a2_distill2`), and the scenes in `assets/scenes/` are released under the [MIT License](LICENSE), Copyright (c) 2026 Taiga Mitsuyuki and Open Reach Tech Inc.

Third-party assets keep their own licenses; see the `LICENSE` and `SOURCE.md` files in each directory:

| Path | Source | License |
|---|---|---|
| `assets/robots/{go2,a2,g1_29dof}/` (excluding Go2/G1 `usd/`) | Unitree Robotics (unitree_ros, unitree_mujoco) | BSD-3-Clause |
| `assets/robots/{go2,g1_29dof}/usd/` | Unitree Robotics (HuggingFace `unitreerobotics/unitree_model`) | Licensing unresolved; upstream has no explicit license |
| `assets/policies/unitree_g1_29dof_velocity_v0/` | Unitree Robotics (unitree_rl_lab) | Apache-2.0 |
| `assets/sensors/mid360.npy` | Livox (livox_laser_simulation) | MIT |
