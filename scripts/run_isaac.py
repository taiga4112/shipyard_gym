"""Run a policy in Isaac Sim (Isaac Lab python env).

    python scripts/run_isaac.py --policy go2_perceptive_mid360_phase5 --scene basic/5_3
    python scripts/run_isaac.py ... --headless --record --duration 20 --command 0.6,0,0

GUI keys (hold to move, release to slow down): W/S (Up/Down) forward / brake, A/D
sideways, Q/E (Left/Right) turn, X stop (clears --command cruise), R reset, C camera view, V start/stop recording, M toggle height-map markers,
H help.
A fall respawns the robot automatically (--reborn start|last|off).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--policy", required=True)
ap.add_argument("--scene", default="builtin/flat")
ap.add_argument("--spawn", type=str, default=None, metavar="X,Y[,YAW_DEG[,GROUND_Z]]",
                help="start here instead of the scene's spawn (use --spawn=-1,... for negatives)")
ap.add_argument("--record", nargs="?", const="", default=None, help="record mp4 from start (optional path)")
ap.add_argument("--duration", type=float, default=None, help="stop after this many seconds of sim time")
ap.add_argument("--command", type=str, default=None,
                help="cruise command vx,vy,yaw: held when no key is pressed (default 0 = stand)")
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--show-heightmap", action="store_true", help="draw the policy's height map")
ap.add_argument("--no-overlay", action="store_true", help="no text overlay in the video")
ap.add_argument("--no-realtime", action="store_true", help="GUI: run as fast as possible")
ap.add_argument("--camera", default=None,
                help="initial follow-camera view: rear_right, rear_left, behind, side_right, top (C cycles)")
ap.add_argument("--reborn", choices=("start", "last", "off"), default="start",
                help="respawn after a fall: at the scene start, where it was ~2 s before falling, or off")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
# The recording camera is an offscreen render product; GUI sessions get one too so
# recording can be toggled with V.
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import carb  # noqa: E402
import omni.appwindow  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

from ortsim.backends.isaac_backend import IsaacBackend  # noqa: E402
from ortsim.camera import FollowCamera  # noqa: E402
from ortsim.command import HELP, KEYMAP  # noqa: E402
from ortsim.config import PolicySpec, SceneSpec  # noqa: E402
from ortsim.reborn import Reborn  # noqa: E402
from ortsim.runner import PolicyRunner  # noqa: E402
from ortsim.video import VideoWriter, default_video_path  # noqa: E402


def main() -> None:
    policy = PolicySpec.load(args.policy)
    scene = SceneSpec.load(args.scene)
    if args.spawn:
        scene = scene.with_spawn(*(float(v) for v in args.spawn.split(",")))
    device = args.device or "cuda:0"
    backend = IsaacBackend(policy, scene, device=device)

    camera = Camera(
        CameraCfg(
            prim_path="/World/RecordCamera",
            update_period=0.0,
            height=args.height,
            width=args.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 200.0)),
        )
    )
    markers = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/HeightMap",
            markers={
                "measured": sim_utils.SphereCfg(radius=0.012, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1))),
                "held": sim_utils.SphereCfg(radius=0.012, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1))),
            },
        )
    )
    backend.start()

    runner = PolicyRunner(policy, backend)
    if args.command:
        runner.command.set([float(x) for x in args.command.split(",")])
    runner.reset()
    runner.command.report()
    reborn = Reborn(args.reborn, policy.step_dt, tilt_deg=float(np.degrees(policy.fall_tilt)))

    cam_cfg = policy.robot.camera
    follow = FollowCamera.from_config(cam_cfg, args.camera or scene.camera)
    state = {"show_map": args.show_heightmap, "writer": None, "reset": False}
    markers.set_visibility(state["show_map"])

    def start_recording(path: str | None = None) -> None:
        out = Path(path) if path else default_video_path(policy.name, scene.id, "isaac")
        state["writer"] = VideoWriter(out, fps=1.0 / policy.step_dt, overlay=not args.no_overlay)

    def stop_recording() -> None:
        if state["writer"] is not None:
            state["writer"].close()
            state["writer"] = None

    if args.record is not None:
        start_recording(args.record or None)

    # --- keyboard ---------------------------------------------------------------------
    held: set[str] = set()

    def on_key(event, *_):
        kind = event.type
        # CHAR events carry the typed character (a str) in event.input, not a key.
        if kind not in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_RELEASE):
            return True
        k = event.input.name
        if kind == carb.input.KeyboardEventType.KEY_RELEASE:
            held.discard(k)
            return True
        c = runner.command
        if k in KEYMAP:
            held.add(k)
        elif k == "X":
            c.stop()
        elif k == "R":
            state["reset"] = True
        elif k == "V":
            stop_recording() if state["writer"] is not None else start_recording()
        elif k == "C":
            follow.cycle()
        elif k == "M":
            state["show_map"] = not state["show_map"]
            markers.set_visibility(state["show_map"])
        elif k == "H":
            print(HELP, flush=True)
        return True

    key_sub = None
    if not args.headless:
        appwindow = omni.appwindow.get_default_app_window()
        key_sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(appwindow.get_keyboard(), on_key)
        print(HELP, flush=True)

    # --- loop -------------------------------------------------------------------------
    step_dt = policy.step_dt
    max_steps = int(args.duration / step_dt) if args.duration else None
    next_tick = time.perf_counter()
    try:
        while app.is_running():
            if max_steps is not None and runner.step_count >= max_steps:
                break
            if state["reset"]:
                state["reset"] = False
                runner.reset()
                reborn.respawned(manual=True)
                follow.reset()
                print("[reset]", flush=True)

            runner.command.update(held, step_dt)
            if not args.headless:
                runner.command.maybe_report()
            s = runner.step()
            due = reborn.update(s, runner.ground_below(s))
            if due is not None:
                reason, pose = due
                where = "start" if pose is None else f"({pose.x:+.2f}, {pose.y:+.2f})"
                print(f"[reborn #{reborn.count}] fell at ({s.base_pos[0]:+.2f}, {s.base_pos[1]:+.2f}): {reason}"
                      f" -> respawn at {where}", flush=True)
                runner.reset(pose)
                reborn.respawned()
                follow.reset()
            eye, target = follow.update(s.base_pos, s.base_quat, backend.raycast)
            if not args.headless:
                backend.sim.set_camera_view(tuple(eye), tuple(target))
            drawn = runner.map_points(s) if state["show_map"] else None
            if drawn is not None:
                pts, flagged = drawn
                markers.visualize(translations=torch.as_tensor(pts, dtype=torch.float32),
                                  marker_indices=torch.as_tensor(flagged.astype(np.int64)))

            writer = state["writer"]
            if writer is not None:
                camera.set_world_poses_from_view(
                    torch.tensor([eye], dtype=torch.float32, device=device),
                    torch.tensor([target], dtype=torch.float32, device=device),
                )
            backend.render()
            if writer is not None:
                camera.update(step_dt)
                rgb = camera.data.output["rgb"][0].cpu().numpy()
                lines = [policy.name, f"scene: {scene.id}   t={runner.step_count * step_dt:5.1f}s", runner.command.describe()]
                lines += [f"reborn x{reborn.count}"] if reborn.count else []
                writer.write(rgb, lines)

            if args.headless:
                if runner.step_count % 50 == 0:
                    p = s.base_pos
                    print(f"[t={runner.step_count * step_dt:5.1f}s] base=({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:.2f})", flush=True)
            elif not args.no_realtime:
                next_tick += step_dt
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        stop_recording()
        key_sub = None


if __name__ == "__main__":
    import faulthandler

    main()
    # Replicator stops the timeline on close, and Isaac Lab's stop handler would then
    # render forever waiting for play; it is only meant for interactive stops.
    sim = sim_utils.SimulationContext.instance()
    if sim is not None:
        sim._disable_app_control_on_stop_handle = True
    faulthandler.dump_traceback_later(120, exit=True)
    app.close()
