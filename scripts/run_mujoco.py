"""Run a policy in MuJoCo.

    uv run scripts/run_mujoco.py --policy go2_perceptive_mid360_phase5 --scene basic/5_3
    uv run scripts/run_mujoco.py ... --headless --record --duration 20 --command 0.6,0,0

Keys (same layout as the Isaac runner; hold to move, release to slow down and stop):
W/S (Up/Down) forward / brake, A/D sideways, Q/E (Left/Right) turn, X stop (clears the
--command cruise speed), R reset, C camera view, V start/stop recording, M height-map markers, H help.
Mouse wheel zooms the follow camera. A fall respawns the robot (--reborn start|last|off).

--remote PORT lets robot_controller's bridge drive the robot over localhost UDP and
streams the robot's head camera back (see ortsim/remote.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

# onnxruntime must be imported before anything that may pull in coacd (see build_mujoco_scene).
import onnxruntime  # noqa: F401,E402

from ortsim.camera import FollowCamera  # noqa: E402
from ortsim.command import HELP, KEYMAP  # noqa: E402
from ortsim.config import PolicySpec, SceneSpec  # noqa: E402
from ortsim.reborn import MODES, Reborn  # noqa: E402
from ortsim.runner import PolicyRunner  # noqa: E402
from ortsim.video import VideoWriter, default_video_path  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--scene", default="builtin/flat")
    ap.add_argument("--spawn", type=str, default=None, metavar="X,Y[,YAW_DEG[,GROUND_Z]]",
                    help="start here instead of the scene's spawn (use --spawn=-1,... for negatives)")
    ap.add_argument("--headless", action="store_true", help="no viewer window (use with --record)")
    ap.add_argument("--record", nargs="?", const="", default=None, help="record mp4 (optional path)")
    ap.add_argument("--duration", type=float, default=None, help="stop after this many seconds of sim time")
    ap.add_argument("--command", type=str, default=None,
                    help="cruise command vx,vy,yaw: held when no key is pressed (default 0 = stand)")
    ap.add_argument("--width", type=int, default=1280, help="window / video width")
    ap.add_argument("--height", type=int, default=720, help="window / video height")
    ap.add_argument("--show-heightmap", action="store_true", help="draw the policy's height map")
    ap.add_argument("--no-overlay", action="store_true", help="no text overlay in the video")
    ap.add_argument("--camera", default=None,
                    help="initial follow-camera view: rear_right, rear_left, behind, side_right, top (C cycles)")
    ap.add_argument("--reborn", choices=MODES, default="start",
                    help="respawn after a fall: at the scene start, where it was ~2 s before falling, or off")
    ap.add_argument("--remote", type=int, default=None, metavar="PORT",
                    help="accept velocity commands on udp://127.0.0.1:PORT and stream the head camera")
    ap.add_argument("--remote-size", default="640x360", help="head-camera stream size WxH")
    ap.add_argument("--remote-fps", type=float, default=25.0, help="head-camera stream frame rate")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # EGL is a Linux backend; macOS and Windows use MuJoCo's platform defaults.
    if args.headless and sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from ortsim.backends.mujoco_backend import MujocoBackend
    from ortsim.mujoco_viewer import set_camera

    policy = PolicySpec.load(args.policy)
    scene = SceneSpec.load(args.scene)
    if args.spawn:
        scene = scene.with_spawn(*(float(v) for v in args.spawn.split(",")))
    backend = MujocoBackend(policy, scene)
    runner = PolicyRunner(policy, backend)
    if args.command:
        runner.command.set([float(x) for x in args.command.split(",")])
    runner.reset()
    runner.command.report()
    reborn = Reborn(args.reborn, policy.step_dt, tilt_deg=float(np.degrees(policy.fall_tilt)))
    step_dt = policy.step_dt

    cam_cfg = policy.robot.camera
    follow = FollowCamera.from_config(cam_cfg, args.camera or scene.camera)
    model, data = backend.model, backend.data
    show_map = args.show_heightmap

    viewer = None
    if not args.headless:
        from ortsim.mujoco_viewer import GlfwViewer

        viewer = GlfwViewer(model, data, f"ortsim - {policy.name} - {scene.id}", args.width, args.height)
        print(HELP, flush=True)

    renderer = None  # headless recording only; the viewer captures from its own context
    headless_cam = mujoco.MjvCamera()
    headless_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    writer: VideoWriter | None = None

    def start_recording(path: str | None = None) -> None:
        nonlocal renderer, writer
        if viewer is None and renderer is None:
            renderer = mujoco.Renderer(model, args.height, args.width)
        out = Path(path) if path else default_video_path(policy.name, scene.id, "mujoco")
        writer = VideoWriter(out, fps=1.0 / step_dt, overlay=not args.no_overlay)

    def stop_recording() -> None:
        nonlocal writer
        if writer is not None:
            writer.close()
            writer = None

    def draw_heightmap(scn: mujoco.MjvScene) -> None:
        drawn = runner.map_points(last_state) if show_map else None
        if drawn is None:
            return
        for p, held_cell in zip(*drawn):
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.012, 0, 0]), p, np.eye(3).ravel(),
                np.array([1, 0.2, 0.2, 1] if held_cell else [0.2, 1, 0.2, 1], dtype=np.float32),
            )
            scn.ngeom += 1

    def overlay_lines() -> list[str]:
        lines = [policy.name, f"scene: {scene.id}   t={runner.step_count * step_dt:5.1f}s", runner.command.describe()]
        return lines + ([f"reborn x{reborn.count}"] if reborn.count else [])

    def respawn(pose=None, manual: bool = False) -> None:
        runner.reset(pose)
        reborn.respawned(manual=manual)
        follow.reset()

    if args.record is not None:
        start_recording(args.record or None)

    remote = None
    if args.remote is not None:
        import cv2

        from ortsim.backends.mujoco_backend import HEAD_CAMERA
        from ortsim.remote import RemotePort

        remote = RemotePort(args.remote)
        head_w, head_h = (int(v) for v in args.remote_size.lower().split("x"))
        head_cam = mujoco.MjvCamera()
        head_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        head_cam.fixedcamid = model.camera(HEAD_CAMERA).id
        head_renderer = None if viewer is not None else mujoco.Renderer(model, head_h, head_w)
        frame_every = max(1, int(round(1.0 / (args.remote_fps * step_dt))))

    def send_head_frame() -> None:
        if viewer is not None:
            rgb = viewer.capture(head_w, head_h, cam=head_cam)
        else:
            head_renderer.update_scene(data, camera=head_cam)
            rgb = head_renderer.render()
        ok, jpg = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            remote.send_frame(jpg.tobytes())

    max_steps = int(args.duration / step_dt) if args.duration else None
    next_tick = time.perf_counter()
    last_state = backend.get_state()
    try:
        while viewer is None or viewer.is_running():
            if max_steps is not None and runner.step_count >= max_steps:
                break

            held: set[str] = set()
            if viewer is not None:
                for k in viewer.poll():
                    if k == "X":
                        runner.command.stop()
                    elif k == "R":
                        respawn(manual=True)
                        print("[reset]", flush=True)
                    elif k == "V":
                        stop_recording() if writer is not None else start_recording()
                    elif k == "C":
                        follow.cycle()
                    elif k == "M":
                        show_map = not show_map
                    elif k == "H":
                        print(HELP, flush=True)
                held = {k for k in viewer.held if k in KEYMAP}
            target = None
            if remote is not None:
                remote.poll()
                if remote.reset_requested:
                    remote.reset_requested = False
                    respawn(manual=True)
                    print("[reset] (remote)", flush=True)
                target = remote.command_target()
            if target is not None:
                runner.command.update_target(target, step_dt)
            else:
                runner.command.update(held, step_dt)
            if viewer is not None:
                runner.command.maybe_report()

            last_state = runner.step()
            due = reborn.update(last_state, runner.ground_below(last_state))
            if due is not None:
                reason, pose = due
                p = last_state.base_pos
                where = "start" if pose is None else f"({pose.x:+.2f}, {pose.y:+.2f})"
                print(f"[reborn #{reborn.count}] fell at ({p[0]:+.2f}, {p[1]:+.2f}): {reason} -> respawn at {where}",
                      flush=True)
                respawn(pose)

            if remote is not None:
                remote.send_state(last_state, runner.command, runner.step_count, step_dt, policy.name, reborn.count)
                if remote.peer is not None and runner.step_count % frame_every == 0:
                    send_head_frame()

            if viewer is not None:
                follow.zoom = viewer.zoom
            eye, target = follow.update(last_state.base_pos, last_state.base_quat, backend.raycast)
            if viewer is not None:
                viewer.set_camera(eye, target)
                if writer is not None:
                    writer.write(viewer.capture(args.width, args.height, draw_heightmap), overlay_lines())
                viewer.render(draw_heightmap, overlay_lines())
                next_tick += step_dt
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
            else:
                if remote is not None:  # a remote driver needs real time, not as fast as possible
                    next_tick += step_dt
                    delay = next_tick - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_tick = time.perf_counter()
                if writer is not None:
                    set_camera(headless_cam, eye, target)
                    renderer.update_scene(data, camera=headless_cam)
                    draw_heightmap(renderer.scene)
                    writer.write(renderer.render(), overlay_lines())
                if runner.step_count % 50 == 0:
                    p = last_state.base_pos
                    print(f"[t={runner.step_count * step_dt:5.1f}s] base=({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:.2f})", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop_recording()
        if remote is not None:
            remote.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
