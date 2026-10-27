"""Minimal GLFW viewer for the MuJoCo runner.

``mujoco.viewer`` only reports key presses (no releases) and binds most letters to its
own toggles, so hold-to-move control is impossible there. This window reports held
keys by the same names Isaac's carb input uses, so both runners share one key layout.
The camera is driven by the caller (follow camera); ``zoom`` (mouse wheel) is read by
the caller and applied before the follow camera's wall check.
"""

from __future__ import annotations

import glfw
import mujoco
import numpy as np

_GLFW_NAMES = {
    glfw.KEY_UP: "UP", glfw.KEY_DOWN: "DOWN", glfw.KEY_LEFT: "LEFT", glfw.KEY_RIGHT: "RIGHT",
    glfw.KEY_SPACE: "SPACE", glfw.KEY_ESCAPE: "ESCAPE",
}


def _key_name(key: int) -> str | None:
    if glfw.KEY_A <= key <= glfw.KEY_Z:
        return chr(key)
    return _GLFW_NAMES.get(key)


class GlfwViewer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, title: str, width: int = 1280, height: int = 720):
        if not glfw.init():
            raise RuntimeError("glfw.init() failed (no display? use --headless)")
        glfw.window_hint(glfw.SAMPLES, 4)
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("could not open a window (no display? use --headless)")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.model, self.data = model, data
        self.scene = mujoco.MjvScene(model, maxgeom=20000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.option = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.zoom = 1.0
        self.held: set[str] = set()
        self.pressed: list[str] = []
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_window_focus_callback(self.window, self._on_focus)

    # --- input ------------------------------------------------------------------------

    def _on_key(self, window, key, scancode, action, mods) -> None:
        name = _key_name(key)
        if name is None:
            return
        if action == glfw.PRESS:
            self.held.add(name)
            self.pressed.append(name)
        elif action == glfw.RELEASE:
            self.held.discard(name)

    def _on_scroll(self, window, dx, dy) -> None:
        self.zoom = float(np.clip(self.zoom * (0.9 ** dy), 0.3, 4.0))

    def _on_focus(self, window, focused) -> None:
        if not focused:
            self.held.clear()  # releases are lost while unfocused; never leave a key stuck

    def poll(self) -> list[str]:
        """Process window events; returns keys pressed since the last call."""
        glfw.poll_events()
        pressed, self.pressed = self.pressed, []
        return pressed

    def is_running(self) -> bool:
        return not glfw.window_should_close(self.window)

    # --- drawing ----------------------------------------------------------------------

    def set_camera(self, eye: np.ndarray, target: np.ndarray) -> None:
        set_camera(self.cam, eye, target)

    def _draw(self, viewport, extra_geoms, lines, cam=None) -> None:
        cam = self.cam if cam is None else cam
        mujoco.mjv_updateScene(self.model, self.data, self.option, None, cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        if extra_geoms is not None:
            extra_geoms(self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        if lines:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport,
                               "\n".join(lines), "", self.context)

    def render(self, extra_geoms=None, lines: list[str] | None = None) -> None:
        w, h = glfw.get_framebuffer_size(self.window)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        self._draw(mujoco.MjrRect(0, 0, w, h), extra_geoms, lines)
        glfw.swap_buffers(self.window)

    def capture(self, width: int, height: int, extra_geoms=None, cam=None) -> np.ndarray:
        """Render the current view (or ``cam``) offscreen at (width, height); returns RGB uint8."""
        width = min(width, self.model.vis.global_.offwidth)
        height = min(height, self.model.vis.global_.offheight)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        rect = mujoco.MjrRect(0, 0, width, height)
        self._draw(rect, extra_geoms, None, cam)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, rect, self.context)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        return np.flipud(rgb)

    def close(self) -> None:
        self.context.free()
        glfw.destroy_window(self.window)
        glfw.terminate()


def set_camera(cam: mujoco.MjvCamera, eye: np.ndarray, target: np.ndarray, zoom: float = 1.0) -> None:
    """Point a free camera from ``eye`` at ``target`` (``zoom`` scales the distance)."""
    d = eye - target
    dist = float(np.linalg.norm(d))
    cam.lookat[:] = target
    cam.distance = dist * zoom
    cam.azimuth = float(np.degrees(np.arctan2(-d[1], -d[0])))
    cam.elevation = float(np.degrees(np.arcsin(-d[2] / dist)))
