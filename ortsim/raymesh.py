"""Batched raycasts against a static triangle mesh with warp (CPU or CUDA).

The MuJoCo backend uses this for rays that do not share an origin (top-down height
scans: thousands of vertical rays per step), which ``mj_multiRay`` cannot batch.
"""

from __future__ import annotations

import numpy as np
import warp as wp

# No init banner / per-module load lines (warp >= 1.17 uses logging-style levels).
if hasattr(wp.config, "log_level"):
    wp.config.log_level = 30
else:
    wp.config.quiet = True


@wp.kernel
def _cast(mesh: wp.uint64, starts: wp.array(dtype=wp.vec3), dirs: wp.array(dtype=wp.vec3),
          max_dist: float, hits: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh, starts[i], dirs[i], max_dist)
    if q.result:
        hits[i] = starts[i] + q.t * dirs[i]
    else:
        inf = float(1.0e30) * float(1.0e30)
        hits[i] = wp.vec3(inf, inf, inf)


class RayMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, device: str | None = None):
        wp.init()
        self.device = device or ("cuda:0" if wp.is_cuda_available() else "cpu")
        self._points = wp.array(np.asarray(vertices, dtype=np.float32), dtype=wp.vec3, device=self.device)
        self._indices = wp.array(np.asarray(faces, dtype=np.int32).ravel(), dtype=wp.int32, device=self.device)
        self.mesh = wp.Mesh(points=self._points, indices=self._indices)

    def cast(self, starts: np.ndarray, directions: np.ndarray, max_distance: float) -> np.ndarray:
        """(N, 3) starts and directions -> (N, 3) hit points, inf where nothing is hit."""
        n = len(starts)
        s = wp.array(np.ascontiguousarray(starts, dtype=np.float32), dtype=wp.vec3, device=self.device)
        d = wp.array(np.ascontiguousarray(directions, dtype=np.float32), dtype=wp.vec3, device=self.device)
        out = wp.empty(n, dtype=wp.vec3, device=self.device)
        wp.launch(_cast, dim=n, inputs=[self.mesh.id, s, d, float(max_distance), out], device=self.device)
        return out.numpy().astype(np.float64)
