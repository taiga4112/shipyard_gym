"""World-space triangle meshes from a USD stage (needs ``pxr``).

Used for the LiDAR raycast mesh in Isaac Sim and to build the MuJoCo scene, so both
backends see exactly the same geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeshData:
    path: str
    vertices: np.ndarray  # (V, 3) world frame, metres
    faces: np.ndarray  # (F, 3) int
    color: tuple[float, float, float]

    @property
    def is_flat(self) -> bool:
        """True when the mesh has no volume (a plane), which MuJoCo cannot collide with."""
        centered = self.vertices - self.vertices.mean(axis=0)
        return np.linalg.svd(centered, compute_uv=False)[-1] < 1e-6 * max(1.0, np.abs(centered).max())

    def is_convex(self, tol: float = 1e-6) -> bool:
        v = self.vertices
        for f in self.faces:
            n = np.cross(v[f[1]] - v[f[0]], v[f[2]] - v[f[0]])
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            d = (v - v[f[0]]) @ (n / norm)
            if d.max() > tol and d.min() < -tol:
                return False
        return True


def stage_correction(stage, up_axis: str | None = None) -> np.ndarray:
    """4x4 (row-vector convention, like USD) that brings a stage to Z-up metres.

    ``up_axis`` overrides the stage's own ``upAxis`` metadata for files that declare one
    axis but were authored in the other (an FBX export marked Z-up holding Y-up data).
    """
    from pxr import UsdGeom

    up = (up_axis or str(UsdGeom.GetStageUpAxis(stage))).upper()
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    m = np.eye(4)
    if up == "Y":
        # (x, y, z) -> (x, -z, y): +Y becomes +Z.
        m[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    elif up != "Z":
        raise ValueError(f"unsupported up axis {up!r}")
    m[:3, :3] *= mpu
    return m


def _triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Fan-triangulate polygons given USD faceVertexCounts / faceVertexIndices."""
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    tris = []
    for k in range(1, int(counts.max(initial=3)) - 1):
        sel = counts > k + 1
        s = starts[sel]
        tris.append(np.stack([indices[s], indices[s + k], indices[s + k + 1]], axis=1))
    return np.concatenate(tris) if tris else np.zeros((0, 3), dtype=np.int64)


def extract_meshes(stage, root_path: str = "/", exclude=None, *, collision: bool = False,
                   correction: np.ndarray | None = None) -> list[MeshData]:
    """Collect Meshes under ``root_path`` in Z-up metres.

    ``collision=False``: the visible meshes (rendered geometry). ``collision=True``: the
    meshes carrying UsdPhysics.CollisionAPI, visible or not -- scenes often hide their
    collision proxies. ``exclude(prim_path) -> bool`` skips prims. ``correction`` (see
    ``stage_correction``) is applied after each prim's world transform; by default the
    stage's own metersPerUnit is applied and a non-Z up axis only warned about.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    if correction is None:
        up = UsdGeom.GetStageUpAxis(stage)
        if up != UsdGeom.Tokens.z:
            print(f"[scene] warning: stage up axis is {up}, expected Z; geometry is used as-is")
        correction = np.eye(4)
        correction[:3, :3] *= UsdGeom.GetStageMetersPerUnit(stage) or 1.0

    root = stage.GetPrimAtPath(root_path)
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meshes = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if collision:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
        else:
            img = UsdGeom.Imageable(prim)
            if img.ComputeVisibility() == UsdGeom.Tokens.invisible or img.ComputePurpose() in (
                UsdGeom.Tokens.guide,
                UsdGeom.Tokens.proxy,
            ):
                continue
        path = str(prim.GetPath())
        if exclude is not None and exclude(path):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts:
            continue
        pts = np.asarray(points, dtype=np.float64)
        mat = np.asarray(xf_cache.GetLocalToWorldTransform(prim), dtype=np.float64) @ correction  # row-vector
        world = (np.hstack([pts, np.ones((len(pts), 1))]) @ mat)[:, :3]

        faces = _triangulate(counts, indices)
        # Mirroring transforms flip the winding; keep outward normals for MuJoCo.
        if np.linalg.det(mat[:3, :3]) < 0:
            faces = faces[:, [0, 2, 1]]

        color = (0.6, 0.6, 0.6)
        colors = mesh.GetDisplayColorAttr().Get() if mesh.GetDisplayColorAttr() else None
        if colors:
            color = tuple(float(x) for x in colors[0])
        meshes.append(MeshData(path, world, faces, color))
    return meshes


def has_authored_collision(stage, root_path: str = "/") -> bool:
    from pxr import Usd, UsdPhysics

    return any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)))


def merge(meshes: list[MeshData]) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, offset = [], [], 0
    for m in meshes:
        verts.append(m.vertices)
        faces.append(m.faces + offset)
        offset += len(m.vertices)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(verts), np.concatenate(faces)
