"""Convert a scene USD into collision/visual meshes MuJoCo can use, cached on disk.

    python -m ortsim.tools.build_mujoco_scene basic/5_3 [--force]

MuJoCo collides meshes as their convex hull, so every non-convex mesh is split with
CoACD (``pip install coacd``). Results go to ``.cache/mujoco_scenes/<key>/scene.npz``;
the MuJoCo backend runs this in a subprocess the first time a scene is used (CoACD
and onnxruntime crash when loaded into the same process in some orders).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from .. import REPO_ROOT
from ..assets import sha256_of
from ..config import SceneSpec

CACHE_DIR = REPO_ROOT / ".cache" / "mujoco_scenes"
FORMAT_VERSION = 1


def cache_path(scene: SceneSpec) -> Path:
    key = [FORMAT_VERSION, sha256_of(scene.usd), scene.collision_exclude]
    # Settings added later only enter the key when used, so existing caches stay valid.
    extra = {k: v for k, v in (("up_axis", scene.up_axis), ("collision", scene.collision),
                                ("mujoco", scene.mujoco or {})) if v not in (None, "auto", {})}
    if extra:
        key.append(extra)
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return CACHE_DIR / f"{scene.id.replace('/', '__')}-{digest}" / "scene.npz"


def build(scene: SceneSpec, out: Path, threshold: float) -> None:
    from pxr import Usd

    from ..scene_mesh import extract_meshes, has_authored_collision, stage_correction

    stage = Usd.Stage.Open(str(scene.usd))
    correction = stage_correction(stage, scene.up_axis)
    authored = scene.collision == "authored" or (scene.collision == "auto" and has_authored_collision(stage))
    visual = extract_meshes(stage, correction=correction)
    if authored:
        source = extract_meshes(stage, exclude=scene.is_excluded, collision=True, correction=correction)
    else:
        source = [m for m in visual if not scene.is_excluded(m.path)]
    mj = scene.mujoco or {}
    arrays: dict[str, np.ndarray] = {}
    info = {"visual": [], "collision": [], "ray": [], "authored": authored}
    excluded_paths = {m.path for m in visual if scene.is_excluded(m.path)}
    for i, m in enumerate(visual):
        arrays[f"v{i}_verts"], arrays[f"v{i}_faces"] = m.vertices, m.faces
        info["visual"].append({"path": m.path, "color": m.color, "excluded": m.path in excluded_paths})
    # What the LiDAR / height-scan rays hit: the collision geometry, as in Isaac.
    for i, m in enumerate(source):
        arrays[f"r{i}_verts"], arrays[f"r{i}_faces"] = m.vertices, m.faces
        info["ray"].append({"path": m.path})
    print(f"[mujoco-scene] {len(visual)} visual meshes, {len(source)} "
          f"{'authored' if authored else 'generated'} collision meshes "
          f"({sum(len(m.faces) for m in source)} triangles)", flush=True)

    if mj.get("collision", "convex") == "hfield":
        info["hfield"] = _hfield(source, mj, arrays)
    else:
        for i, m in enumerate(source):
            if m.is_flat:
                continue  # planes have no volume to collide with; they stay raycastable
            parts = [(m.vertices, m.faces)] if m.is_convex() else _decompose(m, threshold)
            for j, (v, f) in enumerate(parts):
                k = f"c{i}_{j}"
                arrays[f"{k}_verts"], arrays[f"{k}_faces"] = v, f
                info["collision"].append({"key": k, "path": m.path})
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, info=np.array(json.dumps(info)), **arrays)


def _hfield(source, mj: dict, arrays: dict) -> dict:
    """Top-down raster of the collision geometry as a MuJoCo height field.

    One vertical ray per cell from ``hfield_cap`` (default: above everything) down; cells
    with no hit become a pit 1 m below the lowest surface. ``crop`` = [xmin, xmax, ymin,
    ymax] limits the area (large scans).
    """
    from ..raymesh import RayMesh
    from ..scene_mesh import merge

    verts, faces = merge(source)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    x0, x1, y0, y1 = mj.get("crop") or (lo[0], hi[0], lo[1], hi[1])
    res = float(mj.get("hfield_resolution", 0.02))
    nx, ny = int(np.ceil((x1 - x0) / res)) + 1, int(np.ceil((y1 - y0) / res)) + 1
    xs, ys = x0 + np.arange(nx) * res, y0 + np.arange(ny) * res
    cap = float(mj["hfield_cap"]) if mj.get("hfield_cap") is not None else float(hi[2]) + 1.0
    gx, gy = np.meshgrid(xs, ys, indexing="xy")  # rows = y, cols = x (MuJoCo's layout)
    starts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, cap)], axis=1)
    dirs = np.broadcast_to(np.array([0.0, 0.0, -1.0]), starts.shape)
    t0 = time.time()
    hits = RayMesh(verts, faces).cast(starts, dirs, cap - lo[2] + 1.0)
    z = hits[:, 2].reshape(ny, nx)
    hit = np.isfinite(z)
    floor = float(z[hit].min()) if hit.any() else 0.0
    z = np.where(hit, z, floor - 1.0)
    # MuJoCo renormalises hfield data to [0, 1] on load, so map the actual min / max there.
    bottom, top = float(z.min()), float(z.max())
    arrays["hfield"] = ((z - bottom) / max(top - bottom, 1e-6)).astype(np.float32)
    print(f"[mujoco-scene] hfield {nx} x {ny} at {res * 100:.0f} cm, z {bottom:.2f}..{top:.2f} m, "
          f"{100 * (1 - hit.mean()):.1f}% empty, cap {cap:.2f} m ({time.time() - t0:.1f}s)", flush=True)
    return {"x0": float(x0), "y0": float(y0), "nx": nx, "ny": ny, "res": res, "z0": bottom, "z1": top}


def _decompose(m, threshold: float):
    try:
        import coacd
    except ImportError:
        print(f"[mujoco-scene] coacd not installed: {m.path} collides as its convex hull", file=sys.stderr)
        return [(m.vertices, m.faces)]
    coacd.set_log_level("error")
    t0 = time.time()
    parts = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), threshold=threshold)
    print(f"[mujoco-scene] {m.path}: {len(parts)} convex parts ({time.time() - t0:.1f}s)", flush=True)
    return [(np.asarray(v), np.asarray(f)) for v, f in parts]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="+", help="scene ids, e.g. basic/5_3")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.03, help="CoACD concavity threshold")
    args = ap.parse_args(argv)
    for scene_id in args.scene:
        scene = SceneSpec.load(scene_id)
        out = cache_path(scene)
        if out.exists() and not args.force:
            print(f"[mujoco-scene] {scene_id}: cached ({out})")
            continue
        print(f"[mujoco-scene] {scene_id}: converting {scene.usd}", flush=True)
        build(scene, out, args.threshold)
        print(f"[mujoco-scene] {scene_id}: wrote {out}")


if __name__ == "__main__":
    main()
