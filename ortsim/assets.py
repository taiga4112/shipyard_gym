"""Resolve repository-relative asset references written in the configs.

A reference uses a ``repo`` key and may include a ``sha256`` checksum:

    {repo: assets/robots/go2/go2_description.urdf, sha256: ...}

Paths are resolved from the repository root. Checksums detect asset changes
since the configuration was written.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from . import REPO_ROOT

_hash_cache: dict[tuple[str, int, int], str] = {}


def sha256_of(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _hash_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _hash_cache[key] = h.hexdigest()
    return _hash_cache[key]


def resolve(ref: dict, verify: bool = True) -> Path:
    if "repo" not in ref:
        raise ValueError(f"File reference needs a 'repo' key: {ref}")

    path = REPO_ROOT / ref["repo"]
    if not path.is_file():
        raise FileNotFoundError(f"Asset not found: {path}")

    expected = ref.get("sha256")
    if verify and expected and os.environ.get("ORTSIM_SKIP_HASH") != "1":
        actual = sha256_of(path)
        if actual != expected:
            raise RuntimeError(
                f"sha256 mismatch for {path}\n  expected {expected}\n  actual   {actual}\n"
                "The file changed since the config was written. Update the config's sha256 "
                "if the change is intended (or set ORTSIM_SKIP_HASH=1 to bypass)."
            )
    return path
