"""ortsim: run exported locomotion policies (ONNX + deploy.yaml) in Isaac Sim or MuJoCo."""

from pathlib import Path

__version__ = "0.3.0"  # keep in sync with pyproject.toml and the "Release > Main as `x.x.x`" PR title

REPO_ROOT = Path(__file__).resolve().parents[1]
