"""YAML loading that tolerates the ``!!python/...`` tags Isaac Lab writes into deploy/env yaml."""

from __future__ import annotations

from pathlib import Path

import yaml


class _TolerantLoader(yaml.SafeLoader):
    pass


def _construct_python_tag(loader: yaml.SafeLoader, suffix: str, node: yaml.Node):
    # Tuples, slices and other python objects become plain lists / dicts / scalars.
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python_tag)


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=_TolerantLoader) or {}


def dump_yaml(data: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
