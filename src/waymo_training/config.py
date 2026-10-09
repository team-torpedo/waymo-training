"""YAML configuration loading with deep-merge, ${...} interpolation and CLI overrides."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

_REF = re.compile(r"\$\{([^}]+)\}")
_MISSING = object()


def repo_root() -> Path:
    """Repository root (parent of the `config/` directory by default)."""
    return Path(__file__).resolve().parents[2]


def deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _lookup(data: dict, dotted: str, default: Any = _MISSING) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            if default is _MISSING:
                raise KeyError(dotted)
            return default
    return cur


def _resolve_str(value: str, root: dict, depth: int = 0) -> Any:
    if depth > 20:
        raise ValueError(f"Config interpolation too deep / circular: {value!r}")

    def lookup(expr: str) -> Any:
        if expr.startswith("env:"):
            name, _, default = expr[4:].partition(",")
            val = os.environ.get(name.strip(), default if default != "" else None)
            if val is None:
                raise KeyError(f"Environment variable {name!r} is not set (and no default given)")
            return val
        val = _lookup(root, expr.strip())
        return _resolve(val, root, depth + 1)

    whole = _REF.fullmatch(value)
    if whole:  # keep native type for a pure reference
        return lookup(whole.group(1))
    return _REF.sub(lambda m: str(lookup(m.group(1))), value)


def _resolve(value: Any, root: dict, depth: int = 0) -> Any:
    if isinstance(value, str):
        return _resolve_str(value, root, depth)
    if isinstance(value, dict):
        return {k: _resolve(v, root, depth) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, root, depth) for v in value]
    return value


def _set_dotted(data: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = data
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
        if not isinstance(cur, dict):
            raise ValueError(f"Cannot set {dotted!r}: {part!r} is not a mapping")
    cur[parts[-1]] = value


class Config:
    """Read-only attribute / item view over a resolved config dictionary."""

    def __init__(self, data: dict, root: Path | None = None):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_root", Path(root) if root else repo_root())

    # ---- access ---------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Config has no key {name!r}") from None

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value, self._root) if isinstance(value, dict) else value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, dotted: str, default: Any = None) -> Any:
        value = _lookup(self._data, dotted, default)
        return Config(value, self._root) if isinstance(value, dict) else value

    def keys(self):
        return self._data.keys()

    def items(self):
        return ((k, self[k]) for k in self._data)

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    @property
    def root(self) -> Path:
        return self._root

    def path(self, dotted: str) -> Path:
        """Resolve a configured path (relative paths are relative to the repo root)."""
        raw = _lookup(self._data, dotted)
        p = Path(str(raw)).expanduser()
        return p if p.is_absolute() else (self._root / p).resolve()

    def dump(self) -> str:
        return yaml.safe_dump(self._data, sort_keys=False)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Override must look like key.sub=value, got {text!r}")
    key, _, raw = text.partition("=")
    return key.strip(), yaml.safe_load(raw)


def load_config(
    config_dir: str | Path | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load and merge every ``*.yaml`` in ``config_dir`` (``local.yaml`` last)."""
    root = repo_root()
    cdir = Path(config_dir) if config_dir else root / "config"
    if not cdir.is_absolute():
        cdir = (Path.cwd() / cdir).resolve()
    if not cdir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {cdir}")

    files = sorted(cdir.glob("*.yaml"), key=lambda p: (p.name == "local.yaml", p.name))
    if not files:
        raise FileNotFoundError(f"No *.yaml files in {cdir}")

    merged: dict = {}
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            part = yaml.safe_load(fh) or {}
        if not isinstance(part, dict):
            raise ValueError(f"{f} must contain a YAML mapping at the top level")
        merged = deep_merge(merged, part)

    for item in overrides or []:
        key, value = parse_override(item)
        _set_dotted(merged, key, value)

    resolved = _resolve(merged, merged)
    return Config(resolved, root)
