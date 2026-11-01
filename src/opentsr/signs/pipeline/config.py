"""System configuration: one mapping per stage slot, ``null`` to switch a stage off.

A configuration file looks like ``configs/system/full.yaml``. ``load_config`` fills missing slots
with ``null`` and rejects unknown keys so that a typo in a stage name cannot silently disable it.
``ablation_variants`` produces the configurations the ablation driver runs: all on, one off at a
time, one on at a time (relative to the academic baseline) and named presets.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from opentsr.signs.pipeline.stages import STAGE_ORDER

SystemConfig = dict[str, Any]

META_KEYS = {"name", "description", "presets"}


def normalize(cfg: dict[str, Any]) -> SystemConfig:
    unknown = set(cfg) - set(STAGE_ORDER) - META_KEYS
    if unknown:
        raise KeyError(f"unknown keys in system config: {sorted(unknown)}; stages are {STAGE_ORDER}")
    out: SystemConfig = {k: cfg.get(k) for k in META_KEYS if k in cfg}
    for slot in STAGE_ORDER:
        val = cfg.get(slot)
        if val is not None and not isinstance(val, dict):
            raise TypeError(f"stage {slot!r} must be a mapping or null, got {type(val).__name__}")
        out[slot] = copy.deepcopy(val)
    if out["detector"] is None:
        raise ValueError("the detector stage cannot be null")
    return out


def load_config(path: str | Path) -> SystemConfig:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("name", Path(path).stem)
    return normalize(cfg)


def stages_on(cfg: SystemConfig) -> list[str]:
    return [s for s in STAGE_ORDER if cfg.get(s) is not None]


def with_stage(cfg: SystemConfig, slot: str, value: dict[str, Any] | None, name: str) -> SystemConfig:
    out = copy.deepcopy(cfg)
    out[slot] = copy.deepcopy(value)
    out["name"] = name
    return out


def ablation_variants(full: SystemConfig, base: SystemConfig | None = None) -> list[SystemConfig]:
    """All on; one off at a time; one on at a time (over ``base``); plus ``full["presets"]``.

    Presets are a mapping ``name -> {slot: config-or-null}`` applied on top of ``full``.
    """
    out = [with_stage(full, "detector", full["detector"], "full")]
    switchable = [s for s in STAGE_ORDER if s != "detector" and full.get(s) is not None]
    for s in switchable:
        out.append(with_stage(full, s, None, f"full-minus-{s}"))
    if base is not None:
        out.append(with_stage(base, "detector", base["detector"], base.get("name", "base")))
        for s in switchable:
            if base.get(s) is None:
                out.append(with_stage(base, s, full[s], f"base-plus-{s}"))
    for name, overrides in (full.get("presets") or {}).items():
        c = copy.deepcopy(full)
        for slot, value in overrides.items():
            if slot not in STAGE_ORDER:
                raise KeyError(f"preset {name!r}: unknown stage {slot!r}")
            c[slot] = copy.deepcopy(value)
        c["name"] = name
        out.append(c)
    return out
