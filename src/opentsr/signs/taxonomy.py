"""Coarse taxonomy and per-dataset mapping tables (``configs/taxonomy/``).

``Taxonomy.load("coarse_v1")`` reads the family list; ``Taxonomy.mapping("gtsrb")`` reads
``map_gtsrb.yaml`` and returns a ``Mapping`` whose ``coarse_of(label)`` is what the L2 and L3
evaluators take as ``coarse_of``. Mapping files are validated against the family list at load
time, and a label absent from the table raises unless the mapping was loaded with a default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

TAXONOMY_DIR = Path(__file__).resolve().parents[3] / "configs" / "taxonomy"


@dataclass(frozen=True)
class Mapping:
    dataset: str
    taxonomy: str
    table: dict[str, str]
    default: str | None = None

    def coarse_of(self, label: str | int) -> str:
        key = str(label)
        if key in self.table:
            return self.table[key]
        if self.default is not None:
            return self.default
        raise KeyError(f"{self.dataset}: label {label!r} has no entry in map_{self.dataset}.yaml")

    @property
    def families_used(self) -> list[str]:
        return sorted(set(self.table.values()))


@dataclass(frozen=True)
class Taxonomy:
    name: str
    version: int
    families: list[str]
    descriptions: dict[str, dict]

    @classmethod
    def load(cls, name: str = "coarse_v1", root: Path = TAXONOMY_DIR) -> "Taxonomy":
        with open(root / f"{name}.yaml") as f:
            d = yaml.safe_load(f)
        fams = d["families"]
        return cls(name=name, version=int(d.get("version", 1)), families=list(fams), descriptions=dict(fams))

    def mapping(self, dataset: str, default: str | None = None, root: Path = TAXONOMY_DIR) -> Mapping:
        with open(root / f"map_{dataset}.yaml") as f:
            d = yaml.safe_load(f)
        if d.get("taxonomy", self.name) != self.name:
            raise ValueError(f"map_{dataset}.yaml targets taxonomy {d.get('taxonomy')!r}, not {self.name!r}")
        table = {str(k): str(v) for k, v in d["map"].items()}
        bad = {k: v for k, v in table.items() if v not in self.families}
        if bad:
            raise ValueError(f"map_{dataset}.yaml maps to unknown families: {bad}")
        if default is not None and default not in self.families:
            raise ValueError(f"default family {default!r} is not in {self.name}")
        return Mapping(dataset=d.get("dataset", dataset), taxonomy=self.name, table=table, default=default)

    def index(self, family: str) -> int:
        return self.families.index(family)
