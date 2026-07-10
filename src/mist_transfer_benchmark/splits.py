"""Deterministic, molecule-aware split strategies."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass(frozen=True)
class SplitConfig:
    strategy: str = "scaffold"
    seed: int = 42
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    group_column: str = "chemical_family"
    external_column: str = "external_set"

    def __post_init__(self) -> None:
        if self.strategy not in {"external", "group", "random", "scaffold"}:
            raise ValueError(f"unknown split strategy: {self.strategy}")
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(value <= 0 for value in fractions):
            raise ValueError("all split fractions must be positive")
        if not np.isclose(sum(fractions), 1.0):
            raise ValueError("split fractions must sum to 1.0")


def bemis_murcko_scaffold(smiles: str) -> str:
    """Compute a canonical Bemis–Murcko key with explicit acyclic handling."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES passed to scaffold splitting: {smiles!r}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True)
    if scaffold:
        return scaffold
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return f"__ACYCLIC__:{canonical}"


def _allocate_groups(
    labels: pd.Series,
    fractions: dict[str, float],
    seed: int,
) -> pd.Series:
    groups: dict[str, list[object]] = defaultdict(list)
    for index, value in labels.items():
        text = str(value)
        if not text:
            raise ValueError("split group labels must not be empty")
        groups[text].append(index)

    split_names = list(fractions)
    if len(groups) < len(split_names):
        raise ValueError(
            f"need at least {len(split_names)} distinct groups for {len(split_names)} splits; "
            f"found {len(groups)}"
        )

    rng = np.random.default_rng(seed)
    tie_breakers = {name: float(rng.random()) for name in groups}
    ordered_groups = sorted(groups, key=lambda name: (-len(groups[name]), tie_breakers[name]))
    targets = {name: fraction * len(labels) for name, fraction in fractions.items()}
    counts = dict.fromkeys(split_names, 0)
    group_to_split: dict[str, str] = {}

    # Seed every requested partition before greedy allocation. This guarantees
    # non-empty splits when enough indivisible groups exist, including highly
    # imbalanced cases such as 50/40/10.
    seeded = min(len(split_names), len(ordered_groups))
    for split_name, group in zip(split_names, ordered_groups[:seeded], strict=True):
        group_to_split[group] = split_name
        counts[split_name] += len(groups[group])

    for group in ordered_groups[seeded:]:
        size = len(groups[group])
        destination = max(
            split_names,
            key=lambda name: (
                targets[name] - counts[name],
                targets[name],
                -split_names.index(name),
            ),
        )
        group_to_split[group] = destination
        counts[destination] += size

    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise ValueError(f"split allocation produced empty partitions: {empty}")

    assignments = pd.Series(index=labels.index, dtype="object", name="split")
    for group, indices in groups.items():
        assignments.loc[indices] = group_to_split[group]
    return assignments


def _molecule_key(frame: pd.DataFrame) -> pd.Series:
    """Prefer connectivity InChIKey so protomers/stereoisomers remain grouped."""

    if "connectivity_key" in frame.columns:
        return frame["connectivity_key"]
    return frame["canonical_smiles"]


def _check_molecule_leakage(frame: pd.DataFrame, assignments: pd.Series) -> None:
    audit = pd.DataFrame(
        {"molecule_key": _molecule_key(frame), "split": assignments},
        index=frame.index,
    )
    split_counts = audit.groupby("molecule_key", sort=False)["split"].nunique()
    leaking = split_counts[split_counts > 1]
    if not leaking.empty:
        examples = ", ".join(leaking.index[:3])
        raise ValueError(f"molecular connectivity leakage across splits: {examples}")


def make_split(frame: pd.DataFrame, config: SplitConfig) -> pd.Series:
    """Assign every row to train, validation, or test without molecule leakage."""

    if "canonical_smiles" not in frame.columns:
        raise ValueError("frame must contain canonical_smiles; use load_validated_csv")

    fractions = {
        "train": config.train_fraction,
        "validation": config.validation_fraction,
        "test": config.test_fraction,
    }

    if config.strategy != "external" and "external_set" in frame.columns:
        if frame["external_set"].astype(bool).any():
            raise ValueError(
                "non-external split strategies refuse rows marked external_set=true; "
                "use a filtered internal-only CSV or --split external"
            )

    if config.strategy == "random":
        assignments = _allocate_groups(_molecule_key(frame), fractions, config.seed)
    elif config.strategy == "scaffold":
        scaffolds = frame["canonical_smiles"].map(bemis_murcko_scaffold)
        assignments = _allocate_groups(scaffolds, fractions, config.seed)
    elif config.strategy == "group":
        if config.group_column not in frame.columns:
            raise ValueError(f"group column does not exist: {config.group_column}")
        invalid_groups = (
            frame[config.group_column].astype(str).str.strip().isin({"", "not_reported"})
        )
        if invalid_groups.any():
            raise ValueError(
                f"group split requires reported values in {config.group_column}; "
                f"found {int(invalid_groups.sum())} missing/not_reported rows"
            )
        per_molecule_groups = frame.assign(_molecule_key=_molecule_key(frame)).groupby(
            "_molecule_key"
        )[config.group_column].nunique()
        if (per_molecule_groups > 1).any():
            raise ValueError(
                "a canonical molecule has multiple group labels; "
                "consolidate group IDs before splitting"
            )
        assignments = _allocate_groups(frame[config.group_column], fractions, config.seed)
    else:
        if config.external_column not in frame.columns:
            raise ValueError(f"external column does not exist: {config.external_column}")
        external = frame[config.external_column].astype(bool)
        if not external.any() or external.all():
            raise ValueError("external split needs at least one internal and one external row")
        per_molecule_external = frame.assign(_molecule_key=_molecule_key(frame)).groupby(
            "_molecule_key"
        )[config.external_column].nunique()
        if (per_molecule_external > 1).any():
            raise ValueError("a molecular connectivity appears in both internal and external sets")
        for column in ("source_id", "group_id"):
            internal_values = set(frame.loc[~external, column].astype(str))
            external_values = set(frame.loc[external, column].astype(str))
            overlap = sorted(internal_values & external_values)
            if overlap:
                raise ValueError(
                    f"internal/external {column} overlap is forbidden: {overlap[:5]}"
                )

        assignments = pd.Series(index=frame.index, dtype="object", name="split")
        assignments.loc[external] = "test"
        internal = frame.loc[~external]
        internal_total = config.train_fraction + config.validation_fraction
        internal_fractions = {
            "train": config.train_fraction / internal_total,
            "validation": config.validation_fraction / internal_total,
        }
        assignments.loc[~external] = _allocate_groups(
            _molecule_key(internal), internal_fractions, config.seed
        )

    _check_molecule_leakage(frame, assignments)
    if assignments.isna().any():
        raise RuntimeError("split assignment left unassigned rows")
    empty = [name for name, count in split_counts(assignments).items() if count == 0]
    if empty:
        raise ValueError(f"split strategy produced empty partitions: {empty}")
    return assignments


def split_group_keys(frame: pd.DataFrame, config: SplitConfig) -> pd.Series:
    """Return the exact grouping key used to create each split assignment."""

    if config.strategy == "random":
        return _molecule_key(frame).astype(str).rename("split_group_key")
    if config.strategy == "scaffold":
        return frame["canonical_smiles"].map(bemis_murcko_scaffold).rename("split_group_key")
    if config.strategy == "group":
        return frame[config.group_column].astype(str).rename("split_group_key")
    external = frame[config.external_column].astype(bool)
    keys = _molecule_key(frame).astype(str).copy()
    keys.loc[external] = (
        "__EXTERNAL__:"
        + frame.loc[external, "source_id"].astype(str)
        + ":"
        + frame.loc[external, "group_id"].astype(str)
    )
    return keys.rename("split_group_key")


def split_overlap_report(frame: pd.DataFrame, assignments: pd.Series) -> dict[str, object]:
    """Expose held-out scaffold and family overlap with training data."""

    train = assignments == "train"
    scaffolds = frame["canonical_smiles"].map(bemis_murcko_scaffold)
    report: dict[str, object] = {}
    for split_name in ("validation", "test"):
        held_out = assignments == split_name
        split_report: dict[str, object] = {}
        for label, values in (
            ("scaffold", scaffolds),
            ("chemical_family", frame["chemical_family"].astype(str)),
        ):
            train_values = set(values.loc[train]) - {"not_reported"}
            held_values = set(values.loc[held_out]) - {"not_reported"}
            overlap = sorted(train_values & held_values)
            split_report[label] = {
                "count": len(overlap),
                "values": overlap[:100],
                "truncated": len(overlap) > 100,
            }
        report[split_name] = split_report
    return report


def split_counts(assignments: pd.Series) -> dict[str, int]:
    """Return stable split counts including absent standard partitions."""

    counts = assignments.value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in ("train", "validation", "test")}
