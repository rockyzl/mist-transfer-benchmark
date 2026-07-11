"""Strict streaming validation for the exact QM9 CSV source."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from .constants import EXPECTED_HEADER, EXPECTED_ROW_COUNT, TARGET_COLUMNS
from .io import canonical_json_bytes


class QM9DataError(ValueError):
    """Raised when the local CSV violates the frozen Phase 1 contract."""


@dataclass(frozen=True)
class ValidatedQM9:
    header: tuple[str, ...]
    mol_ids: tuple[str, ...]
    source_smiles: tuple[str, ...]
    header_sha256: str
    source_row_index_sha256: str
    mol_id_sha256: str
    raw_smiles_sha256: str
    row_identity_sha256: str

    @property
    def row_count(self) -> int:
        return len(self.mol_ids)

    def record_id(self, source_row_index: int) -> str:
        return f"qm9:{source_row_index:06d}:{self.mol_ids[source_row_index]}"

    def metadata(self) -> dict[str, object]:
        return {
            "header": list(self.header),
            "header_sha256": self.header_sha256,
            "row_count": self.row_count,
            "target_columns": list(TARGET_COLUMNS),
            "source_row_index_sha256": self.source_row_index_sha256,
            "mol_id_sha256": self.mol_id_sha256,
            "raw_smiles_sha256": self.raw_smiles_sha256,
            "row_identity_sha256": self.row_identity_sha256,
            "column_hash_serialization": "one canonical JSON value/object per line, UTF-8, LF",
        }


def _update_json_line(digest: hashlib._Hash, value: object) -> None:
    digest.update(canonical_json_bytes(value))
    digest.update(b"\n")


def validate_qm9_csv(
    path: str | Path,
    *,
    expected_header: tuple[str, ...] = EXPECTED_HEADER,
    expected_rows: int = EXPECTED_ROW_COUNT,
) -> ValidatedQM9:
    """Validate exact header/order, stable IDs, and finite values without transforming rows."""

    mol_ids: list[str] = []
    smiles_values: list[str] = []
    seen_mol_ids: set[str] = set()
    index_digest = hashlib.sha256()
    mol_id_digest = hashlib.sha256()
    smiles_digest = hashlib.sha256()
    identity_digest = hashlib.sha256()

    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise QM9DataError("QM9 CSV is empty") from error
        if header != expected_header:
            raise QM9DataError(
                "QM9 header/order mismatch:\n"
                f"observed={list(header)!r}\nexpected={list(expected_header)!r}"
            )
        positions = {name: header.index(name) for name in (*TARGET_COLUMNS, "mol_id", "smiles")}
        for source_row_index, row in enumerate(reader):
            if len(row) != len(header):
                raise QM9DataError(
                    f"row {source_row_index} has {len(row)} fields; expected {len(header)}"
                )
            mol_id = row[positions["mol_id"]]
            smiles = row[positions["smiles"]]
            if not mol_id:
                raise QM9DataError(f"row {source_row_index} has an empty mol_id")
            if mol_id in seen_mol_ids:
                raise QM9DataError(f"duplicate mol_id at row {source_row_index}: {mol_id}")
            if not smiles:
                raise QM9DataError(f"row {source_row_index} has an empty smiles value")
            seen_mol_ids.add(mol_id)
            for target in TARGET_COLUMNS:
                raw_value = row[positions[target]]
                try:
                    value = float(raw_value)
                except ValueError as error:
                    raise QM9DataError(
                        f"row {source_row_index} target {target} is not numeric: {raw_value!r}"
                    ) from error
                if not math.isfinite(value):
                    raise QM9DataError(
                        f"row {source_row_index} target {target} is not finite: {raw_value!r}"
                    )
            mol_ids.append(mol_id)
            smiles_values.append(smiles)
            _update_json_line(index_digest, source_row_index)
            _update_json_line(mol_id_digest, mol_id)
            _update_json_line(smiles_digest, smiles)
            _update_json_line(
                identity_digest,
                {
                    "mol_id": mol_id,
                    "source_row_index": source_row_index,
                    "source_smiles": smiles,
                },
            )

    if len(mol_ids) != expected_rows:
        raise QM9DataError(f"parsed {len(mol_ids)} rows; expected exactly {expected_rows}")
    return ValidatedQM9(
        header=header,
        mol_ids=tuple(mol_ids),
        source_smiles=tuple(smiles_values),
        header_sha256=hashlib.sha256(canonical_json_bytes(list(header))).hexdigest(),
        source_row_index_sha256=index_digest.hexdigest(),
        mol_id_sha256=mol_id_digest.hexdigest(),
        raw_smiles_sha256=smiles_digest.hexdigest(),
        row_identity_sha256=identity_digest.hexdigest(),
    )


def load_qm9_identities(
    path: str | Path,
    *,
    expected_header: tuple[str, ...] = EXPECTED_HEADER,
    expected_rows: int = EXPECTED_ROW_COUNT,
) -> ValidatedQM9:
    """Read only identity fields after byte authentication; do not parse any target value."""

    mol_ids: list[str] = []
    smiles_values: list[str] = []
    seen_mol_ids: set[str] = set()
    index_digest = hashlib.sha256()
    mol_id_digest = hashlib.sha256()
    smiles_digest = hashlib.sha256()
    identity_digest = hashlib.sha256()
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise QM9DataError("QM9 CSV is empty") from error
        if header != expected_header:
            raise QM9DataError("QM9 identity source header/order mismatch")
        mol_id_offset = header.index("mol_id")
        smiles_offset = header.index("smiles")
        for source_row_index, row in enumerate(reader):
            if len(row) != len(header):
                raise QM9DataError(
                    f"row {source_row_index} has {len(row)} fields; expected {len(header)}"
                )
            mol_id = row[mol_id_offset]
            smiles = row[smiles_offset]
            if not mol_id or mol_id in seen_mol_ids:
                raise QM9DataError(f"invalid or duplicate mol_id at row {source_row_index}")
            if not smiles:
                raise QM9DataError(f"row {source_row_index} has an empty smiles value")
            seen_mol_ids.add(mol_id)
            mol_ids.append(mol_id)
            smiles_values.append(smiles)
            _update_json_line(index_digest, source_row_index)
            _update_json_line(mol_id_digest, mol_id)
            _update_json_line(smiles_digest, smiles)
            _update_json_line(
                identity_digest,
                {
                    "mol_id": mol_id,
                    "source_row_index": source_row_index,
                    "source_smiles": smiles,
                },
            )
    if len(mol_ids) != expected_rows:
        raise QM9DataError(f"parsed {len(mol_ids)} identity rows; expected {expected_rows}")
    return ValidatedQM9(
        header=header,
        mol_ids=tuple(mol_ids),
        source_smiles=tuple(smiles_values),
        header_sha256=hashlib.sha256(canonical_json_bytes(list(header))).hexdigest(),
        source_row_index_sha256=index_digest.hexdigest(),
        mol_id_sha256=mol_id_digest.hexdigest(),
        raw_smiles_sha256=smiles_digest.hexdigest(),
        row_identity_sha256=identity_digest.hexdigest(),
    )
