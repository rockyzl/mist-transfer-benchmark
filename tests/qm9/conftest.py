from __future__ import annotations

import csv
from pathlib import Path

from mist_transfer_benchmark.qm9.constants import EXPECTED_HEADER, TARGET_COLUMNS


def write_qm9_csv(path: Path, smiles: list[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(EXPECTED_HEADER)
        for index, value in enumerate(smiles):
            row = {column: str(index + 1) for column in EXPECTED_HEADER}
            row.update(
                {
                    target: str(index + offset / 10)
                    for offset, target in enumerate(TARGET_COLUMNS)
                }
            )
            row["mol_id"] = f"gdb_{index + 1}"
            row["smiles"] = value
            writer.writerow([row[column] for column in EXPECTED_HEADER])
    return path
