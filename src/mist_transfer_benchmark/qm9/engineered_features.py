"""Raw, leakage-safe engineered features for repeated QM9 evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from scipy import sparse

GLOBAL_DESCRIPTOR_NAMES = (
    "mol_wt",
    "heavy_atoms",
    "all_atoms",
    "bonds",
    "rings",
    "rotatable_bonds",
    "h_donors",
    "h_acceptors",
    "tpsa",
    "logp",
    "fraction_csp3",
    "formal_charge",
    "carbon",
    "nitrogen",
    "oxygen",
    "fluorine",
    "aromatic_atoms",
)


def build_count_ecfp4_plus_globals(
    smiles: Sequence[str],
    *,
    fp_size: int = 2048,
    progress: Callable[[str], None] | None = None,
) -> sparse.csr_matrix:
    """Build unscaled count ECFP4 plus 17 physical/topological descriptors.

    Values are deliberately left raw. Scaling is fitted independently inside
    each model and split using training rows only.
    """

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=fp_size,
        includeChirality=True,
        useBondTypes=True,
        includeRingMembership=True,
    )
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    globals_matrix = np.empty((len(smiles), len(GLOBAL_DESCRIPTOR_NAMES)), dtype=np.float64)
    for row, raw_smiles in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(raw_smiles), sanitize=True)
        if molecule is None:
            raise ValueError(f"RDKit could not parse source row {row}")
        counts = generator.GetCountFingerprint(molecule).GetNonzeroElements()
        for column, count in sorted(counts.items()):
            rows.append(row)
            columns.append(int(column))
            values.append(float(count))
        atoms = list(molecule.GetAtoms())
        symbols = [atom.GetSymbol() for atom in atoms]
        globals_matrix[row] = [
            Descriptors.MolWt(molecule),
            molecule.GetNumHeavyAtoms(),
            sum(1 + atom.GetTotalNumHs() for atom in atoms),
            molecule.GetNumBonds(),
            Lipinski.RingCount(molecule),
            Lipinski.NumRotatableBonds(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Crippen.MolLogP(molecule),
            rdMolDescriptors.CalcFractionCSP3(molecule),
            Chem.GetFormalCharge(molecule),
            symbols.count("C"),
            symbols.count("N"),
            symbols.count("O"),
            symbols.count("F"),
            sum(atom.GetIsAromatic() for atom in atoms),
        ]
        if progress is not None and (row + 1) % 10_000 == 0:
            progress(f"engineered features: {row + 1:,}/{len(smiles):,}")
    count_matrix = sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(smiles), fp_size),
        dtype=np.float64,
    )
    result = sparse.hstack(
        (count_matrix, sparse.csr_matrix(globals_matrix)),
        format="csr",
        dtype=np.float64,
    )
    result.sort_indices()
    if result.shape != (len(smiles), fp_size + len(GLOBAL_DESCRIPTOR_NAMES)):
        raise ValueError("engineered feature matrix has an unexpected shape")
    if not np.all(np.isfinite(result.data)):
        raise ValueError("engineered feature matrix contains non-finite values")
    return result


def engineered_feature_schema(*, fp_size: int = 2048) -> dict[str, object]:
    return {
        "schema_version": "qm9-count-ecfp4-plus-globals-v1",
        "representation": "raw-count-ECFP4-plus-global-descriptors",
        "columns": fp_size + len(GLOBAL_DESCRIPTOR_NAMES),
        "fingerprint": {
            "method": "RDKit GetCountFingerprint",
            "radius": 2,
            "fp_size": fp_size,
            "include_chirality": True,
            "use_bond_types": True,
            "include_ring_membership": True,
        },
        "global_descriptors": list(GLOBAL_DESCRIPTOR_NAMES),
        "scaling": "fit-inside-each-model-on-that-cell-training-rows-only",
        "dtype": "float64",
        "storage": "scipy.sparse.csr_matrix",
    }
