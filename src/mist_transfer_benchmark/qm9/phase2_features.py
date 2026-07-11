"""Frozen, sparse ECFP4 feature construction for the QM9 classical benchmark."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse

from .io import canonical_hash, canonical_json_bytes, sha256_file


class FeatureContractError(ValueError):
    """Raised when feature construction differs from the frozen protocol."""


@dataclass(frozen=True)
class MorganFeatureContract:
    radius: int
    fp_size: int
    count_simulation: bool
    include_chirality: bool
    use_bond_types: bool
    only_nonzero_invariants: bool
    include_ring_membership: bool
    count_bounds: None
    atom_invariants_generator: None
    bond_invariants_generator: None
    include_redundant_environments: bool

    def generator_kwargs(self) -> dict[str, object]:
        return {
            "radius": self.radius,
            "countSimulation": self.count_simulation,
            "includeChirality": self.include_chirality,
            "useBondTypes": self.use_bond_types,
            "onlyNonzeroInvariants": self.only_nonzero_invariants,
            "includeRingMembership": self.include_ring_membership,
            "countBounds": self.count_bounds,
            "fpSize": self.fp_size,
            "atomInvariantsGenerator": self.atom_invariants_generator,
            "bondInvariantsGenerator": self.bond_invariants_generator,
            "includeRedundantEnvironments": self.include_redundant_environments,
        }

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": "qm9-morgan-feature-contract-v1",
            "rdkit_api": "rdFingerprintGenerator.GetMorganGenerator",
            "molecule_parser": "Chem.MolFromSmiles(source_smiles, sanitize=True)",
            "fingerprint_method": "GetFingerprint(mol)",
            "parameters": asdict(self),
            "rdkit_runtime": rdkit.__version__,
        }


def contract_from_config(config: dict[str, object]) -> MorganFeatureContract:
    """Read and strictly validate every generator option frozen in TOML."""

    features = config["features"]
    exact_strings = {
        "representation": "binary Morgan fingerprint (ECFP4)",
        "rdkit_api": "rdFingerprintGenerator.GetMorganGenerator",
        "molecule_parser": "Chem.MolFromSmiles(source_smiles, sanitize=True)",
        "fingerprint_method": "GetFingerprint(mol)",
        "feature_matrix_storage": "scipy.sparse.csr_matrix",
        "feature_matrix_dtype": "float64",
        "feature_matrix_values": "binary-0.0-or-1.0",
        "tanimoto_storage": "RDKit ExplicitBitVect",
    }
    for key, expected in exact_strings.items():
        if features.get(key) != expected:
            raise FeatureContractError(f"features.{key} differs from the frozen protocol")
    if features.get("scalable_sparse_or_bounded_memory_implementation_required") is not True:
        raise FeatureContractError("the scalable sparse feature requirement is not enabled")
    none_sentinel = config["serialization"]["python_none_sentinel"]
    none_fields = (
        "count_bounds",
        "atom_invariants_generator",
        "bond_invariants_generator",
    )
    for key in none_fields:
        if features.get(key) != none_sentinel:
            raise FeatureContractError(f"features.{key} must use the frozen None sentinel")
    expected_types = {
        "radius": int,
        "fp_size": int,
        "count_simulation": bool,
        "include_chirality": bool,
        "use_bond_types": bool,
        "only_nonzero_invariants": bool,
        "include_ring_membership": bool,
        "include_redundant_environments": bool,
    }
    for key, expected_type in expected_types.items():
        if type(features.get(key)) is not expected_type:
            raise FeatureContractError(f"features.{key} has an invalid type")
    frozen_values = {
        "radius": 2,
        "fp_size": 2048,
        "count_simulation": False,
        "include_chirality": True,
        "use_bond_types": True,
        "only_nonzero_invariants": False,
        "include_ring_membership": True,
        "include_redundant_environments": False,
    }
    for key, expected in frozen_values.items():
        if features[key] != expected:
            raise FeatureContractError(f"features.{key} differs from the frozen protocol")
    if features["radius"] < 0 or features["fp_size"] <= 0:
        raise FeatureContractError("Morgan radius/fp_size must be valid positive dimensions")
    return MorganFeatureContract(
        radius=features["radius"],
        fp_size=features["fp_size"],
        count_simulation=features["count_simulation"],
        include_chirality=features["include_chirality"],
        use_bond_types=features["use_bond_types"],
        only_nonzero_invariants=features["only_nonzero_invariants"],
        include_ring_membership=features["include_ring_membership"],
        count_bounds=None,
        atom_invariants_generator=None,
        bond_invariants_generator=None,
        include_redundant_environments=features["include_redundant_environments"],
    )


def build_ecfp4_csr(
    source_smiles: Sequence[str],
    contract: MorganFeatureContract,
    *,
    progress: Callable[[str], None] | None = None,
) -> sparse.csr_matrix:
    """Build one float64 binary CSR row per source SMILES without filtering."""

    generator = rdFingerprintGenerator.GetMorganGenerator(**contract.generator_kwargs())
    indices: list[int] = []
    indptr = np.empty(len(source_smiles) + 1, dtype=np.int32)
    indptr[0] = 0
    for source_row_index, value in enumerate(source_smiles):
        molecule = Chem.MolFromSmiles(value, sanitize=True)
        if molecule is None:
            raise FeatureContractError(
                f"RDKit failed to parse source row {source_row_index}; rows may not be dropped"
            )
        on_bits = list(generator.GetFingerprint(molecule).GetOnBits())
        if on_bits != sorted(set(on_bits)):
            raise FeatureContractError(f"row {source_row_index} fingerprint bits are not canonical")
        indices.extend(on_bits)
        indptr[source_row_index + 1] = len(indices)
        if progress is not None and (source_row_index + 1) % 10_000 == 0:
            progress(f"fingerprinted {source_row_index + 1:,}/{len(source_smiles):,} rows")
    matrix = sparse.csr_matrix(
        (
            np.ones(len(indices), dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            indptr,
        ),
        shape=(len(source_smiles), contract.fp_size),
        dtype=np.float64,
    )
    matrix.sort_indices()
    validate_feature_matrix(matrix, rows=len(source_smiles), columns=contract.fp_size)
    return matrix


def validate_feature_matrix(matrix: sparse.spmatrix, *, rows: int, columns: int) -> None:
    if not sparse.isspmatrix_csr(matrix):
        raise FeatureContractError("feature matrix is not scipy.sparse.csr_matrix")
    if matrix.shape != (rows, columns):
        raise FeatureContractError(f"feature matrix shape {matrix.shape} is invalid")
    if matrix.dtype != np.dtype(np.float64):
        raise FeatureContractError(f"feature matrix dtype {matrix.dtype} is not float64")
    if matrix.has_canonical_format is not True:
        raise FeatureContractError("feature matrix is not in canonical CSR form")
    if matrix.nnz and not np.all(matrix.data == 1.0):
        raise FeatureContractError("feature matrix contains values outside binary 1.0 entries")
    if len(matrix.indptr) != rows + 1 or matrix.indptr[-1] != matrix.nnz:
        raise FeatureContractError("feature CSR indptr is inconsistent")


def csr_canonical_sha256(matrix: sparse.csr_matrix) -> str:
    """Hash CSR semantics independently of ZIP metadata or host byte order."""

    validate_feature_matrix(matrix, rows=matrix.shape[0], columns=matrix.shape[1])
    arrays = (
        ("indptr", np.asarray(matrix.indptr, dtype="<i8")),
        ("indices", np.asarray(matrix.indices, dtype="<i8")),
        ("data", np.asarray(matrix.data, dtype="<f8")),
    )
    digest = hashlib.sha256()
    header = {
        "schema_version": "qm9-csr-canonical-v1",
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "storage": "scipy.sparse.csr_matrix",
        "value_dtype": "float64-little-endian",
        "index_dtype": "int64-little-endian-for-hash",
        "nnz": int(matrix.nnz),
    }
    digest.update(canonical_json_bytes(header))
    digest.update(b"\n")
    for name, values in arrays:
        digest.update(
            canonical_json_bytes(
                {"name": name, "dtype": values.dtype.str, "length": int(values.size)}
            )
        )
        digest.update(b"\n")
        digest.update(values.tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def save_csr_atomic(path: str | Path, matrix: sparse.csr_matrix) -> dict[str, object]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        sparse.save_npz(temporary, matrix, compressed=True)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    loaded = sparse.load_npz(destination)
    validate_feature_matrix(loaded, rows=matrix.shape[0], columns=matrix.shape[1])
    canonical_sha256 = csr_canonical_sha256(matrix)
    if csr_canonical_sha256(loaded) != canonical_sha256:
        raise FeatureContractError("persisted feature matrix changed CSR semantics")
    return {
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "file_sha256": sha256_file(destination),
        "canonical_csr_sha256": canonical_sha256,
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "storage": "scipy.sparse.csr_matrix",
        "dtype": str(matrix.dtype),
        "values": "binary-0.0-or-1.0",
        "has_canonical_format": bool(matrix.has_canonical_format),
    }


def feature_manifest(
    contract: MorganFeatureContract,
    matrix_artifact: dict[str, object],
    *,
    source_row_identity_sha256: str,
    source_smiles_sha256: str,
) -> dict[str, object]:
    generator = contract.manifest()
    return {
        "schema_version": "qm9-feature-manifest-v1",
        "generator": generator,
        "generator_canonical_json_sha256": canonical_hash(generator),
        "source_row_identity_sha256": source_row_identity_sha256,
        "source_smiles_sha256": source_smiles_sha256,
        "row_order": "ascending zero-based source_row_index; no filtering or reordering",
        "matrix": matrix_artifact,
    }
