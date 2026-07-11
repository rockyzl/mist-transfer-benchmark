"""Frozen candidate construction, parameter hashing, and validation selection."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from .io import canonical_hash


class SelectionContractError(ValueError):
    """Raised when candidates or selection differ from the frozen protocol."""


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    estimator: object
    parameters: dict[str, object]
    parameters_sha256: str


def _candidate(candidate_id: str, estimator: object) -> Candidate:
    parameters = estimator.get_params(deep=False)
    return Candidate(candidate_id, estimator, parameters, canonical_hash(parameters))


def ridge_candidates(config: dict[str, object]) -> list[Candidate]:
    section = config["classical"]["ridge"]
    expected_order = ["alpha-0.01", "alpha-0.1", "alpha-1", "alpha-10", "alpha-100"]
    if section.get("candidate_order") != expected_order:
        raise SelectionContractError("Ridge candidate order differs from the frozen protocol")
    candidates = {item["id"]: item for item in section["candidates"]}
    result: list[Candidate] = []
    none_sentinel = config["serialization"]["python_none_sentinel"]
    if section.get("random_state") != none_sentinel:
        raise SelectionContractError("Ridge random_state must use the frozen None sentinel")
    for candidate_id in expected_order:
        item = candidates.get(candidate_id)
        if item is None or set(item) != {"id", "alpha"}:
            raise SelectionContractError(f"Ridge candidate {candidate_id} is invalid")
        estimator = Ridge(
            alpha=item["alpha"],
            fit_intercept=section["fit_intercept"],
            copy_X=section["copy_X"],
            max_iter=section["max_iter"],
            tol=section["tol"],
            solver=section["solver"],
            positive=section["positive"],
            random_state=None,
        )
        result.append(_candidate(candidate_id, estimator))
    return result


def random_forest_candidates(config: dict[str, object]) -> list[Candidate]:
    section = config["classical"]["random_forest"]
    order = section["candidate_order"]
    candidate_map = {item["id"]: item for item in section["candidates"]}
    none_sentinel = config["serialization"]["python_none_sentinel"]
    for field in ("max_depth", "max_leaf_nodes", "max_samples", "monotonic_cst"):
        if section.get(field) != none_sentinel:
            raise SelectionContractError(f"random_forest.{field} must use the None sentinel")
    result: list[Candidate] = []
    for candidate_id in order:
        item = candidate_map.get(candidate_id)
        if item is None or set(item) != {"id", "max_features", "min_samples_leaf"}:
            raise SelectionContractError(f"random-forest candidate {candidate_id} is invalid")
        estimator = RandomForestRegressor(
            n_estimators=section["n_estimators"],
            criterion=section["criterion"],
            max_depth=None,
            min_samples_split=section["min_samples_split"],
            min_samples_leaf=item["min_samples_leaf"],
            min_weight_fraction_leaf=section["min_weight_fraction_leaf"],
            max_features=item["max_features"],
            max_leaf_nodes=None,
            min_impurity_decrease=section["min_impurity_decrease"],
            bootstrap=section["bootstrap"],
            oob_score=section["oob_score"],
            n_jobs=section["n_jobs"],
            random_state=section["random_state"],
            verbose=section["verbose"],
            warm_start=section["warm_start"],
            ccp_alpha=section["ccp_alpha"],
            max_samples=None,
            monotonic_cst=None,
        )
        result.append(_candidate(candidate_id, estimator))
    return result


def select_first_minimum(
    results: list[dict[str, object]], candidate_order: list[str]
) -> dict[str, object]:
    by_id = {str(item["candidate_id"]): item for item in results}
    if set(by_id) != set(candidate_order) or len(results) != len(candidate_order):
        raise SelectionContractError("validation results do not cover the frozen candidate order")
    selected_id = min(
        candidate_order,
        key=lambda candidate_id: (
            float(by_id[candidate_id]["mean_normalized_mae_across_12_targets"]),
            candidate_order.index(candidate_id),
        ),
    )
    return by_id[selected_id]
