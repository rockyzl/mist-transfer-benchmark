# QM9 paper evaluation v1

This protocol turns the exploratory comparison into a repeated, auditable evaluation. It does
not claim to re-fine-tune MIST. Traditional models are trained locally; MIST can enter only as a
frozen prediction file whose bytes are hashed in the result.

## Frozen design

- Five predetermined seeds: `20260713`, `20260729`, `20260811`, `20260823`, `20260907`.
- Two primary outer splits: grouped-random and Bemis-Murcko scaffold-grouped. Grouped-random uses
  canonical connectivity identity, so exact, canonical, and connectivity-equivalent structures
  never cross partitions. Scaffold groups are merged with connectivity groups for the same reason.
- Each split is 80% train, 10% validation, and 10% test. Groups never cross partitions in the two
  structure-aware splits.
- Ridge, XGBoost, and MLP candidates are selected using validation normalized MAE only. The winner
  is refit on train+validation, then test is predicted once.
- A traditional ensemble is selected from validation predictions only. An all-model ensemble is
  created only in a future protocol where an eligible external model is explicitly enabled; that
  layer receives paired bootstrap delta CIs against MIST and the traditional ensemble.
- Paired-row percentile bootstrap gives a 95% confidence interval for the mean normalized MAE and
  each of the 12 target normalized MAEs. The full protocol uses 2,000 resamples.
- ECFP4 nearest-training Tanimoto is a fixed test-set stratification (`<0.3`, `0.3-0.6`,
  `0.6-0.8`, `>=0.8`), not a third split and never a selection signal.
- Every cell records selection, frozen-refit, inference, and wall time. A manifest and one atomic
  checkpoint per split/seed make interrupted runs resumable.
- `summary.json` aggregates every traditional model across seeds/splits; each cell includes a paired
  bootstrap delta CI for the traditional ensemble against its component models.

Exact nearest-training Tanimoto analysis is expensive. For a full run, compute/cache its values per
split cell in data preparation and provide them through the resolver interface. It must not be
recomputed inside hyperparameter search.

## Leakage boundary

Feature construction and structural group IDs may use SMILES but never target values. Candidate
choice uses only each cell's train and validation indices through `ArrayTargetLoader.load_selection`.
With repeated splits, a row that is test in one seed may legitimately be training data in another;
this protocol does not claim impossible physical isolation across all seeds. The hard guarantee is
that no cell selection API receives that cell's test indices, and **all** split/seed/model/ensemble
selections are globally frozen before any test metric or evaluation. The durable freeze-gate hash
is the only value that authorizes `load_test`. A
test metric must never be fed back into model, feature, weight, or threshold selection.
Imported external predictions, when enabled by a future protocol, must contain exact validation
and test prediction arrays plus both source-row-index arrays. Every cell is mandatory. Provenance
is exact-schema validated and hashed with the artifact. The enabled protocol must independently
freeze the expected artifact-manifest SHA-256 for every cell; self-reported provenance alone is not
accepted.

Phase 1 keeps external predictions disabled. The already released QM9 MIST predictions and any
artifact declaring `task_finetuned_on_qm9=true` are explicitly denied. A later protocol may admit
new MIST results only after its training provenance and split-specific fine-tuning are independently
reviewed; this phase does not pretend that work has happened.

Resume identity binds feature bytes/schema, SMILES, all group IDs, train/validation-visible target
bytes, immutable full-target artifact SHA-256, target provenance, external artifacts/provenance,
and cached similarity bytes. Protocol snapshot, input identity, freeze gate, selections, predictions,
similarity caches, cells, and summary are all content-hashed in the manifest. Any change is rejected.
Manifest events record selection writes, the global gate, similarity caches, test-cell completion,
and summary creation in sequence.

## Smoke run

```bash
uv run python scripts/run_qm9_paper_evaluation.py \
  --config configs/qm9_paper_evaluation_smoke.toml \
  --output results/qm9-paper-evaluation-smoke
```

Re-running the same command resumes completed cells and reuses each split cell's similarity cache.
Changing the config or any input identity while reusing an output directory is rejected.

For a prepared dataset, pass a safe NPZ containing numeric `x` `[rows, features]`, numeric `y`
`[rows, 12]`, and Unicode `smiles` `[rows]`. Do not use object arrays. For a full-scale run, also
include cached `scaffold_group_ids`; provide per-cell nearest-train similarity arrays through the
library resolver in the full run. The production config is
`configs/qm9_paper_evaluation_v1.toml`. The complete run is intentionally not started by CI or by
the smoke command.

Install the optional production dependency before enabling XGBoost:

```bash
uv sync --extra paper
```
