# QM9 fixed-split v2 live tasks

Status date: 2026-07-18

## Objective

Upgrade the completed single-run comparison into a repeated evaluation that keeps the released
fine-tuned MIST checkpoint on its reconstructed fixed QM9 split. Repeat only the locally trained
traditional pipeline across five seeds; never move the fixed MIST artifact onto new random or
scaffold splits.

## Frozen interface

Inputs:

- authenticated QM9 CSV and reconstructed fixed train/validation/test row indices;
- raw count-ECFP4 plus 17 global descriptors;
- existing fixed MIST test predictions in source-row order; fixed MIST validation predictions are
  optional and required only for the supplemental all-model ensemble;
- frozen traditional candidate choices from the completed v1 comparison.

Outputs:

- `manifest.json`: exact hashes for split, source, features, code/config, and MIST predictions;
- `seeds/<seed>.json`: Ridge, XGBoost, MLP, traditional ensemble, and supplemental all-model
  ensemble validation/test records;
- `predictions/<seed>-<model>.npy`: test predictions in frozen test-row order;
- `summary.json`: method mean/std across seeds, fixed MIST bootstrap interval, paired deltas versus
  MIST, all 12 target metrics, runtime, and seen/unseen-scaffold test subgroups;
- `loss-monitor.html`: per-seed MLP training and validation-error curves with anomaly flags.

An incomplete output directory is fail-closed and is not resumable. After any failed or interrupted
attempt, preserve it for diagnosis and start the next attempt in a fresh output directory.

## Acceptance criteria

- [ ] Existing released MIST is inference-only and evaluated only on its compatible fixed split.
- [ ] Five predetermined traditional-model seeds run on the same split.
- [ ] Hyperparameters and feature schema are frozen before test access.
- [ ] MLP scales all 12 targets using training labels only and records learning curves.
- [ ] Ensemble weights use validation predictions only.
- [ ] Test labels are read only after all seed selections and ensemble weights are frozen.
- [ ] Fixed MIST gets paired-row bootstrap uncertainty, not fake seed variance.
- [ ] Structural analysis reports only seen/unseen-scaffold subgroups inside the fixed test set.
- [ ] Tests, lint, smoke run, and artifact-schema checks pass.
- [ ] Durable docs distinguish completed evidence from planned work.

## Team ownership

- Statistical protocol worker: `docs/qm9_fixed_split_v2_protocol.md` only.
- Experiment engineer: runner/library/config/tests, excluding coordination/protocol docs.
- Doc worker: this live task file plus reproducibility/README handoff only.
- Lead: contracts, integration, QA coordination, final validation, commit, and push.

## Progress

- [x] Existing v1 evidence and failure mode identified.
- [x] MLP target-scaling and curve-monitoring primitives implemented on the working branch.
- [x] Statistical protocol drafted, independently reviewed, and reconciled after QA.
- [x] Fixed-split v2 runner implemented.
- [x] Reproducibility and README handoff updated without claiming v2 results.
- [x] Independent QA completed; requested changes were implemented and lead-revalidated.
- [x] Deterministic smoke and hardened private-artifact preflight completed.
- [ ] Full five-seed run executed (not started).

## Decisions

- The primary comparison is MIST versus individual traditional models and the traditional-only
  ensemble. The all-model ensemble remains supplemental.
- A repeated seed changes model stochasticity, not data membership.
- A true new scaffold split requires new split-specific MIST fine-tuning and is outside v2.

## Current handoff status

The completed v1 fixed-split result remains the only scientific result reported by this repository.
A later repeated experiment that created new random and scaffold splits was stopped because the
released fine-tuned MIST checkpoint could not be evaluated on those changed splits without an
unknown overlap with its fine-tuning data. Its outputs must not be presented as a MIST comparison.

The v2 runner and preflight path are implemented. Deterministic smoke, private-artifact preflight,
independent QA, the targeted correction round, and lead revalidation completed. No full five-seed
summary, bootstrap interval, or seen/unseen-scaffold result has been produced. The runner interface
is recorded in `docs/reproducibility.md` and is ready for a fresh, reviewed scientific run.

## Durable next-session assumptions

- Keep the candidate reconstructed MIST train/validation/test membership byte-for-byte fixed.
- Treat released MIST predictions as one immutable inference artifact, not five seeded model runs.
- Repeat only locally trained traditional models with the five preregistered seeds.
- Freeze features, candidate hyperparameters, selection rules, and ensemble construction before
  reading fixed-test labels.
- Fit MLP target scaling on training labels only; restore predictions to original units for metrics.
- Use validation predictions only for model selection and ensemble weights.
- Estimate MIST uncertainty by paired resampling of the same fixed test rows.
- Report only seen/unseen-scaffold subgroups inside the fixed test cohort; v2 does not promise a
  Tanimoto or continuous-similarity analysis.
- Keep the traditional-only ensemble primary and the ensemble that includes MIST supplemental.
- Treat incomplete outputs as failed attempts: do not resume or overwrite them, and use a fresh
  output directory for every retry.
- Do not update the article or scientific headline until artifacts pass schema checks and
  independent QA.
