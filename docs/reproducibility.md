# Reproducibility protocol

For each scientific result, retain:

1. input CSV SHA-256 and completed data card;
2. benchmark repository commit, Git dirty state, and `uv.lock` SHA-256;
3. complete split-assignment CSV and SHA-256;
4. seed, fingerprint/model configuration, and tuning budget;
5. Python, RDKit, NumPy, pandas, and scikit-learn versions;
6. predictions for every row, not only summary metrics;
7. model/checkpoint revision and license for MIST experiments;
8. hardware, runtime, peak memory, and failure logs.

The v0.1.1 CLI writes the first six items it can determine into `run.json` and companion CSV files.
If the repository has no commit, `source_control.revision` is `null`; if files differ from the
revision, `source_control.dirty` is true. Either state is suitable for local testing but not a
reviewed result. The lock hash and source-control identity participate in the run fingerprint.

## Re-run checklist

```bash
uv sync --extra dev --frozen
uv run mist-transfer validate INTERNAL_ONLY_DATA.csv
uv run mist-transfer run-baseline INTERNAL_ONLY_DATA.csv \
  --output-dir results/RUN_NAME \
  --split scaffold \
  --seed 42
```

Repeat all pre-registered seeds and splits. Never overwrite a reviewed run. A new data hash, split,
dependency lock, or code revision creates a new result lineage.

Use the separate CSV containing `external_set=true` only with `--split external`. The other three
split strategies reject marked external rows so they cannot enter training accidentally.

The synthetic fixture is only a CI smoke test and must not appear in a scientific comparison.

## QM9 fixed-split v2 repeated evaluation

The completed v1 QM9 comparison used the released fine-tuned MIST predictor on its candidate
reconstructed fixed split. A later attempt to repeat evaluation on newly generated random and
scaffold splits was stopped: the released checkpoint may have seen members of those new test sets
during fine-tuning, so those runs are not a valid MIST comparison.

The pending v2 evaluation keeps the original fixed row membership and fixed MIST predictions. It
repeats only Ridge, XGBoost, MLP, and the validation-selected traditional ensemble across five
predetermined model seeds. MIST receives paired-row bootstrap uncertainty rather than artificial
seed variance. Structural reporting is limited to seen/unseen-scaffold subgroups inside the fixed
test cohort, not a new split or a promised Tanimoto analysis.

Before a reviewed v2 run, archive or verify all of the following:

1. authenticated QM9 source hash and fixed train/validation/test row-index hash;
2. frozen feature schema and hashes for count-ECFP4 plus 17 global descriptors;
3. fixed MIST validation/test prediction hashes and row-order agreement;
4. the five predetermined seeds and frozen traditional-model candidate configuration;
5. repository revision, clean/dirty status, dependency lock hash, environment, and hardware;
6. per-seed validation/test predictions, MLP curves, anomaly flags, runtime, and peak memory;
7. validation-only ensemble weights and an audit proving test labels were not used for selection;
8. paired bootstrap settings and fixed-test seen/unseen-scaffold subgroup definitions.

The runner, deterministic smoke, private preflight, and final QA revalidation are complete. The
commands below are the reviewed interface; their presence is not evidence that the full five-seed
v2 experiment has run.

Install the real-training backends before private preflight or execution:

```bash
uv sync --extra paper --frozen
```

The preflight fails closed if PyTorch or XGBoost is unavailable and records their versions plus
CUDA availability. CPU execution remains valid but will be materially slower than the GPU route.

Deterministic smoke:

```bash
.venv/bin/python scripts/run_qm9_fixed_split_evaluation.py \
  --config configs/qm9_fixed_split_evaluation_v2.toml \
  --output results/qm9-fixed-split-v2-smoke
```

Private-artifact preflight:

```bash
.venv/bin/python scripts/run_qm9_fixed_split_evaluation.py \
  --config configs/qm9_fixed_split_evaluation_v2.toml \
  --output results/qm9-fixed-split-v2-preflight-<ATTEMPT_ID> \
  --qm9-csv data/private/qm9/qm9.csv \
  --feature-matrix data/private/qm9/paper-evaluation-v1/feature_matrix.npz \
  --feature-manifest data/private/qm9/paper-evaluation-v1/manifest.json \
  --phase1-dir results/qm9-phase1-v2 \
  --mist-dir results/qm9-phase3-mist-v1 \
  --preflight
```

The full run uses the same private inputs, omits `--preflight`, and writes to a new permanent
output directory. The optional
`--mist-validation-predictions <STRICT_NPZ>` enables the supplemental all-model ensemble; omitting
it must record an explicit omission rather than fabricating validation predictions.

The runner is fail-closed. An interrupted or incomplete output is not resumable and must not be
overwritten or reused. Keep it as failure evidence and choose a fresh output directory for the
next preflight or full-run attempt.

Expected outputs are `manifest.json`, one record and prediction set per seed, `summary.json`, and
`loss-monitor.html`, as specified in
[`qm9_fixed_split_v2_live_tasks.md`](qm9_fixed_split_v2_live_tasks.md). Until the five-seed run and
its output verification are complete, v2 must be described as implemented but not yet executed.
