# MIST Transfer Benchmark

[**Live benchmark explorer**](https://rockyzl.github.io/mist-transfer-benchmark/)
· [**Official MIST notebooks**](https://github.com/BattModels/mist-demo)

> **Site status:** The [GitHub Pages explorer](https://rockyzl.github.io/mist-transfer-benchmark/)
> now serves the owner-authorized aggregate-only QM9 results view alongside the separate synthetic
> redox explorer. The browser does not execute MIST.

> Does large-scale SMILES pretraining learn chemistry that transfers to new molecular
> families, or does it mainly provide an expensive way to encode molecular similarity?

This repository now keeps two related but deliberately separate research tracks:

| Track | Question | Status |
|---|---|---|
| **QM9-28M released-predictor benchmark** | How does the already fine-tuned public MIST-28M QM9 predictor compare with classical models trained on the same candidate split reconstructed from public MIST code? | Local inference, locked comparison, aggregate report, and static results view complete; aggregate-only publication authorized |
| **Redox transfer study** | Can a pretrained MIST encoder, after task-specific downstream training, transfer beyond similar redox molecules? | Redox validation, split, and classical-baseline infrastructure implemented; MIST training not implemented |

The QM9 track is the first planned executable comparison. Its canonical
[`benchmark process`](docs/qm9_28m_benchmark_process.md),
[`data card`](docs/qm9_28m_data_card.md), and
[`machine-readable protocol`](configs/qm9_28m.toml) preregister the comparison before any test
score is calculated. It uses the already fine-tuned 12-output MIST checkpoint for inference only;
it does not compare a raw foundation checkpoint or retrain MIST. The available evidence supports a
candidate reconstruction of the checkpoint's split, not a claim that the publisher released exact
row memberships.

The original redox track turns the broader transfer question into a falsifiable benchmark.
Redox-potential regression is useful for organic molecules and battery electrolytes while also
being unusually easy to evaluate incorrectly. The redox-specific material below remains active and
is not silently repurposed as a QM9 data contract.

## The question in plain language

Suppose a model sees millions of unlabeled SMILES during pretraining. Later, we give it a
small table of molecules and measured reduction potentials. A good score can mean two very
different things:

1. The model learned reusable chemical structure patterns.
2. The test molecules look almost the same as molecules in the training split.

Random train/test splits often cannot distinguish these explanations. This project compares
the same methods on progressively harder splits and records how similar every held-out molecule
is to its nearest training molecule.

The benchmark never treats the raw MIST foundation checkpoint as a redox predictor. Each planned
MIST arm starts from an existing public pretrained checkpoint hosted on Hugging Face, attaches and
trains a scalar regression head, and uses the same labeled redox training rows as the classical
models. Depending on the arm, the encoder stays frozen, LoRA/adapter updates are trained alongside
the regression head, or the full encoder is fine-tuned with that head. The resulting task-specific
predictor—not the raw checkpoint—is compared with ECFP ridge, random-forest, and nearest-neighbor
predictors. In a scientific run, all methods are evaluated against the same held-out measured values.

```text
same labeled redox training rows
    ├── ECFP → ridge / random forest / nearest neighbor → classical predictor
    └── pretrained MIST → attach regression head → train head (+ optional LoRA/encoder updates)
                                                              → MIST predictor

same held-out molecules → both predictors → compare predictions with measured redox values
```

This is a practical comparison of final task predictors; it is not a comparison between a raw
foundation model and a trained classical model. This redox MIST execution remains planned; the
separate released-predictor QM9 inference is complete.

## Pre-registered hypotheses

- **Transfer hypothesis:** a pretrained molecular encoder should retain an advantage over ECFP
  baselines on scaffold, chemical-family, and genuinely external test sets.
- **Similarity hypothesis:** if the advantage disappears as train/test similarity falls, the
  model may primarily be a costly similarity representation for this task.
- **Data-quality hypothesis:** inconsistent solvent, reference electrode, pH, temperature,
  protonation, or charge can dominate architecture choice. A large model cannot repair an
  undefined target.

These remain untested hypotheses for the redox track. The QM9 track now has a preliminary local
released-predictor result, but it does not answer the harder redox transfer question.

## What works now

- a preregistered QM9-28M process plus completed local source, exact Datasets-split comparison,
  RDKit duplicate audit, locked classical result, reviewed released-MIST inference, and one-shot
  comparison;
- a strict redox-couple CSV validator that preserves measurement conditions, original values,
  conversion provenance, and molecular identity;
- deterministic molecule-grouped random, Bemis–Murcko scaffold, user-defined group, and
  external-set splits;
- RDKit ECFP fingerprints with mean, Tanimoto 1-NN, ridge, and random-forest regressors;
- a default comparability gate that rejects mixed target definitions, source types, conditions,
  and computation/measurement protocols before molecular-only fitting;
- leakage checks and nearest-training-molecule Tanimoto diagnostics;
- machine-readable run metadata, split assignments, predictions, and metrics;
- a local static, accessible QM9 aggregate-results view plus a separate explorer for comparing
  split behavior on synthetic redox fixtures;
- a browser-only SMILES prediction client with a four-model response contract; it stays disabled
  until a separately operated local/private API base URL is configured;
- a synthetic fixture for testing the software only.

QM9 MIST inference is implemented against one fixed-revision, ignored local snapshot; weights and
QM9 source rows are not redistributed. Redox MIST downstream training is not implemented, and no
private or real redox dataset is bundled. See
[`docs/mist_integration.md`](docs/mist_integration.md) for the verified integration boundary.

### Preliminary QM9 result

The expanded independent comparison includes tuned XGBoost, MLP, engineered
Ridge, and validation-selected ensembles using count ECFP4 plus 17 global
molecular descriptors. On the same 13,389-row candidate test cohort, mean
normalized MAE was `0.087400` for the traditional-only XGBoost/MLP/Ridge
ensemble, `0.094336` for XGBoost, `0.095064` for the released fine-tuned MIST
checkpoint, `0.100364` for MLP, and `0.149839` for engineered Ridge. XGBoost
and MIST should be read as near parity in this single point estimate. As a
second-layer systems result, adding MIST to the ensemble improves the score to
`0.081159`, indicating complementary errors. The traditional-only reporting
correction is explicitly post-specified because it was made after the first
test report; its weights still use validation labels only. See the
[`extended comparison report`](docs/qm9_extended_comparison_v1.md) for the
training process, feature/hyperparameter selection, runtime, per-target scope,
and limitations.

The released 12-output MIST checkpoint was run once on all `13,389` candidate reconstructed test
rows after a separate 128-row train/validation smoke test. Lower is better for the aggregate mean
normalized MAE:

| Cohort | Released MIST | Locked ECFP Ridge | MIST − Ridge |
|---|---:|---:|---:|
| Complete test | `0.0950643233` | `0.3700485581` | `-0.2749842348` |
| Duplicate-clean test | `0.0951035618` | `0.3700476436` | `-0.2749440818` |

That is a `74.31%` reduction in aggregate error on the complete cohort and `74.30%` on the
duplicate-clean cohort. MIST also had lower MAE on each of the 12 targets in this comparison.

This is a comparison on a **candidate split reconstructed from public MIST code**, not a claim that
the publisher certified these row memberships or that this reproduces the checkpoint's official
test score. It uses QM9 DFT-computed labels, not experimental battery data. Point estimates remain
preliminary because no uncertainty method was preregistered. Complete metrics, hashes, commands,
runtime evidence, and rights restrictions are in the focused
[`QM9 result report`](docs/qm9_28m_results.md). The static view consumes the reproducible,
aggregate-only [`result summary`](site/qm9-results.json); the complete decision trail remains in
the [`QM9 process document`](docs/qm9_28m_benchmark_process.md).

### Interactive prediction client

The static page includes a client for comparing Ridge, XGBoost, MLP, and MIST predictions across
all 12 QM9 properties. It sends one `{"smiles":"CCO"}` request to a configured
`POST /v1/predict` endpoint and highlights HOMO, LUMO, and gap while retaining the complete table.
The default [`site/live-config.js`](site/live-config.js) contains no API URL, credentials, model
path, or weight reference, so the form remains visibly disabled on an ordinary static deployment.
The exact browser/backend boundary is documented in the
[`live prediction client contract`](docs/live_demo_client_contract.md).

The QM9 Phase 1 command deliberately keeps Datasets 3.2.0 in a separate temporary environment, then
requires exact membership agreement before writing ignored local artifacts:

```bash
uv venv --python 3.12 /tmp/qm9-datasets-3.2.0-audit
uv pip install --python /tmp/qm9-datasets-3.2.0-audit/bin/python \
  "datasets==3.2.0" "numpy==2.5.1"

uv run mist-transfer qm9-audit \
  --config configs/qm9_28m.toml \
  --cache-dir data/private/qm9 \
  --output-dir results/qm9-phase1 \
  --datasets-python /tmp/qm9-datasets-3.2.0-audit/bin/python
```

This command downloads the CSV but no model weights. Raw data, row manifests, and generated audit
artifacts remain ignored; only verified metadata and hashes are recorded in the protocol documents.

## Quick start

The commands below assume a source checkout of this repository with `uv.lock` and the synthetic
fixtures present.

Install the development environment:

```bash
uv sync --extra dev
```

Validate the synthetic fixture:

```bash
uv run mist-transfer validate data/fixtures/redox_tiny_internal.csv
```

Run a scaffold-split software smoke test:

```bash
uv run mist-transfer run-baseline data/fixtures/redox_tiny_internal.csv \
  --output-dir results/smoke-scaffold \
  --split scaffold \
  --seed 42
```

The output directory contains:

```text
run.json                 configuration, versions, hashes, counts, and metrics
predictions.csv          one prediction per molecule, split, and model
split_assignments.csv    auditable train/validation/test assignment
```

The internal fixture contains no rows marked external. `redox_tiny.csv` additionally contains a
synthetic external family for exercising `--split external`. Random, scaffold, and group strategies
deliberately reject any input containing `external_set=true`; filter a real dataset into an
immutable internal-only CSV before using those strategies.

The fixture values are synthetic and deliberately have no scientific meaning. A successful run
proves only that the pipeline works. See [`data/fixtures/NOTICE`](data/fixtures/NOTICE) for its CC0
legal notice.

The [deployed benchmark explorer](https://rockyzl.github.io/mist-transfer-benchmark/) serves the
aggregate-only QM9 result and synthetic redox track. Static hosting does not run the private live
models; that client requires an explicitly configured prediction API. With the authenticated,
ignored Phase 2/3 aggregate artifacts present, both tracked data files are reproducible:

```bash
uv run python scripts/build_qm9_results.py
uv run python scripts/build_demo_data.py
git diff --exit-code -- site/qm9-results.json
git diff --exit-code -- site/demo-data.json
```

## Redox experiment ladder

| Stage | Representation and model | Status |
|---|---|---|
| A | Mean; ECFP Tanimoto 1-NN; ECFP + ridge/random forest | Implemented |
| B | Public pretrained MIST encoder (frozen) + label-trained linear or MLP head | Planned |
| C | Public pretrained MIST + LoRA and regression head, downstream-trained on labels | Planned |
| D | Public pretrained MIST + full downstream fine-tuning | Future only; requires a justified data/compute budget |

Every stage must use the same immutable row IDs, split assignments, target definition, and
evaluation metrics. Full details are in [`docs/experiment_matrix.md`](docs/experiment_matrix.md).

## Why the data contract is strict

`-0.4 V` is not a complete scientific label. It only becomes interpretable with context such as:

```text
redox couple + modeled molecular state + electron/proton count + multiplicities
+ reduction/oxidation + potential definition + original/target reference electrode
+ solvent + supporting electrolyte + pH + temperature + conversion provenance
```

The required columns and allowed missing-value markers are documented in
[`docs/data_contract.md`](docs/data_contract.md). Do not silently convert reference electrodes
or merge measurements made under different conditions.

## Research safeguards

- Canonically identical molecules cannot cross splits.
- Scaffold and group splits move whole groups, never individual rows.
- External data remain test-only.
- Internal and external partitions cannot share `source_id` or `group_id`.
- Molecular-only models reject heterogeneous target cohorts by default. The deliberately alarming
  `--unsafe-allow-condition-ignorant-mixing` flag records an override but never permits mixed
  `source_type` values.
- Hyperparameters must be selected without looking at the test set.
- Real data need a completed data card and redistribution permission.
- Results must report all planned splits and seeds, not only the best run.
- Similarity-stratified errors accompany aggregate metrics.

See [`docs/leakage_controls.md`](docs/leakage_controls.md) and
[`docs/reproducibility.md`](docs/reproducibility.md).

## Project scope

The emitted machine-readable redox claim gate remains closed in v0.1.1. MIST comparisons, learning
curves, repeated-seed aggregation, bootstrap intervals, uncertainty analysis, and an independently
curated external dataset are still required before a redox transfer-learning claim. For QM9, the
local Phase 2 classical Ridge/mean/1-NN result, validation-only RF supplement, reviewed Phase 3 MIST
inference, and MIST-versus-locked-Ridge comparison are complete. Publication remains subject to the
documented dataset/provenance review and the checkpoint model card's stricter research-only,
no-redistribution, and non-commercial restrictions.

This is an independent benchmark, not an official MIST project and not affiliated with the
MIST authors. MIST is developed in the
[`BattModels/mist`](https://github.com/BattModels/mist) project; official fine-tuning tutorials
live in [`BattModels/mist-demo`](https://github.com/BattModels/mist-demo).

## License

Code and documentation are MIT licensed. Datasets and model checkpoints retain their own
licenses; the repository license never grants permission to redistribute third-party data or
weights.
