# MIST Transfer Benchmark

[**Live benchmark explorer**](https://rockyzl.github.io/mist-transfer-benchmark/)
· [**Official MIST notebooks**](https://github.com/BattModels/mist-demo)

> **Demo status:** The live explorer uses tiny, synthetic fixture values to demonstrate the
> software and split logic. It does not execute MIST and contains no scientific benchmark result.

> Does large-scale SMILES pretraining learn chemistry that transfers to new molecular
> families, or does it mainly provide an expensive way to encode molecular similarity?

This repository turns that question into a falsifiable benchmark. The first target task is
redox-potential regression because it is useful for organic molecules and battery
electrolytes, while also being unusually easy to evaluate incorrectly.

## The question in plain language

Suppose a model sees millions of unlabeled SMILES during pretraining. Later, we give it a
small table of molecules and measured reduction potentials. A good score can mean two very
different things:

1. The model learned reusable chemical structure patterns.
2. The test molecules look almost the same as molecules in the training split.

Random train/test splits often cannot distinguish these explanations. This project compares
the same methods on progressively harder splits and records how similar every held-out molecule
is to its nearest training molecule.

## Pre-registered hypotheses

- **Transfer hypothesis:** a pretrained molecular encoder should retain an advantage over ECFP
  baselines on scaffold, chemical-family, and genuinely external test sets.
- **Similarity hypothesis:** if the advantage disappears as train/test similarity falls, the
  model may primarily be a costly similarity representation for this task.
- **Data-quality hypothesis:** inconsistent solvent, reference electrode, pH, temperature,
  protonation, or charge can dominate architecture choice. A large model cannot repair an
  undefined target.

These are hypotheses, not results. This repository contains no scientific benchmark result yet.

## What works now

- a strict redox-couple CSV validator that preserves measurement conditions, original values,
  conversion provenance, and molecular identity;
- deterministic molecule-grouped random, Bemis–Murcko scaffold, user-defined group, and
  external-set splits;
- RDKit ECFP fingerprints with mean, Tanimoto 1-NN, ridge, and random-forest regressors;
- a default comparability gate that rejects mixed target definitions, source types, conditions,
  and computation/measurement protocols before molecular-only fitting;
- leakage checks and nearest-training-molecule Tanimoto diagnostics;
- machine-readable run metadata, split assignments, predictions, and metrics;
- a static, accessible benchmark explorer for comparing split behavior on synthetic fixtures;
- a synthetic fixture for testing the software only.

MIST execution is intentionally not implemented here yet. No MIST weights are downloaded or
redistributed, and no private or real redox dataset is bundled. See
[`docs/mist_integration.md`](docs/mist_integration.md) for the verified integration boundary.

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

The [live benchmark explorer](https://rockyzl.github.io/mist-transfer-benchmark/) visualizes the
same software-only outputs without requiring a local install. Its committed data are reproducible:

```bash
uv run python scripts/build_demo_data.py
git diff --exit-code -- site/demo-data.json
```

## Experiment ladder

| Stage | Representation and model | Status |
|---|---|---|
| A | Mean; ECFP Tanimoto 1-NN; ECFP + ridge/random forest | Implemented |
| B | Frozen MIST encoder + linear or small MLP head | Planned |
| C | MIST LoRA + the same task head | Planned |
| D | Full MIST fine-tuning | Future only; requires a justified data/compute budget |

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

The emitted machine-readable claim gate remains closed in v0.1.1. MIST comparisons, matched
random-initialization controls, learning curves, repeated-seed aggregation, bootstrap intervals,
uncertainty analysis, and an independently curated external dataset are still required before a
transfer-learning claim.

This is an independent benchmark, not an official MIST project and not affiliated with the
MIST authors. MIST is developed in the
[`BattModels/mist`](https://github.com/BattModels/mist) project; official fine-tuning tutorials
live in [`BattModels/mist-demo`](https://github.com/BattModels/mist-demo).

## License

Code and documentation are MIT licensed. Datasets and model checkpoints retain their own
licenses; the repository license never grants permission to redistribute third-party data or
weights.
