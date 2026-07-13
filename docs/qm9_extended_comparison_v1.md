# QM9 extended comparison v1

Status: preliminary point estimates on one candidate reconstructed split

Author: Lu Zhang (张鲁), independent research

Run date: 2026-07-12

## The question

Can an already fine-tuned molecular foundation model match carefully optimized,
task-specific models on the same 12 QM9 molecular properties?

This independent benchmark first compares the released MIST QM9 checkpoint
with a task-specific pipeline built from Ridge, XGBoost, and an MLP. It then
asks a second systems question: what happens when both model families are
combined? MIST was developed by the University of Michigan Electrochemical
Energy Group; this benchmark is not an official MIST result and is not
affiliated with the model authors.

## Data and split

The exact locally hashed DeepChem QM9 CSV contains 133,885 small organic
molecules and 12 DFT-computed targets. The candidate split reconstructed from
pinned public MIST code contains:

- 107,108 training rows;
- 13,388 validation rows;
- 13,389 test rows;
- 13,370 rows in the exact-duplicate-clean test sensitivity cohort.

Targets are dipole moment, polarizability, HOMO, LUMO, HOMO-LUMO gap,
electronic spatial extent, zero-point vibrational energy, internal energies at
0 K and 298 K, enthalpy, free energy, and heat capacity.

## Training and selection

All feature and hyperparameter choices used training and validation labels
only. The test labels were loaded once after the selection manifest was
written.

Four classical feature representations were screened with Ridge:

| Representation | Validation mean normalized MAE |
|---|---:|
| Binary ECFP4 | 0.369690 |
| Count ECFP4 | 0.233293 |
| Binary ECFP4 + 17 global descriptors | 0.156159 |
| Count ECFP4 + 17 global descriptors | **0.151303** |

The selected 2,065-dimensional representation combines 2,048 count-Morgan
features with molecular weight, atom/bond/ring counts, rotatable bonds,
hydrogen-bond donors/acceptors, TPSA, LogP, fraction sp3, formal charge,
C/N/O/F counts, and aromatic-atom count.

Five XGBoost and five MLP configurations were evaluated. Early stopping was
used for XGBoost; MLP selection used validation loss with patience. The best
XGBoost configuration was `deep_slow` at 0.094030 validation mean normalized
MAE. The best MLP was `wide_low_dropout` at 0.101237. This was an iterative
systems-engineering process rather than a one-shot fit: the original
binary-ECFP Ridge began at 0.369690, count fingerprints reduced the validation
score to 0.233293, 17 global descriptors reduced it to 0.151303, and tuned
nonlinear models reduced it further.

The full run took 2,183 seconds (36.4 minutes) on an NVIDIA RTX PRO 2000
Blackwell Generation Laptop GPU. Candidate-level runtimes and full parameters
are retained in `results/qm9-extended-comparison-v1/aggregate_metrics.json`.

## Corrected two-layer interpretation

The first report used an all-model blend as the leading result. That is a useful
engineering result, but it mixes the fine-tuned model into the comparator and
therefore obscures the cleaner scientific question. The corrected primary
comparison uses a traditional-only blend of engineered Ridge, XGBoost, and MLP.
Its nonnegative weights were selected using validation labels only: 59.02%
XGBoost, 40.98% MLP, and approximately 0% Ridge.

This reporting correction was made after the first test report. It is therefore
explicitly post-specified rather than a pristine preregistered primary analysis.
No test label was used to select the corrected blend weights. Reproducible
details are stored in
`results/qm9-traditional-ensemble-correction-v1/aggregate_metrics.json`.

## Layer 1: common-test model comparison

Mean normalized MAE is the unweighted mean across all 12 targets after each
target MAE is divided by its training-set population standard deviation. Lower
is better.

| Rank | Model | Full test | Duplicate-clean test |
|---:|---|---:|---:|
| 1 | Traditional-only ensemble | **0.087400** | **0.087445** |
| 2 | XGBoost | 0.094336 | 0.094365 |
| 3 | Fine-tuned MIST | 0.095064 | 0.095104 |
| 4 | MLP | 0.100364 | 0.100432 |
| 5 | Engineered Ridge | 0.149839 | 0.149845 |
| 6 | Locked binary-ECFP Ridge | 0.370049 | 0.370048 |

XGBoost and MIST are separated by only 0.000728 in this single test point
estimate. Without repeated seeds or uncertainty intervals, this should be read
as near parity, not a decisive XGBoost win. The traditional-only ensemble
improves on either traditional constituent, demonstrating the benefit of the
modeling pipeline without incorporating MIST into the comparator.

## Layer 2: all-model ensemble

The original validation-selected all-model ensemble combines 39.32% MIST,
32.72% MLP, 27.96% XGBoost, and 0% Ridge. It scores **0.081159** on the full
test and **0.081204** on the duplicate-clean test, a 7.1% reduction from the
traditional-only ensemble. This second-layer result supports error
complementarity across the two model families; it is not the primary evidence
for comparing fine-tuning against the traditional route.

## What this result does and does not show

The result shows that a released fine-tuned molecular foundation model can
match a heavily optimized task-specific single model on these 12 QM9 targets.
The traditional route needed feature engineering, hyperparameter search, and
an ensemble to reach its best result. Adding MIST in a second-layer ensemble
improved the result again.

It does not isolate the causal value of pretraining: the raw MIST encoder,
head-only training, LoRA, and full fine-tuning were not compared. It is one
candidate random split, QM9 labels are computed rather than experimental, and
the MLP run emitted a CUDA/cuBLAS bitwise-determinism warning. A paper-grade
study should add multiple seeds, confidence intervals, scaffold or
similarity-aware splits, controlled MIST fine-tuning ablations, and additional
datasets.

## Primary resources

- MIST paper: <https://arxiv.org/abs/2510.18900>
- Official MIST repository: <https://github.com/BattModels/mist>
- Official fine-tuning demos: <https://github.com/BattModels/mist-demo>
- Released QM9 checkpoint used here:
  <https://huggingface.co/mist-models/mist-26.9M-kkgx0omx-qm9>
- Original QM9 data paper: <https://doi.org/10.1038/sdata.2014.22>
