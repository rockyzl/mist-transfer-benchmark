# QM9 extended comparison v1

Status: preliminary point estimates on one candidate reconstructed split

Author: Lu Zhang (张鲁), independent research

Run date: 2026-07-12

## The question

Can an already fine-tuned molecular foundation model match carefully optimized,
task-specific models on the same 12 QM9 molecular properties?

This independent benchmark compares the released MIST QM9 checkpoint with
Ridge, XGBoost, an MLP, and a validation-selected ensemble. MIST was developed
by the University of Michigan Electrochemical Energy Group; this benchmark is
not an official MIST result and is not affiliated with the model authors.

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
MAE. The best MLP was `wide_low_dropout` at 0.101237. The ensemble used
nonnegative validation-selected weights that sum to one: 39.32% MIST, 32.72%
MLP, 27.96% XGBoost, and 0% Ridge.

The full run took 2,183 seconds (36.4 minutes) on an NVIDIA RTX PRO 2000
Blackwell Generation Laptop GPU. Candidate-level runtimes and full parameters
are retained in `results/qm9-extended-comparison-v1/aggregate_metrics.json`.

## Common frozen-test result

Mean normalized MAE is the unweighted mean across all 12 targets after each
target MAE is divided by its training-set population standard deviation. Lower
is better.

| Rank | Model | Full test | Duplicate-clean test |
|---:|---|---:|---:|
| 1 | Ensemble | **0.081159** | **0.081204** |
| 2 | XGBoost | 0.094336 | 0.094365 |
| 3 | Fine-tuned MIST | 0.095064 | 0.095104 |
| 4 | MLP | 0.100364 | 0.100432 |
| 5 | Locked binary-ECFP Ridge | 0.370049 | 0.370048 |

XGBoost and MIST are separated by only 0.000728 in this single test point
estimate. Without repeated seeds or uncertainty intervals, this should be read
as near parity, not a decisive XGBoost win. The ensemble improves over both,
supporting the narrower observation that their errors contain complementary
information.

## What this result does and does not show

The result shows that a released fine-tuned molecular foundation model can
match a heavily optimized task-specific model on these 12 QM9 targets, while a
simple validation-selected ensemble performs better than either alone.

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
