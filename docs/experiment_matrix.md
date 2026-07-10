# Experiment matrix

The research comparison changes the molecular representation and adaptation method while holding
the rows, target, split, metrics, and selection rules fixed.

| ID | Representation | Trained component | v0.1 status |
|---|---|---|---|
| `mean-only` | None | Training-set mean | Implemented sanity check |
| `ecfp-tanimoto-1nn` | RDKit ECFP, radius 2, 2048 bits | Nearest training label | Implemented similarity control |
| `ecfp-ridge` | RDKit ECFP, radius 2, 2048 bits | Ridge regressor | Implemented |
| `ecfp-random-forest` | Same ECFP | Random forest | Implemented |
| `mist-frozen-linear` | Pinned MIST checkpoint | Linear head only | Planned |
| `mist-frozen-mlp` | Pinned MIST checkpoint | Small MLP head only | Planned |
| `mist-lora-mlp` | Same checkpoint | LoRA and MLP head | Planned |
| `mist-random-init-mlp` | MIST architecture, random initialization | Matched MLP/training budget | Planned control |
| `mist-random-init-lora-mlp` | MIST architecture, random initialization | Matched LoRA/MLP budget | Planned control |
| `mist-full-finetune` | Same checkpoint | All weights and head | Future only |

The declarative draft is in [`configs/experiment_matrix.toml`](../configs/experiment_matrix.toml).

## Required comparisons

Run every implemented method on the same five pre-declared seeds for:

1. molecule-grouped random split;
2. Bemis–Murcko scaffold split;
3. curated chemical-family group split;
4. untouched external set.

Report MAE, RMSE, R² when defined, row counts, and nearest-training ECFP Tanimoto distributions.
Aggregate metrics alone are insufficient: plot absolute error against nearest-training similarity.

## Fairness rules

- Select hyperparameters using train/validation only.
- Give methods an explicit, comparable tuning budget.
- Do not change data cleaning or split membership per model.
- Report model/checkpoint revision, trainable parameter count, runtime, and peak memory.
- Run the classical baseline before spending GPU time.
- Full fine-tuning remains out of scope until data size, compute, and stopping rules are justified.

No MIST row in this matrix is a result. Its `planned` status means the adapter and protocol still
require implementation and review.

## Claim gate

Every run artifact contains `claim_gate.ready_for_transfer_claim=false`. It must remain false until
the pretrained MIST rows, matched random-initialization controls, learning curves, repeated-seed
aggregation, bootstrap intervals, uncertainty analysis, and a genuinely external dataset are all
implemented and reviewed. v0.1.1 is infrastructure, not claim-ready evidence.
