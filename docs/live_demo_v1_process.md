# Live demo v1: model-bundle process

## What this is

This is a local, interactive prediction bundle.  Given a SMILES string, the
future demo service will run four already-built predictors:

```text
ECFP4 + Ridge
ECFP4 + XGBoost (one regressor per property)
ECFP4 + compact MLP (12 outputs)
released fine-tuned MIST-28M (served separately)
```

The first three models are trained only to make the interactive comparison
real.  They are not a new published benchmark.

## Strict data rule

The builder opens the reconstructed QM9 CSV only for the 107,108 rows marked
`train` in the existing split manifest.  It never requests target values for
the existing `validation` or `test` rows.  It makes a new deterministic 90/10
internal split *inside those train rows* solely to choose XGBoost and MLP
settings.  After choosing, each serving model is retrained on all 107,108
train rows.

Therefore a number displayed by the future live demo is a model prediction,
not a test score.  This work must not update the existing QM9 benchmark tables
or make a claim that XGBoost/MLP beat MIST.

## Reproducible inputs

- Exact source: DeepChem QM9 CSV SHA-256
  `3e668f8c34e4bc392a90d417a50a5eed3b64b842a817a633024bdc054c68ccb4`.
- Reconstructed split: `results/qm9-phase1-v2/split_assignments.tsv`.
- Feature contract: chirality-aware binary Morgan/ECFP4, radius 2, 2,048 bits.
- Bundle contract and selected-candidate search: `configs/live_demo_v1.toml`.
- All output files, including weights, stay below ignored
  `data/private/qm9/live-demo-v1/`.

## Build command

The isolated audited MIST runtime is extended locally with fixed serving
dependencies.  It is not the original publisher training environment.

```bash
uv pip install --python data/private/qm9/mist-phase3/runtime/bin/python \
  'xgboost==3.1.3' 'fastapi==0.116.1' 'uvicorn==0.35.0'

PYTHONPATH=src data/private/qm9/mist-phase3/runtime/bin/python \
  scripts/build_live_demo_bundle.py --overwrite
```

The builder records complete `pip freeze`, package versions, source/split
hashes, ECFP contract, internal-split seed, selected settings, model-file
SHA-256 values, GPU/CPU choice, and private reload smoke predictions in the
bundle manifest.

## Safety and reload checks

- Ridge coefficients are an `.npz`, not a pickle.
- Each XGBoost target model is JSON, not pickle/joblib.
- MLP weights are `safetensors`; architecture and target scaler are separate
  JSON/NPZ files.
- The loader validates every file hash, rejects symlinks/path traversal, checks
  target order, and rejects a manifest without the no-test declaration.
- No CI target reads, training, model downloads, or private-bundle access are
  added.

## Actual run record

Local build completed on 2026-07-11 in **565.14 seconds**.  It used the
private source and split hashes declared above and wrote 18 checked files to
`data/private/qm9/live-demo-v1/`.

| Item | Actual result |
| --- | --- |
| Runtime | Python 3.12.12; torch 2.9.0; XGBoost 3.1.3; FastAPI 0.116.1; Uvicorn 0.35.0; RDKit 2026.3.3 |
| Dependency freeze | 83 distributions; SHA-256 `8f9f1f9f17dda481d1cb2e8d7b99d2ae4b7aabb30849f1698f1126490ac87b68` |
| Frozen bundle config | `configs/live_demo_v1.toml`, SHA-256 `84789a6c2c6f0acd0f98f10c7aa45bb870ff3a1e39f159e09115ef4fba4f09b6` |
| Internal partition | 96,397 internal-train / 10,711 internal-validation, seed `20260711` |
| Ridge | Fixed `alpha=10`; compact coefficient bundle, 187,236 bytes |
| XGBoost choice | `wider`: depth 9, learning rate 0.05, 450 rounds, subsample 0.85, column sample 0.85; 12 JSON models, 107,862,624 bytes total |
| XGBoost internal evidence | normalized MAE: compact 0.328278; wider 0.309902; these are **selection-only** figures, not benchmark test results |
| MLP choice | `wide`: hidden layers `[768, 384]`, dropout 0.10, 16 epochs; safetensors file 7,494,640 bytes |
| MLP internal evidence | normalized MAE: compact 0.247603 (15 epochs); wide 0.239126 (16 epochs); selection-only, not a benchmark test result |
| Resource route | GPU was available and used for XGBoost/MLP; the manifest records the actual route and package versions |

The final serving weights were retrained only on all 107,108 reconstructed
train rows.  No official validation/test label was read by the builder.
`validate_bundle()` checks all 18 file hashes, and a separate fresh Python
process loaded Ridge, all 12 XGBoost JSON models, and the MLP safetensors file.
For three non-test query SMILES it returned finite arrays of shape `(3, 12)` in
native QM9 CSV units for every model.  Exact query strings and values remain in
the ignored private self-test artifact, rather than in this public document.

After a serving-path review, the MLP reload helper was explicitly changed to
call `eval()` and `torch.inference_mode()` before prediction. This is required
because its architecture contains Dropout. The ignored self-test was regenerated
through that reload path (SHA-256
`c93b7003e58990a0e3fdeba94228a0f28c59ead08f870d9eeec4eeca442dd958`), and
two consecutive reload predictions were byte-identical and matched that
self-test within an absolute tolerance of `1e-7`. No model was retrained and
no label was read for this correction.

This completes the local model-bundle phase only. It does **not** add a new
XGBoost/MLP test result, alter the published Ridge-versus-MIST result, change
the website, commit, push, or deploy anything. The following local API is a
separate serving wrapper around that unchanged private bundle.

## Local private API

The bundle can be served only from the local private runtime. The API loads the
fixed Ridge/XGBoost/MLP bundle and the already audited local MIST snapshot once
at startup; it never takes a model, filesystem path, device, or download URL
from a request. It exposes `GET /healthz`, `GET /v1/model-card`, and
`POST /v1/predict` with one SMILES (maximum 512 characters). RDKit validates
the SMILES before prediction. MLP and MIST inference use evaluation mode and
inference mode. Access logs are disabled in the launch command.

```bash
cd /home/lu2/dev/personal/projects/scifm/mist-transfer-benchmark
PYTHONPATH=src data/private/qm9/mist-phase3/runtime/bin/python \
  -m mist_transfer_benchmark.live_demo_service
```

The server binds only to `127.0.0.1:8765`. It is not deployed and does not
redistribute MIST weights.
