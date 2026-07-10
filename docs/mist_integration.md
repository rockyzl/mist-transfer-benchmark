# MIST integration boundary

Status: planned, not implemented. This repository neither downloads nor redistributes MIST
checkpoints.

## What was verified from official sources

The following was checked on 2026-07-10 against the official projects:

- [`BattModels/mist`](https://github.com/BattModels/mist) describes MIST as molecular foundation
  models pretrained with masked language modeling on Smirk-tokenized Enamine REAL Space SMILES and
  then fine-tuned for property prediction.
- [`BattModels/mist-demo`](https://github.com/BattModels/mist-demo) is the official tutorial project
  for property inference and fine-tuning.
- Its official
  [`run_finetuning.ipynb`](https://github.com/BattModels/mist-demo/blob/main/tutorials/run_finetuning.ipynb)
  says MIST inputs are kekulized SMILES, uses `SmirkTokenizerFast`, loads an encoder through the
  Transformers `AutoModel` mechanism, attaches a two-layer regression MLP, and demonstrates LoRA
  on attention query/value modules.
- That tutorial states that the 28M encoder is publicly released and that the listed 1.8B model
  requires access. The official
  [`molecular_property_prediction.ipynb`](https://github.com/BattModels/mist-demo/blob/main/tutorials/molecular_property_prediction.ipynb)
  also warns that the fine-tuned property checkpoints used in that notebook were not publicly
  released at the time of its text.
- The official [`mist-models` Hugging Face organization card](https://huggingface.co/mist-models)
  states research-only use, no redistribution without permission, and no commercial use without a
  licensing agreement. At least some individual MIST model cards simultaneously expose
  `apache-2.0` license metadata. Those signals are internally conflicting and must not be resolved
  by assumption.

These facts can change upstream. Pin both the repository commit and checkpoint revision when MIST
experiments are added.

## Safe v0.1 boundary

Do not guess an embedding API. The official fine-tuning tutorial demonstrates an end-to-end task
wrapper, but it does not define a stable, benchmark-neutral “export frozen embeddings” contract for
this repository. Before implementing Stage B:

1. reproduce the official 28M tutorial in a separate environment;
2. pin the exact `mist-demo`, checkpoint, Transformers, Smirk, and RDKit revisions;
3. verify with upstream code which hidden state and pooling operation represent a molecule;
4. add a small adapter test using only the synthetic fixture;
5. record preprocessing, maximum sequence length, truncation count, and failed molecules;
6. keep checkpoint files outside Git and honor their upstream license/access terms.

Before any weight download, a human must obtain clarification from the model publisher about the
Hugging Face metadata/card conflict and record the applicable terms. Never redistribute MIST
weights from this repository. Pin the exact model revision rather than relying on a moving `main`.

The official examples use Transformers with `trust_remote_code=True`. That setting executes Python
from the model repository. Pin the revision and review the remote code at that revision before
running it; do not enable remote code implicitly in CI or on machines containing secrets.

Only then add frozen-linear and frozen-MLP runners. LoRA follows the official tutorial as a separate
stage. Full fine-tuning remains future work and must not be presented as implemented.

## Comparison contract

MIST methods must consume the same `record_id` split artifact as ECFP methods. No MIST-specific
cleaning, row deletion, or split regeneration is allowed after test results are visible. If MIST
cannot tokenize a row, record the failure and apply one pre-declared policy to every method.
