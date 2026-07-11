# MIST integration boundary

Status: released-QM9 adapter implemented and run once from an ignored fixed-revision local snapshot;
this repository does not redistribute MIST checkpoints. Redox downstream training remains planned.

## What was verified from official sources

The following was checked on 2026-07-10 against the official projects:

- [`BattModels/mist` at audited commit `62ec2ed6`](https://github.com/BattModels/mist/tree/62ec2ed605021cb16d5e329b48e4280d27c151b7)
  describes MIST as molecular foundation
  models pretrained with masked language modeling on Smirk-tokenized Enamine REAL Space SMILES and
  then fine-tuned for property prediction.
- [`BattModels/mist-demo`](https://github.com/BattModels/mist-demo) is the official tutorial project
  for property inference and fine-tuning.
- Its official
  [`run_finetuning.ipynb` at `c14adf7`](https://github.com/BattModels/mist-demo/blob/c14adf73d69e0a29e0174917714eee830f12e0dd/tutorials/run_finetuning.ipynb)
  says MIST inputs are kekulized SMILES, uses `SmirkTokenizerFast`, loads an encoder through the
  Transformers `AutoModel` mechanism, attaches a two-layer regression MLP, and demonstrates LoRA
  on attention query/value modules. The audited notebook SHA-256 is
  `62a1cca551de43b1e7d358ac69791c3bf79f67bb658c70ace76d9a61fb94cb9e`.
- That tutorial states that the 28M encoder is publicly released and that the listed 1.8B model
  requires access. The official
  [`molecular_property_prediction.ipynb` at `c14adf7`](https://github.com/BattModels/mist-demo/blob/c14adf73d69e0a29e0174917714eee830f12e0dd/tutorials/molecular_property_prediction.ipynb)
  also warns that the fine-tuned property checkpoints used in that notebook were not publicly
  released at the time of its text. That notebook statement is historical: the official Hugging
  Face organization now lists public fine-tuned property repositories, so current availability must
  be checked checkpoint by checkpoint rather than inferred from the older notebook prose. The
  audited notebook SHA-256 is
  `db95bd41bbc0819148c658b7d1b4f505e7f20fd0efff560367c6dffdcfeca04c`.
- The selected QM9 repository,
  [`mist-models/mist-26.9M-kkgx0omx-qm9` at the pinned revision](https://huggingface.co/mist-models/mist-26.9M-kkgx0omx-qm9/tree/65ceeed479609e9dcaef04e687556e2b39e25f23),
  is an already fine-tuned 12-output MIST-28M predictor. The benchmark pins repository revision
  `65ceeed479609e9dcaef04e687556e2b39e25f23`. The expected safetensors payload is `108,614,208`
  bytes with SHA-256
  `f92e42f932c75e39a1dcb070fca8fd1c3fb3a4dcb763fb15447f035d770a9618`.
- That checkpoint's declared output order is `mu`, `alpha`, `homo`, `lumo`, `gap`, `r2`, `zpve`,
  `u0`, `u298`, `h298`, `g298`, `cv`. Phase 3 verified that its reviewed `predict` path returns the
  value scales named by exact unit strings `debye`, `cubic bohr`, `hartree`, `square bohr`, and
  `calorie / mole / kelvin`; the benchmark applied no second inverse transform.
- The checkpoint repository declares `datasets==3.2.0`, `transformers==4.57.1`,
  `torch==2.9.0`, `scikit-learn==1.7.2`, and `smirk==0.2.0`. These versions are recorded in the
  protocol. Phase 3 installed them in a separately frozen 66-distribution inference environment;
  they are not claimed to reproduce the unknown historical training environment.
- The official [`mist-models` Hugging Face organization card](https://huggingface.co/mist-models)
  states research-only use, no redistribution without permission, and no commercial use without a
  licensing agreement. At least some individual MIST model cards simultaneously expose
  `apache-2.0` license metadata. Those signals are internally conflicting and must not be resolved
  by assumption.

Moving upstream facts can change. This protocol therefore pins the reconstruction source commit,
checkpoint revision, tokenizer revision, and audited small-file hashes. Phase 3 verified the local
weight SHA-256 and the exact ten-file allowlist before inference.

The candidate split reconstruction is based on public MIST commit
`62ec2ed605021cb16d5e329b48e4280d27c151b7` and these audited files:

| File | SHA-256 |
|---|---|
| [`electrolyte_fm/data_modules/molnet_dataset.py`](https://github.com/BattModels/mist/blob/62ec2ed605021cb16d5e329b48e4280d27c151b7/electrolyte_fm/data_modules/molnet_dataset.py) | `c55c89792c5a7f706831037129059b14a9e8ab4178c0236cfa4a0040c32ad5aa` |
| [`electrolyte_fm/data_modules/utils.py`](https://github.com/BattModels/mist/blob/62ec2ed605021cb16d5e329b48e4280d27c151b7/electrolyte_fm/data_modules/utils.py) | `60652b02681f3442c60ce2a283126d3ca5de7cc038258fbfb2830744a88225ff` |
| [`submit/moleculenet_tasks.libsonnet`](https://github.com/BattModels/mist/blob/62ec2ed605021cb16d5e329b48e4280d27c151b7/submit/moleculenet_tasks.libsonnet) | `f719bdbe68d36ad46fa103a3bf01b39a9cedda4309cb64a36ec01714eeb6306e` |
| [`opt/package/datasets/qm9.yaml`](https://github.com/BattModels/mist/blob/62ec2ed605021cb16d5e329b48e4280d27c151b7/opt/package/datasets/qm9.yaml) | `9f3673c447638e751b5cb98f624a2df07fc61c7f03345979b9e524fd23a62c96` |

This is our audited reconstruction source, not evidence that the checkpoint was trained from that
exact commit.

## Released QM9 predictor boundary

The QM9-28M track is an inference comparison, not a new fine-tuning experiment. It may load only the
already fine-tuned task checkpoint pinned above. It must not:

- score a raw foundation checkpoint as if it predicted QM9 properties;
- attach or train a new head;
- update MIST, LoRA, or adapter weights;
- regenerate a favorable split after seeing predictions;
- use manually canonicalized SMILES in place of the raw CSV `smiles` input.

The exact process and stop gates are preregistered in
[`qm9_28m_benchmark_process.md`](qm9_28m_benchmark_process.md). The model must consume the raw source
SMILES rows through only the reviewed official preprocessing. The tokenizer must be loaded with the
same explicit revision as the model: the prediction implementation performs an internal tokenizer
lookup, and that lookup does not by itself guarantee revision parity.

Before inference, Phase 3 reviewed every remote Python file reachable under
`trust_remote_code=True`, recorded its hash, and ran the adapter in an isolated offline environment
without credentials. The non-test smoke verified output count, order, units, native scaling, and
row completeness before the durable one-shot gate opened.

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

For the completed local run, the recorded conflict policy applied the stricter model-card terms:
research use only, no redistribution without permission, and no commercial use without a licensing
agreement. Public or ungated access is not itself permission for another intended use. Never
redistribute MIST weights from this repository. Any future download or broader use needs a renewed
rights decision. Pin the exact model revision rather than relying on a moving `main`, and verify the
local weight hash against the preregistered value.

The official examples use Transformers with `trust_remote_code=True`. That setting executes Python
from the model repository. Pin the revision and review the remote code at that revision before
running it; do not enable remote code on machines containing secrets. CI must never download MIST
weights, fetch moving remote code, or run network-dependent checkpoint inference. Unit tests must
use local stubs or synthetic fixtures.

The released-QM9 inference adapter now enforces those guards. For the separate redox track,
frozen-linear and frozen-MLP runners remain future work; LoRA follows the official tutorial as a
later, separately reviewed stage. Full fine-tuning remains future work and must not be presented as
implemented.

## Comparison contract

MIST methods must consume the same `record_id` split artifact as ECFP methods. No MIST-specific
cleaning, row deletion, or split regeneration is allowed after test results are visible. If MIST
cannot tokenize a row, stop the matched-row comparison, record the failure, and apply a reviewed
pre-declared policy to every method before any locked test rerun. A method-specific deletion is not
permitted.
