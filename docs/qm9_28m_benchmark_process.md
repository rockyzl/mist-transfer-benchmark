# QM9-28M benchmark process

Status: Phase 4 aggregate report and local static results view complete; preliminary local result  
Protocol date: 2026-07-10  
Protocol ID: `qm9-28m-v0.1-draft`

This is the canonical process document for the first QM9 benchmark track in this repository. It is
also the living decision and execution log. If this document and an implementation disagree, stop:
the implementation must be reviewed against this document before any test score is calculated.

## The question in plain language

The experiment compares two ways of predicting the same QM9 molecular properties:

```text
the same QM9 rows and the same 12 labels
    |-- ECFP features -> classical multi-output regressors
    `-- already fine-tuned MIST-28M QM9 checkpoint -> released task predictions

the same candidate reconstructed test rows -> both methods -> the same QM9 reference values
```

The selected MIST checkpoint is already fine-tuned to predict QM9 properties. This experiment does
**not** treat a raw foundation checkpoint as a property predictor, attach a new task head, or retrain
MIST. The classical models are trained from scratch on the training rows from the candidate split
reconstructed from pinned public MIST code.

The primary fairness constraint is matched downstream supervision: the released MIST checkpoint
was trained as a 12-target QM9 predictor, so the primary classical regressors must also receive all
12 target columns during training. HOMO, LUMO, and the HOMO-LUMO gap are highlighted because they
are especially relevant to molecular and battery-material screening, but they are not the only
labels supplied to either side.

This benchmark can test whether the released predictor is more useful than the declared classical
baselines on the candidate reconstructed test rows. It cannot, by itself, prove causal chemical
understanding, isolate the effect of pretraining, or establish performance on experimental battery
systems.

## Fixed comparison

### Released MIST predictor

- Model: [`mist-models/mist-26.9M-kkgx0omx-qm9` at the pinned revision](https://huggingface.co/mist-models/mist-26.9M-kkgx0omx-qm9/tree/65ceeed479609e9dcaef04e687556e2b39e25f23)
- Pinned repository revision: `65ceeed479609e9dcaef04e687556e2b39e25f23`
- Expected architecture: MIST-28M encoder with the released 12-output property head
- Expected safetensors payload reported by the Hugging Face API on 2026-07-10:
  `108,614,208` bytes
- Expected safetensors SHA-256:
  `f92e42f932c75e39a1dcb070fca8fd1c3fb3a4dcb763fb15447f035d770a9618`
- Allowed operation in this protocol: inference only
- Forbidden operations in this protocol: MIST retraining, MIST fine-tuning, new head training,
  adapter/LoRA training, and raw-foundation-model scoring

The repository revision pins the files to review; it is not a license grant. Phase 3 locally
verified the retrieved weight bytes and hash. Hugging Face metadata says Apache-2.0, while the model
card additionally says research use only, no redistribution without permission, and no commercial
use without a licensing agreement. The local run applied the stricter model-card restrictions and
did not redistribute weights. Publication and any broader use still require an independent rights
decision.

### Classical predictors

The preregistered classical side is:

1. training-set mean for each of the 12 targets;
2. ECFP Tanimoto 1-nearest-neighbor, copying the complete 12-target vector of the nearest training
   molecule;
3. multi-output ridge regression on ECFP;
4. multi-output random-forest regression on ECFP.

The feature contract is a binary Morgan fingerprint, radius 2 (ECFP4), 2,048 bits. Under RDKit
`2026.03.3`, instantiate `rdFingerprintGenerator.GetMorganGenerator` with
`radius=2`, `fpSize=2048`, `countSimulation=False`, `includeChirality=True`, `useBondTypes=True`,
`onlyNonzeroInvariants=False`, `includeRingMembership=True`, `countBounds=None`,
`atomInvariantsGenerator=None`, `bondInvariantsGenerator=None`, and
`includeRedundantEnvironments=False`, then call `GetFingerprint(mol)`. Parse each raw source SMILES
with `Chem.MolFromSmiles(source_smiles, sanitize=True)`. Preserve the RDKit `ExplicitBitVect` objects
for `DataStructs.BulkTanimotoSimilarity(..., returnDistance=False)` and materialize model features as
a SciPy CSR matrix with `float64` binary values. No feature scaling is applied. Hash a manifest with
the feature shape, storage, dtype, generator options, and materialized matrix before fitting. The
current dense redox implementation is not evidence that the QM9-scale calculation is safe.

Before fitting the learned multi-output baselines, use scikit-learn `1.9.0`
`StandardScaler(copy=True, with_mean=True, with_std=True)` on the 12-column training target matrix.
Its population variance convention is `ddof=0`. A zero-variance target stops the run. Apply the
frozen scaler to training targets and inverse-transform predictions exactly once before metrics are
calculated. This prevents high-magnitude targets from dominating a multi-output objective. The mean
and 1-nearest-neighbor controls are reported in display units; copying a complete nearest-neighbor
target vector is equivalent under the invertible training transform. The mean is NumPy `float64`
`mean(axis=0)`. Run 1-NN queries and training rows in ascending `source_row_index` order with one
worker. Save the scaler parameters and its fitted `mean_`, `var_`, `scale_`, `n_samples_seen_`, and
`n_features_in_` state with canonical SHA-256 values.

The validation-only candidate grids are frozen as follows:

- Ridge `alpha`: `0.01`, `0.1`, `1`, `10`, `100`; `solver="lsqr"`, `tol=0.0001`,
  `max_iter=10000`, `fit_intercept=True`, `copy_X=True`, `positive=False`, and
  `random_state=None`.
- Random forest: `n_estimators=256`, `criterion="squared_error"`, `max_depth=None`,
  `min_samples_split=2`, `min_samples_leaf` in `1`, `2`, or `4`,
  `min_weight_fraction_leaf=0.0`, `max_features` in `"sqrt"` or `0.25`,
  `max_leaf_nodes=None`, `min_impurity_decrease=0.0`, `bootstrap=True`, `oob_score=False`,
  `n_jobs=16`, `random_state=42`, `verbose=0`, `warm_start=False`, `ccp_alpha=0.0`,
  `max_samples=None`, and `monotonic_cst=None`.

The random-forest candidate order is frozen as: `sqrt` with leaf sizes `1`, `2`, `4`, followed by
`0.25` with leaf sizes `1`, `2`, `4`. The Ridge order is ascending `alpha` as listed above.

Use model seed `42`. An exact Tanimoto tie is resolved by the lowest training
`source_row_index`. Candidate order in the machine-readable config resolves an exact validation
score tie. Serialize `get_params(deep=False)` with
`json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False)`, UTF-8 encoding, and no
trailing newline. Record its SHA-256 for the scaler and every Ridge and random-forest candidate. For
fitted scaler state, convert NumPy arrays/scalars to JSON lists/numbers and use the same canonical
encoding. A missing or changed parameter/state hash closes the test gate.

For each model family, choose the candidate with the smallest arithmetic mean of the 12 validation
MAEs after each target's MAE is divided by that target's training-set standard deviation. The
standard deviations are computed from training rows only. Candidate order in
[`configs/qm9_28m.toml`](../configs/qm9_28m.toml) is authoritative. Test labels cannot be used to
add candidates, choose a candidate, alter preprocessing, or choose a reporting subset.

These controls match labels and rows, not training compute or inductive bias. The final report must
state that limitation rather than calling the comparison perfectly equal.

### Resource ceilings

The protocol ceilings are 16 CPU workers, 64 GiB peak RSS separately for the parent audit process
and the isolated Datasets reference process, 24 wall-clock hours per validation candidate, 24
wall-clock hours for the locked test evaluation, and 100 GiB of local artifact storage. These are
per-process peaks, not a process-tree aggregate. Hardware must be approved before execution and
observed CPU/GPU/RAM, runtime, peak RSS, and storage must be recorded. Crossing any ceiling stops the
phase and requires a protocol amendment before a rerun; it does not authorize silently reducing
rows, targets, trees, or candidates.

## Pinned public-code evidence

The candidate split and source routing are reconstructed from public MIST commit
[`62ec2ed605021cb16d5e329b48e4280d27c151b7`](https://github.com/BattModels/mist/tree/62ec2ed605021cb16d5e329b48e4280d27c151b7).
The small audited files were fetched to a temporary directory on 2026-07-10 and hashed:

| Role | Immutable source | SHA-256 |
|---|---|---|
| QM9 URL and data-module routing | [`electrolyte_fm/data_modules/molnet_dataset.py`](https://github.com/BattModels/mist/blob/62ec2ed605021cb16d5e329b48e4280d27c151b7/electrolyte_fm/data_modules/molnet_dataset.py) | `c55c89792c5a7f706831037129059b14a9e8ab4178c0236cfa4a0040c32ad5aa` |
| two-stage random split | [`electrolyte_fm/data_modules/utils.py`](https://github.com/BattModels/mist/blob/62ec2ed605021cb16d5e329b48e4280d27c151b7/electrolyte_fm/data_modules/utils.py) | `60652b02681f3442c60ce2a283126d3ca5de7cc038258fbfb2830744a88225ff` |
| QM9 target order and declared split type | [`submit/moleculenet_tasks.libsonnet`](https://github.com/BattModels/mist/blob/62ec2ed605021cb16d5e329b48e4280d27c151b7/submit/moleculenet_tasks.libsonnet) | `f719bdbe68d36ad46fa103a3bf01b39a9cedda4309cb64a36ec01714eeb6306e` |
| row count, channel order, and exact unit strings | [`opt/package/datasets/qm9.yaml`](https://github.com/BattModels/mist/blob/62ec2ed605021cb16d5e329b48e4280d27c151b7/opt/package/datasets/qm9.yaml) | `9f3673c447638e751b5cb98f624a2df07fc61c7f03345979b9e524fd23a62c96` |

This commit is the code audited for our reconstruction. It is not claimed to be the unknown commit
or software environment used to train the released checkpoint. Every executable run must re-verify
these four hashes before constructing the split.

## Data and target contract

The source is the DeepChem-hosted CSV referenced by pinned public MIST MoleculeNet code:

`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv`

Keep three kinds of data evidence separate:

1. **HTTP HEAD observations on 2026-07-10:** `Content-Length=29,856,825`,
   `ETag=84d1e24e955bf96ed6b2986687119ad9-4`, and `Last-Modified=2020-07-10`.
2. **Upstream declarations:** `133,885` rows and the ordered 12-target schema shown below. The pinned
   MIST sources route the CSV but do not declare `mol_id`; `mol_id` and `smiles` were protocol
   expectations and were subsequently observed by local header verification.
3. **Local Phase 1 verification:** the authorized atomic GET returned HTTP `200`, the same URL,
   `29,856,825` bytes, ETag `"84d1e24e955bf96ed6b2986687119ad9-4"`, and
   `Last-Modified: Fri, 10 Jul 2020 07:00:04 GMT`. The body SHA-256 is
   `3e668f8c34e4bc392a90d417a50a5eed3b64b842a817a633024bdc054c68ccb4`. Strict local
   parsing observed `mol_id` and `smiles`, exactly `133,885` rows, the expected ordered header, and
   finite numeric values for all 12 targets.

The multipart ETag is HTTP object metadata, not an MD5 or SHA-256 content hash. HEAD observations do
not verify the CSV body. Phase 1 separately hashed and parsed the local bytes as described above.
Future retrievals must repeat both checks; a mismatch does not retroactively make either observation
false, but it stops the new run and requires a reviewed source decision.

The locally observed header canonical-JSON SHA-256 is
`b866cddb483b1a3950bac9fe63eec5771560413846040ae9f397f9afbc93de4f`. One canonical JSON
value/object per UTF-8 line with a final LF gives source-row-index SHA-256
`4a06d0363bd28b7aa9d694cca0b4b64794d2a654d1a9cbf055f357b42b16afe0`, `mol_id` SHA-256
`4bcced7297703c1f931a8fc51384f2623c42759fd17f7f015715a025eec250b5`, raw-SMILES SHA-256
`365defbce98172d8f99f5f85d3443432043c2fdb03c51b8b69e6cb8fa215614f`, and combined row-identity
SHA-256 `1394c544f06ca49e2ecc656653c8ebe41b41ee5da10c0817f9a41cf5bbac0307`.

The target order is frozen. Validation must compare the exact checkpoint unit strings, while reports
may use the separate display units:

| Position | Column | Meaning | Exact checkpoint unit string | Display unit |
|---:|---|---|---|---|
| 1 | `mu` | dipole moment | `debye` | D |
| 2 | `alpha` | isotropic polarizability | `cubic bohr` | bohr^3 |
| 3 | `homo` | HOMO energy | `hartree` | hartree |
| 4 | `lumo` | LUMO energy | `hartree` | hartree |
| 5 | `gap` | HOMO-LUMO gap | `hartree` | hartree |
| 6 | `r2` | electronic spatial extent | `square bohr` | bohr^2 |
| 7 | `zpve` | zero-point vibrational energy | `hartree` | hartree |
| 8 | `u0` | internal energy at 0 K | `hartree` | hartree |
| 9 | `u298` | internal energy at 298.15 K | `hartree` | hartree |
| 10 | `h298` | enthalpy at 298.15 K | `hartree` | hartree |
| 11 | `g298` | free energy at 298.15 K | `hartree` | hartree |
| 12 | `cv` | heat capacity at 298.15 K | `calorie / mole / kelvin` | cal/(mol K) |

All values are QM9 quantum-chemistry calculations. They are not experimental measurements. The
benchmark must not describe them as measured solubility, measured redox potential, or measured
battery performance.

Every parsed row receives:

- `source_row_index`: zero-based data-row position in the unchanged CSV order;
- `record_id`: `qm9:{source_row_index padded to six digits}:{mol_id}`;
- `source_smiles`: the unmodified source string;
- `canonical_smiles`: a derived RDKit canonical isomeric SMILES used only for identity and duplicate
  auditing.

`mol_id` must be nonempty and unique. Source order, source SMILES, and source targets are immutable.
The MIST path receives the raw CSV `smiles` cell, without manual canonicalization, and applies only
the preprocessing verified in the pinned official implementation. The tokenizer must be explicitly
pinned to the same revision as the model because the prediction implementation's internal tokenizer
lookup does not itself guarantee that revision. The classical path constructs ECFP from the same
molecular rows. Neither path may silently delete or replace a molecule.

The reviewed `predict` path is expected to inverse-transform its 12 outputs to the value scales named
by the exact checkpoint unit strings above. The implementation must verify that behavior and must
not inverse-transform a second time.

Full provenance and rights constraints are in
[`docs/qm9_28m_data_card.md`](qm9_28m_data_card.md).

## Reconstruction environment

The candidate reconstruction and classical baseline environment selected by this project is:

| Component | Exact reconstruction version |
|---|---|
| Python | `3.12.12` |
| NumPy | `2.5.1` |
| pandas | `2.3.3` |
| RDKit distribution/runtime | `2026.3.3` / `2026.03.3` |
| SciPy | `1.18.0` |
| scikit-learn | `1.9.0` |
| Hugging Face Datasets | `3.2.0` |

The core values were verified in the current locked runtime on 2026-07-10. The current core
`uv.lock` has SHA-256
`ab5ba87bc0a8fcc511332c2b369a0ec5158312d4a488d7fbf08cedf81dae94e5`, but it does not yet contain
`datasets==3.2.0`. Phase 1 therefore ran the independent reference in an isolated Python `3.12.12`
environment containing Datasets `3.2.0`, NumPy `2.5.1`, PyArrow `25.0.0`, pandas `3.0.3`, fsspec
`2024.9.0`, and the complete recorded freeze. Its canonical-JSON freeze SHA-256 is
`51ce2416c3b9d4e3678f7d6ffce452b5d9d698f4e57ea566bd0c85ddb3174720`; the inspected
`Dataset.train_test_split` source SHA-256 is
`1408e3485102e6b041f236adc31d6a05f32b0acc77338853196efa9eb756349f`. The core canonical
runtime-version record has SHA-256
`04a2a8a50aa37262a83b1d89c23b1b47f668250e3a965599309fc0950665499e`. Its exact UTF-8 bytes, with
no trailing newline, are:

```json
{"numpy":"2.5.1","python":"3.12.12","rdkit":"2026.03.3","scikit-learn":"1.9.0","scipy":"1.18.0"}
```

These are **our reconstruction versions**. They are not known historical checkpoint-training
versions, and matching public code behavior does not prove historical environment equivalence. The
checkpoint repository separately declares `datasets==3.2.0`, `transformers==4.57.1`,
`torch==2.9.0`, `scikit-learn==1.7.2`, and `smirk==0.2.0` for its inference environment; that
declaration also does not prove what was installed during training.

Phase 3 used a third, separately named environment: the **project-chosen MIST inference runtime**.
Its direct versions were Python `3.12.12`, NumPy `2.5.1`, RDKit `2026.3.3`, Datasets `3.2.0`,
Transformers `4.57.1`, Torch `2.9.0`, scikit-learn `1.7.2`, and Smirk `0.2.0`. The complete
66-distribution freeze has canonical SHA-256
`1f668948dc94738e74e97dc0e8e14530f090415af5b7ebdc14f7354340ea1bc8`; the freeze file and
runtime JSON SHA-256 values are `28e6c054855c6ca015e2b9207385e5ef88417d4c94c2c36e875caaf0bd223d1d`
and `11df2ae29cce2a9d3a2c1d69aaf4b4afec868906b5c9d7c77ffbc6bc0020ed2a`. These are project
choices informed by the checkpoint repository's declarations; they are not claimed historical
training versions.

## Candidate split reconstructed from public MIST code

The released checkpoint does not include a signed manifest of the row IDs used for training,
validation, and test. The model card describes an 80/10/10 random split, and pinned public MIST
commit `62ec2ed605021cb16d5e329b48e4280d27c151b7` uses the following two-stage Hugging Face Datasets
procedure:

```python
train_other = dataset.train_test_split(test_size=0.2, seed=42)
validation_test = train_other["test"].train_test_split(test_size=0.5, seed=42)

train = train_other["train"]
validation = validation_test["train"]
test = validation_test["test"]
```

Our reconstruction used `datasets==3.2.0`, `numpy==2.5.1`, default `shuffle=True`, seed `42`, the
locally verified exact CSV row order, and no sorting or canonicalization before splitting. It
produced these candidate counts:

| Split | Algorithm-derived rows |
|---|---:|
| Train | 107,108 |
| Validation | 13,388 |
| Test | 13,389 |
| Total | 133,885 |

The local NumPy equivalent and an isolated real `datasets==3.2.0` Dataset were compared
element-by-element in emitted split order and matched exactly. Ordered index SHA-256 values are
`b523014c…727f0` (train), `c0692f48…19084` (validation), and `736daeb0…5a73` (test); sorted
membership SHA-256 values are `738b0590…3781`, `b27c56be…af1`, and `a124fa09…b9dd`, respectively.
The source-ordered assignment TSV file SHA-256 is `7cd0a097…d5de`; its header-free assignment-row
stream SHA-256 is `2be8065e…54cc`.

The historical randomization and numerical environment of the actual released training run remain
ambiguous. The resulting artifact must always be called the **candidate split reconstructed from
public MIST code**. It must never be called official, publisher-certified, or the publisher's exact
split unless the publisher later provides matching row IDs or another authoritative artifact.
Reconstructability and matching counts are evidence, not proof that the released checkpoint saw
precisely those rows.

The primary experiment preserves this candidate split even if the duplicate audit discovers identity
overlap. A new scaffold or family split is not valid for inference-only evaluation of this released
checkpoint because the model may already have trained on rows moved into that test set.

## Evaluation cohorts

### Primary: complete candidate reconstructed test

Phase 1 verified and froze all `13,389` candidate reconstructed test rows. Every later model must use
those row IDs and order from the one immutable local assignment artifact shared by all methods. A
method-specific subset is forbidden.

### Secondary: duplicate-clean candidate reconstructed test

Before model scores are visible, derive a second fixed cohort using canonical molecular identity:

1. exclude a test row if its `canonical_smiles` appears in train or validation;
2. among the remaining test rows, retain only the lowest `source_row_index` for each
   `canonical_smiles`.

The completed RDKit `2026.03.3` audit parsed all `133,885` rows with zero failures and found
`133,798` unique canonical identities. There are `87` duplicated identities (`174` rows): `15`
train-test overlap identities involving `15` rows on each side, `15` train-validation identities,
and `3` validation-test identities. The frozen duplicate-clean test retains `13,370` rows: `18`
test rows are excluded for an identity in train or validation and one higher-index within-test
duplicate is excluded. These are data-audit observations, not property-prediction results.
The retained and excluded test-index streams have SHA-256
`fecbe1890c8c1eb7bbace00c7a7a390a9ab0d5a719cf311d38d0266f3e609189` and
`0a732f197d7f53b0dca43a9509571ed0c0520bf881c2fcd067f35f99ff33a32f`, respectively.

This is an evaluation subset only; no model is retrained for it. Record the excluded IDs and reason
codes. If canonicalization fails or depends on an unpinned RDKit build, stop before inference.

This secondary cohort asks a narrower question about exact-identity novelty. It is not a scaffold
or chemical-family generalization test.

## Metrics and reporting

For every method, cohort, and target, report:

- number of reference/prediction pairs;
- MAE on the checkpoint-native value scale, labeled with both the exact checkpoint unit string and
  its display unit;
- RMSE on that same scale and with the same two unit fields;
- R^2 when mathematically defined;
- training-standard-deviation-normalized MAE.

Also report the unweighted mean normalized MAE across all 12 targets. HOMO, LUMO, and gap receive a
prominent table, but the complete 12-target table and row-level predictions remain mandatory. A
method cannot be declared the overall winner using only one favorable target.

Predictions must retain at least `record_id`, `source_row_index`, `mol_id`, split, cohort membership,
model ID, target name, exact checkpoint unit string, display unit, reference value, prediction,
absolute error, and run ID.
Published results require uncertainty intervals chosen before the locked run; until that method is
added to the protocol, point estimates must be labeled preliminary.

The paper and model card do not use perfectly consistent aggregate metric wording for every task.
This benchmark therefore computes MAE, RMSE, and R^2 directly from saved row-level values instead of
transcribing a headline score.

## One-shot test rule

Test labels are available in the source CSV, so procedural isolation is mandatory:

1. data preparation writes the frozen split and a feature/ID-only test view;
2. training and hyperparameter-selection code receives train and validation labels only;
3. a separate evaluator joins frozen predictions to test labels by `record_id`;
4. the final test evaluator runs once for one immutable combination of source hash, split hash,
   code revision, dependency lock, model revision, preprocessing revision, and selected baseline
   configurations.

Looking at test predictions without labels is allowed for mechanical completeness checks only.
No test-derived chemical rule, filter, feature, hyperparameter, or model choice may be introduced.

If an implementation failure invalidates the locked run, preserve the failed run and log its cause.
A rerun requires a new run ID, a documented bug fix, and review confirming that no scientific choice
was made from the failed test values. Never run repeatedly and report the most favorable score.

## Stop conditions

Stop the current phase without test scoring if any of the following occurs:

- HTTP HEAD observations are presented as locally verified CSV-body facts, or a future local source
  differs from the frozen SHA-256, exact header, or row count;
- a required ID or target is missing, duplicated where uniqueness is required, nonnumeric, nonfinite,
  or has an unresolved unit/definition;
- any of the four public-MIST source files differs from its pinned commit or SHA-256;
- the hashed core environment plus isolated Datasets-reference freeze no longer resolve to Python
  `3.12.12`, NumPy `2.5.1`, RDKit `2026.3.3`, SciPy `1.18.0`, scikit-learn `1.9.0`, and Datasets
  `3.2.0` as recorded;
- the candidate split cannot be reproduced or its counts differ from the values derived from the
  locally verified source row count;
- the split artifact, input file, code revision, or dependency environment cannot be hashed;
- the duplicate audit or canonicalization report cannot be generated before inference;
- target metadata differs from an exact checkpoint unit string, even if its display notation is
  dimensionally equivalent;
- any Morgan/ECFP4 generator option differs from the frozen feature contract;
- `StandardScaler` does not use the frozen options and `ddof=0`, or any training target has zero
  variance;
- canonical `get_params(deep=False)` SHA-256 artifacts are missing or differ for the scaler, ridge,
  or random-forest candidates, or the fitted scaler-state hash is missing;
- rights for the exact DeepChem CSV or released checkpoint are insufficient or remain unresolved for
  the proposed use or publication;
- `trust_remote_code` files at the pinned model revision have not been reviewed and recorded;
- the project-chosen MIST inference runtime lacks its complete hashed lock or isolated smoke test;
- local weight bytes and SHA-256 do not match the recorded Hugging Face API observation;
- model preprocessing, output dimension, output order, units, scaling, or inverse transform cannot
  be verified from the pinned implementation;
- MIST or a baseline fails on a row and the same-row comparison can no longer be honored;
- test labels enter feature construction, fitting, candidate generation, or candidate selection;
- a prior test evaluation for the same frozen run already exists;
- the benchmark Git revision/dirty state or runtime/peak-memory/hardware manifest is missing;
- any frozen runtime, memory, worker, wall-clock, or storage ceiling is exceeded.

A stop is a useful result. Record it; do not weaken the protocol silently.

## Required artifacts

Every executable phase must retain hashes and versions for:

- source CSV and completed data card;
- public MIST commit plus all four audited source-file SHA-256 values;
- split assignments and duplicate-audit table;
- benchmark Git revision and dirty state;
- final dependency lock and exact Python/NumPy/RDKit/SciPy/Datasets/scikit-learn versions, explicitly
  labeled as our reconstruction environment rather than historical checkpoint facts;
- project-chosen MIST inference direct versions, complete transitive lock, lock SHA-256, and isolated
  smoke-test record, separately labeled from both checkpoint declarations and historical training;
- model repository revision, reviewed remote-code files, and locally verified weight hashes;
- selected classical configurations and complete validation scores;
- complete Morgan-generator options and feature-matrix shape/storage/dtype/value manifest plus hashes;
  fitted StandardScaler state; canonical scaler, Ridge, and random-forest `get_params(deep=False)`
  JSON plus SHA-256 for every candidate;
- feature/preprocessing configuration, failure logs, resource ceilings, observed runtime, peak
  memory, storage, worker count, and hardware;
- all row-level predictions and all metric tables;
- command log, result log, and any failed-run record.

Raw QM9 data, row-level predictions, and MIST weights remain outside Git. On 2026-07-11, the
repository owner authorized publication of the aggregate-only report and static results view; that
authorization does not publish raw data, row-level predictions, or weights, and the stricter
model-card restrictions remain in effect.

The successful hardened Phase 1 cache-only verification ran from
`2026-07-10T21:58:17.205938+00:00` to `2026-07-10T21:58:25.031837+00:00` (`7.826` seconds) on a
16-CPU WSL2 host. It reused the locally authenticated CSV without making an HTTP request. The
parent audit process reported `0.281143` GiB peak RSS and the isolated Datasets reference reported
`0.227024` GiB; these are separate per-process peaks, not a process-group aggregate. Its ignored
artifacts in `results/qm9-phase1-v2/` are:

| Local artifact | SHA-256 |
|---|---|
| `.qm9-phase1-owner.json` | `897b3490dd5560fe67316b83b98d63eed4eed9b5ed71f268333c2ac9bc52c6ac` |
| `phase1_run.json` | `82a0abba1b8c7ec8cea4a680eeb9f83c08ce3c5f3c140928b27d318ff45b97a4` |
| `source_manifest.json` | `5cbfdca28a6a27289af4304ecfad71224821de013617f40f056f0cd4a921a4b9` |
| `code_provenance.json` | `1b0e1e843bfb6f7d7f7f3af1f18ef43ad942d0e7c51307d52ebb3fc56d601e44` |
| `datasets_environment.freeze.txt` | `83948102d64e44d124cbe6e6b644c2f68c3ea6fe83ab415510d9a4be9466902f` |
| `datasets_reference.json` | `3dc738ad00acc90db9ddbf4a075df9e7687b6fcd08c7788ba4080451c56dab0c` |
| `split_assignments.tsv` | `7cd0a0970a4643812ce263a40dfa27130216536b2c9096f69b556570b80ad5de` |
| `row_manifest.jsonl` | `7075ade8845168a54a6e2dcdc5a8916fe853ac1705786f8aae4a82bf8e24ef5c` |
| `duplicate_events.jsonl` | `80a579afa6acf87442ea74d756354c510a436f2e4b4a46a145d57ef2906d14ca` |
| `duplicate_summary.json` | `214db1043e9ab1a3c23d7ed169d6343895cf82cb77b6479f9250a30c1c2d8c2c` |

The run captured benchmark revision `1c414e847016bf41829b5034dc362ece66415a29` with a dirty
worktree because Phase 1 implementation was intentionally uncommitted. Its exact execution-file
manifest hash is `1cdab288879bd3737ee2fbea9e940e8570b939d71c95a437846553ffe2a23d27`
and its aggregate code-provenance hash is
`546de5002ac5fa402158a465ea873e4ecd7f9549de03000007bf7b62e78af552`.

## Local Phase 2 classical observation

Phase 2 used the authenticated Phase 1 v2 source and split without downloading model weights. The
identity/feature pass read only the header, `mol_id`, and SMILES fields; train and validation labels
were then loaded by explicit source-row index. The durable selection reservation was created and
fsynced before the test-label loader was authorized. That reservation is now `completed`, and its
selection fingerprint is
`64969e850383e563ae26135b433007ae041e3353ab987214caf7ba020b1d2600`.

The complete feature matrix has shape `133,885 × 2,048`, `2,730,913` nonzero binary float64 entries,
and canonical CSR SHA-256
`3b069c0c77e12616857e5f00d70f16a5904860aa29a311073b356b14966f9b9c`. No row was dropped.
The five Ridge validation scores, in the preregistered candidate order, were:

| Candidate | Mean normalized validation MAE |
|---|---:|
| `alpha-0.01` | `0.3708113506` |
| `alpha-0.1` | `0.3707966163` |
| `alpha-1` | `0.3706546679` |
| `alpha-10` | **`0.3696898823`** |
| `alpha-100` | `0.3712913823` |

The locked test therefore used `alpha-10`. The training-mean, Tanimoto 1-NN, and selected Ridge
12-target aggregate normalized MAEs were:

| Method | Full test (`13,389`) | Duplicate-clean (`13,370`) |
|---|---:|---:|
| training-target mean | `0.7790809935` | `0.7788916072` |
| ECFP Tanimoto 1-NN | `0.6146773891` | `0.6153394038` |
| selected ECFP Ridge | **`0.3700485581`** | **`0.3700476436`** |

Selected Ridge highlighted-target native metrics are:

| Cohort | Target | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| full | HOMO | `0.0086641014` | `0.0116617605` | `0.7262759504` |
| full | LUMO | `0.0135544387` | `0.0177217491` | `0.8571573944` |
| full | gap | `0.0155197705` | `0.0203479930` | `0.8147326598` |
| duplicate-clean | HOMO | `0.0086667410` | `0.0116658686` | `0.7261078378` |
| duplicate-clean | LUMO | `0.0135486005` | `0.0177148459` | `0.8570758368` |
| duplicate-clean | gap | `0.0155117696` | `0.0203416993` | `0.8146056503` |

The locked execution invoked `run_random_forest=False` to prioritize the mandatory mean/Ridge
result; this was an execution choice, not an end-user request. After the one-shot test completed,
all six random-forest candidates were attempted in isolated, bounded **validation-only** workers.
Their scores were `0.3849865386`, `0.4033652295`, `0.4311451315`, `0.3586980847`,
`0.3582335181`, and `0.3671745211`; `fraction-0.25-leaf-2` was best on validation. It has no test
metric or prediction because a second test evaluation would violate the frozen lock. The RF
supplement did not load test labels and left the completed selection unchanged.

The locked run took `333.403` seconds and reported `0.372120` GiB parent peak RSS. The RF supplement
took `1,026.544` seconds; its largest worker peak was `6.547077` GiB. Both stayed under the frozen
ceilings. Key ignored artifact hashes are:

| Artifact | SHA-256 |
|---|---|
| locked `phase2_run.json` | `33bd8012d292818dcc05c03a6d1dedcb0cd6b80b414d1cc2a7728942d5bdf9ab` |
| execution `protocol_config.snapshot.toml` | `0c4f89123f1483d28fcd83970db2ca304f378ed254398f2e1c4774ece533c496` |
| `selection_lock.json` | `b15d6c98bc06c7a516bc5356f9e9869c07fc25ba6e425d2d2d01ab5a92247751` |
| `validation_metrics.json` | `e33ffd3e621c605b31387d9fef10fadee6d0b8ec74105fc7cc762eabf8c22052` |
| `test_metrics.json` | `27a76672b56c3dffd34aa1c2f051d5e505933f3dc06b471b844a94ca0ea9fb6d` |
| `predictions.jsonl` | `71ede81bcebd63e42eede04b9ec463ab8a3951aefd2b1226b6be363aae6fbb9f` |
| RF supplement `phase2_run.json` | `7dad8bf62d4045ddda8a5495d3cd1afe1af65ed29bda5dd906cb4e78f00482d7` |
| RF `random_forest_attempt.json` | `a7904e43eeae12edfc9be06a28c70d3a778002d421862151f4bcf50b71780ef1` |

The machine-readable observation was appended to the live TOML after execution. Consequently the
current TOML intentionally differs from the preserved execution snapshot; the one-shot test was
not rerun merely to refresh provenance. These are local classical results on a candidate split, not
MIST results and not evidence about the unknown historical checkpoint split.

### Exact Phase 2 execution commands

All three commands below were run from:

```text
/home/lu2/dev/personal/projects/scifm/mist-transfer-benchmark
```

The authenticated feature stage wrote `results/qm9-phase2-features-v1/`:

```bash
uv run python - <<'PY'
import json
from mist_transfer_benchmark.qm9.phase2_pipeline import run_phase2_feature_stage

result = run_phase2_feature_stage(
    config_path="configs/qm9_28m.toml",
    cache_dir="data/private/qm9",
    phase1_dir="results/qm9-phase1-v2",
    output_dir="results/qm9-phase2-features-v1",
    overwrite=True,
)
print(json.dumps(result, indent=2, sort_keys=True))
PY
```

The classical-only locked execution explicitly omitted RF from that invocation, wrote
`results/qm9-phase2-classical-v1/`, and consumed the durable test reservation exactly once:

```bash
uv run python - <<'PY'
import json
from mist_transfer_benchmark.qm9.phase2_pipeline import run_phase2_classical

run = run_phase2_classical(
    config_path="configs/qm9_28m.toml",
    cache_dir="data/private/qm9",
    phase1_dir="results/qm9-phase1-v2",
    feature_dir="results/qm9-phase2-features-v1",
    output_dir="results/qm9-phase2-classical-v1",
    run_random_forest=False,
)
print("PHASE2_RESULT", json.dumps(run, sort_keys=True))
PY
```

After that reservation was completed, the bounded RF supplement wrote
`results/qm9-phase2-rf-attempt-v1/`. It was validation-only and performed no second test-label read
or test prediction:

```bash
uv run python - <<'PY'
import json
from mist_transfer_benchmark.qm9.phase2_rf_attempt import run_rf_validation_supplement

result = run_rf_validation_supplement(
    config_path="configs/qm9_28m.toml",
    cache_dir="data/private/qm9",
    phase1_dir="results/qm9-phase1-v2",
    feature_dir="results/qm9-phase2-features-v1",
    locked_run_dir="results/qm9-phase2-classical-v1",
    output_dir="results/qm9-phase2-rf-attempt-v1",
)
print("RF_SUPPLEMENT_RESULT", json.dumps(result, sort_keys=True))
PY
```

Those scientific artifacts were produced by the exact Python function calls above. The repository
now also exposes equivalent CLI entry points for future fresh output directories; these commands
were added after the recorded run and must not be misread as its historical command log:

```bash
uv run mist-transfer qm9-features \
  --phase1-dir results/qm9-phase1-v2 \
  --output-dir results/qm9-phase2-features-new

uv run mist-transfer qm9-classical \
  --phase1-dir results/qm9-phase1-v2 \
  --feature-dir results/qm9-phase2-features-new \
  --output-dir results/qm9-phase2-classical-new

uv run mist-transfer qm9-rf-supplement \
  --phase1-dir results/qm9-phase1-v2 \
  --feature-dir results/qm9-phase2-features-new \
  --locked-run-dir results/qm9-phase2-classical-new \
  --output-dir results/qm9-phase2-rf-attempt-new
```

`qm9-classical` fixes `run_random_forest=False`; the separate RF command is validation-only and
cannot produce a second test evaluation.

Post-observation verification used these commands; they do not rerun a model or access labels:

```bash
uv run ruff check .
uv run pytest
uv run mist-transfer --help >/dev/null
uv run mist-transfer qm9-features --help >/dev/null
uv run mist-transfer qm9-classical --help >/dev/null
uv run mist-transfer qm9-rf-supplement --help >/dev/null
python - <<'PY'
import tomllib
from pathlib import Path

with Path("configs/qm9_28m.toml").open("rb") as handle:
    config = tomllib.load(handle)
assert config["phase_2_observation"]["locked_selection_fingerprint"] == (
    "64969e850383e563ae26135b433007ae041e3353ab987214caf7ba020b1d2600"
)
print("TOML_OBSERVATION_OK")
PY
git diff --check
```

## Local Phase 3 released-MIST observation

Phase 3 acquired exactly the ten allowlisted files from
`mist-models/mist-26.9M-kkgx0omx-qm9` revision
`65ceeed479609e9dcaef04e687556e2b39e25f23`. The local snapshot stayed under
`data/private/qm9/mist-phase3/model/` and remains ignored. `model.safetensors` was exactly
`108,614,208` bytes with SHA-256
`f92e42f932c75e39a1dcb070fca8fd1c3fb3a4dcb763fb15447f035d770a9618`; the complete
ten-file allowlist manifest has canonical SHA-256
`76549ea643813af23d733da5811427fb34ea72172ee9b216af3378f8e548e830`.

The pre-execution audit read all `783` lines of the pinned Python implementation and inspected the
safetensors header without loading Torch or executing remote code. It found `145` contiguous,
non-overlapping F32 tensors with complete extents, including the `[12, 512]` / `[12]` output head
and 12-value transform mean/std tensors. The notebook contains a dynamic `!pip install` and was
explicitly forbidden. The model implementation contains tokenizer network fallbacks and
`save_pretrained` methods, so the runtime required an explicit local same-revision tokenizer,
offline/local-files-only loading, safetensors only, a hard error for `tokenizer=None`, and no save
call. The audit passed with `remote_code_executed=false`.

The isolated smoke used 64 train and 64 validation rows, no test rows. It completed one CUDA batch
of 128 without an operational failure or batch-size reduction. Only after that smoke passed did the
pipeline atomically create the durable test-inference reservation
`444e4dfede09a97573da2927334c8b701821f3bf0e4a71e4a4e0c1f0aefba11c`. The one authorized
test pass then emitted all `13,389 × 12` values in 105 batches of 128 (final batch partial), with no
row drop, retry, or second inference. `model.eval()` and `torch.inference_mode()` were active;
named outputs were stacked in frozen config order. The released `predict` method already returns
native target scales, so no manual inverse transform was applied.

### Phase 3 result

The aggregate is the unweighted mean across 12 targets of MAE divided by each target's frozen
training-set standard deviation. Lower is better.

| Cohort | Rows | Released MIST | Already locked ECFP Ridge | MIST − Ridge |
|---|---:|---:|---:|---:|
| complete candidate test | 13,389 | `0.09506432328592` | `0.37004855807165865` | `-0.27498423478573863` |
| duplicate-clean candidate test | 13,370 | `0.09510356184623144` | `0.37004764362542336` | `-0.27494408177919194` |

Complete released-MIST metrics follow. MAE and RMSE use each target's checkpoint-native unit;
NMAE is MAE divided by its training-target standard deviation.

| Target (native unit) | Full MAE | Full RMSE | Full R² | Full NMAE | Clean MAE | Clean RMSE | Clean R² | Clean NMAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mu` (debye) | `0.4784568885024821` | `0.7342468405950063` | `0.7758032006715276` | `0.3131496840012973` | `0.47874856999636023` | `0.7346070609538805` | `0.7756839385858486` | `0.3133405893259606` |
| `alpha` (cubic bohr) | `0.6155666764156991` | `1.0717296449808045` | `0.982752169804098` | `0.07518364589980743` | `0.615762337318712` | `1.0722408002958068` | `0.9827424879090473` | `0.07520754339233288` |
| `homo` (hartree) | `0.004143533882100081` | `0.006140502533567574` | `0.9241086030307173` | `0.18731326685646793` | `0.004146346099735106` | `0.006143869166562673` | `0.9240322325378314` | `0.18744039642444454` |
| `lumo` (hartree) | `0.005035296503056042` | `0.007339010940833366` | `0.9755026185501939` | `0.10725373781215485` | `0.005036354342594373` | `0.007340901743534177` | `0.9754568826240957` | `0.1072762701981668` |
| `gap` (hartree) | `0.006128544720396808` | `0.009433142960857647` | `0.9601829994643277` | `0.1289060650358369` | `0.006129854176065885` | `0.009435597472057928` | `0.960110293074521` | `0.1289336077536175` |
| `r2` (square bohr) | `34.05327372906133` | `52.442444569559875` | `0.9640730307497981` | `0.12161575691054431` | `34.08412466944447` | `52.47687037944346` | `0.9640593912943676` | `0.1217259360520844` |
| `zpve` (hartree) | `0.0009410450891896802` | `0.0013757117401776275` | `0.9982951015607988` | `0.02829935158129112` | `0.0009406735522275951` | `0.0013756694424523622` | `0.9982930986714368` | `0.028288178625567453` |
| `u0` (hartree) | `0.7969571243829455` | `1.0771854744333453` | `0.9992713805779946` | `0.019864602153685986` | `0.7961845081303738` | `1.0763201419873185` | `0.9992722253952468` | `0.019845344261378805` |
| `u298` (hartree) | `1.0425620835282963` | `1.3125857048916065` | `0.9989181183369714` | `0.025986585140578305` | `1.0424060206007943` | `1.312616617737062` | `0.9989175834323821` | `0.02598269516355258` |
| `h298` (hartree) | `1.4907405962617597` | `1.763574559751578` | `0.9980469558079901` | `0.03715774632534351` | `1.4903325057197976` | `1.763298328587191` | `0.9980466942594568` | `0.03714757438471613` |
| `g298` (hartree) | `1.6410151751768856` | `1.8841868718283226` | `0.997770759820947` | `0.04090269902416906` | `1.6406990532577863` | `1.8838308177266319` | `0.997770605791641` | `0.04089481960909268` |
| `cv` (calorie / mole / kelvin) | `0.22409760886330485` | `0.3258032143752294` | `0.9934973242891736` | `0.05513873868986337` | `0.2241831543071455` | `0.32594322238322876` | `0.9934896613272012` | `0.055159786963863096` |

The result is encouraging but narrow. It compares one released, already QM9-fine-tuned predictor
with an ECFP Ridge baseline on a random candidate split reconstructed from public code. The
publisher did not release certified row memberships, so this is not an official checkpoint test
reproduction. It does not isolate pretraining, prove mechanistic understanding, test new scaffolds
or chemical families, or establish accuracy on experimental battery properties. QM9 labels are
DFT-computed. No uncertainty method was preregistered, so the point estimates are preliminary.

### Exact Phase 3 environment and execution commands

Commands were run from the repository root. The runtime was created from the pinned direct
requirements below; the resulting full transitive freeze, rather than a claim about the historical
training environment, is the reproducibility record.

```bash
uv venv --python 3.12.12 data/private/qm9/mist-phase3/runtime
uv pip install --python data/private/qm9/mist-phase3/runtime/bin/python \
  "numpy==2.5.1" "pandas==2.3.3" "rdkit==2026.3.3" \
  "datasets==3.2.0" "transformers==4.57.1" "torch==2.9.0" \
  "scikit-learn==1.7.2" "smirk==0.2.0"
uv pip freeze --python data/private/qm9/mist-phase3/runtime/bin/python \
  > data/private/qm9/mist-phase3/runtime.freeze.txt
```

The project recorded the freeze inside each ignored result directory; its file hash is the
`28e6c054…23d1d` value above. Model acquisition was the only networked model step:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m mist_transfer_benchmark.cli \
  qm9-mist-acquire \
  --model-dir data/private/qm9/mist-phase3/model
```

Static audit ran before any remote-code execution:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m mist_transfer_benchmark.cli \
  qm9-mist-audit \
  --model-dir data/private/qm9/mist-phase3/model \
  --runtime-python data/private/qm9/mist-phase3/runtime/bin/python \
  --output-dir results/qm9-phase3-audit-v1
```

The single inference command performed the non-test smoke, created the durable reservation, then
performed the one authorized candidate-test inference:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m mist_transfer_benchmark.cli \
  qm9-mist-infer \
  --phase1-dir results/qm9-phase1-v2 \
  --phase2-dir results/qm9-phase2-classical-v1 \
  --audit-dir results/qm9-phase3-audit-v1 \
  --model-dir data/private/qm9/mist-phase3/model \
  --runtime-python data/private/qm9/mist-phase3/runtime/bin/python \
  --output-dir results/qm9-phase3-mist-v1 \
  --device auto \
  --initial-batch-size 128
```

The worker used an NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU with `8,546,484,224` bytes
of device memory and Torch CUDA `12.8`. Test worker time was `9.019580934982514` seconds; total
orchestrator time, including verification and artifact writing, was `23.857965409988537` seconds.
Peak worker/orchestrator RSS was `1.2622833251953125` / `0.7814979553222656` GiB. Peak CUDA
allocated/reserved memory was `205,038,592` / `346,030,080` bytes.

### Phase 3 artifact hashes and failures

| Artifact or evidence | SHA-256 |
|---|---|
| audit `phase3_audit_run.json` | `0660c6ef1c0e7184303957696a7d73aaf6d7a92ea2ead097e4bceadd2983dc1f` |
| audit/inference `model_audit.json` | `56d0dd48697e4366dbbaff3eec083d1ee170b25ed10e9196fb2efedc0e981ca1` |
| execution `protocol_config.snapshot.toml` | `14b9d76ee63a3e116e1285df27ec352ccc55496deeb78d373b598cd1530acfe4` |
| `runtime_environment.freeze.txt` | `28e6c054855c6ca015e2b9207385e5ef88417d4c94c2c36e875caaf0bd223d1d` |
| runtime freeze canonical record | `1f668948dc94738e74e97dc0e8e14530f090415af5b7ebdc14f7354340ea1bc8` |
| `runtime_environment.json` | `11df2ae29cce2a9d3a2c1d69aaf4b4afec868906b5c9d7c77ffbc6bc0020ed2a` |
| `smoke.json` | `98240614ca87f3114f59519b0aa3d6122952f2af70bab4e68c8c0c6c8252c5f0` |
| inference `worker_report.json` | `0641056e9b018f0170418711f0660aec34b4f94af2c123dda566a0377e510ec8` |
| Phase 3 code-provenance file | `979dddfe612857fa60fc3f4dc130f9f4abef83d57507dc39717e8e79044ab4f4` |
| Phase 3 code-provenance aggregate | `475c7d5a26e7cefaf0c78ba1fe86e9b841fffc7b4ddec50eb06b595a18680af7` |
| `mist_predictions.jsonl` | `c3b7abf994f870f6066f0f890ea1c4d01ce10061b2ee0af115f920a28a5dcc6f` |
| canonical prediction matrix | `9f7e4e2c051f42099fbdb65de1d9a611efd4ece7cbde1dc89dce84e3de206317` |
| `mist_metrics.json` | `fe84b07b329039c2540b2e7cf23da2eb92b13f1f469a5096f5bdf40e0a0da2f3` |
| `comparison.json` | `45534119f79ea5708dd58cee050290fe8847cc9ea933fcf26acfc9207caa229e` |
| inference `failure_log.json` | `843f02ac2ac68eb1fa085d87e0e5f0c52d076820f1c1cfbae86b2db1dbbfd32a` |
| inference `phase3_run.json` | `5ca43007476bbf0b182f90be43beabab30434ecb8d143753d5f0764f53d908a0` |

Both audit and inference failure logs contain empty event arrays. Inference records zero test
retries. The completed reservation stores the final run, predictions, metrics, and comparison
hashes, preventing a silent second use of the same inference fingerprint. The live TOML observation
was appended after execution, so it intentionally differs from the immutable execution snapshot;
the model and test were not rerun to refresh documentation.

## Phase checkpoints

| Phase | Deliverable | Entry gate | Exit gate | Status on 2026-07-10 |
|---|---|---|---|---|
| 0 | Protocol, data card, frozen config | Read-only source audit | Human protocol review | Complete |
| 1 | Local source manifest, validator, candidate split, duplicate audit | Phase 0 approved; explicit local retrieval authorization | Source facts, environment hashes, source-code hashes, counts, IDs, split and duplicate tests pass | Technical audit complete; raw and row-level artifacts remain private |
| 2 | Scalable 12-target classical baselines and validation selection | Phase 1 frozen | Memory smoke test and validation artifacts reviewed | Ridge/mean/1-NN locked test complete; RF validation-only supplement complete, with no RF test |
| 3 | Reviewed released-MIST inference adapter | Stricter local research-use rights boundary and remote-code guards recorded | Output order/units/preprocessing, smoke, reservation, and row completeness verified | Complete; one inference, zero retries |
| 4 | MIST-vs-classical comparative locked evaluation | Phase 3 frozen; classical selection immutable | Compare MIST predictions with already locked Ridge artifacts on identical rows | Complete for full and duplicate-clean cohorts |
| 5 | Scientific review and public report | Phase 4 independently reviewed | Aggregate-only scope, limitations, uncertainty, and claims approved | Owner-authorized aggregate report and static view ready for deployment; uncertainty remains documented |
| 6 | Optional 1.8B single-target extension | 28M result is trustworthy and compute justified | Separate preregistration | Out of scope |

Moving to a later phase requires recording the previous phase's exit evidence. Phase 0 approval does
not authorize model/data download, test execution, publication, or a change in scientific claims.

## Command, result, and failure log template

Append one entry per material action. Store large stdout/stderr as a hashed artifact and link it; do
not paste credentials, access tokens, or private paths.

```text
Timestamp UTC:
Operator:
Phase and checkpoint:
Purpose:
Command or notebook cell ID:
Working directory:
Expected inputs and SHA-256:
Expected output:
Exit status:
Observed counts/checks:
Output artifact paths and SHA-256:
Runtime / peak memory / hardware:
Failure category and full log path, if any:
Decision (continue / stop / invalidate / review required):
Reviewer and review date:
```

## Living process log

| Date | Phase | Entry | Evidence/status |
|---|---|---|---|
| 2026-07-10 | Discovery | Selected the public fine-tuned 28M multitask QM9 checkpoint as the first serious candidate because the checkpoint and large labeled dataset are accessible and HOMO/LUMO/gap are included. | Research decision only; no model or data downloaded. |
| 2026-07-10 | Split audit | Pinned public MIST commit `62ec2ed605021cb16d5e329b48e4280d27c151b7`, four relevant source-file hashes, and the two-stage 80/10/10 candidate reconstruction. | Historical training randomization/environment and exact row membership remain unknown. |
| 2026-07-10 | Environment audit | Recorded the exact current Python/NumPy/RDKit/SciPy/scikit-learn runtime and required Datasets version. | These are our chosen reconstruction versions, not claimed historical checkpoint versions; final lock remains a Phase 1 gate. |
| 2026-07-10 | Phase 0 | Recorded this protocol, the separate data card, and machine-readable configuration. | Documentation only; all executable phases remain closed. |
| 2026-07-10 | Phase 1 source audit | Atomically retrieved and locally validated the routed DeepChem CSV. | HTTP 200; 29,856,825 bytes; SHA-256 `3e668f8c…ccb4`; exact header, 133,885 rows, unique `mol_id`, nonempty raw SMILES, and 12 finite targets passed. Raw data remain ignored and unredistributed. |
| 2026-07-10 | Phase 1 split audit | Compared the local two-stage NumPy reconstruction with an isolated real Datasets 3.2.0 execution. | Exact ordered membership match; 107,108/13,388/13,389 rows; assignment artifacts frozen. This remains a candidate reconstruction, not a publisher-certified split. |
| 2026-07-10 | Phase 1 duplicate audit | Canonicalized every row with RDKit 2026.03.3 and emitted hashed identity artifacts. | Zero parse drops; 87 duplicated identities; 15 train-test overlap identities; duplicate-clean test has 13,370 rows. No model was run. |
| 2026-07-10 | Phase 2 features | Generated the frozen full-row ECFP4 float64 binary CSR matrix after Phase 1 authentication. | Shape 133,885 × 2,048; 2,730,913 nonzeros; no drops; canonical SHA-256 `3b069c0c…f9b9c`. |
| 2026-07-10 | Phase 2 locked classical test | Selected Ridge `alpha-10` from five validation candidates, durably reserved the selection, then evaluated mean/Ridge/1-NN once on full and duplicate-clean test cohorts. | Reservation completed; selection `64969e85…2600`; locked run `33bd8012…f9ab`; no MIST execution. |
| 2026-07-10 | Phase 2 RF supplement | Ran all six RF candidates in isolated 16-worker validation-only processes after the classical test lock completed. | `fraction-0.25-leaf-2` had the best validation score (`0.3582335181`); no RF test-label access, test prediction, or second evaluation. |
| 2026-07-10 | Phase 3 model/runtime audit | Acquired the fixed ten-file snapshot, verified every hash and safetensors structure, reviewed pinned remote code, and froze the 66-package isolated runtime. | Audit run `0660c6ef…dc1f`; remote code not executed during audit; stricter model-card restrictions applied; weights remain ignored and unredistributed. |
| 2026-07-10 | Phase 3 smoke and reservation | Ran 64 train + 64 validation rows on CUDA, then atomically reserved the immutable inference fingerprint. | Smoke passed at batch 128 with no test rows or failures; reservation `444e4dfe…a11c` created before test inference. |
| 2026-07-10 | Phase 3 one-shot inference | Ran the released checkpoint once on all 13,389 candidate test rows and compared its saved predictions with the already locked Ridge result. | Zero retries/drops; MIST aggregate NMAE `0.0950643233` full and `0.0951035618` clean versus Ridge `0.3700485581` / `0.3700476436`; run `5ca43007…08a0`. |
| 2026-07-10 | Phase 4 aggregate presentation | Generated a tracked aggregate-only JSON, focused result report, and local static QM9 results view from authenticated Phase 2/3 aggregate artifacts. | Summary `d21274ad…d76c`; no row IDs, labels, predictions, source data, or weights included; no metric or inference change; deployed Pages URL remains stale until a future push. |
| 2026-07-11 | Aggregate-only publication authorization | Recorded repository-owner authorization to deploy the static aggregate report and demo. | Regenerated summary `3079c647…f910`; raw data, row-level predictions, and weights remain private; stricter model-card restrictions remain in effect. |
| 2026-07-11 | GitHub Pages deployment | Merged PR #3 into `main` and verified the deployed aggregate JSON and static QM9 section. | Pages workflow `29160041138` succeeded; the live JSON SHA-256 is `3079c647…f910`; no model runs in the browser. |

## Change control

Any change to data identity, target order/units, split membership, duplicate policy, model revision,
baseline candidates, selection metric, evaluation cohorts, or final metrics increments the protocol
version and requires approval before test evaluation. Editorial corrections that do not change
behavior may keep the version but must be logged.
