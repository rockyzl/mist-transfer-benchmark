# QM9-28M benchmark data card

Status: Phase 3 local released-MIST evaluation complete; raw data and weights not redistributed  
Review date: 2026-07-10  
Related protocol: [`qm9_28m_benchmark_process.md`](qm9_28m_benchmark_process.md)

## Identity

- Benchmark dataset ID: `deepchem-qm9-for-mist-28m-reconstruction`
- Dataset family: QM9
- Upstream-declared rows: `133,885`; locally parsed rows: `133,885`
- Upstream-declared task count: `12`; local ordered schema and finite-value validation: passed
- Intended use: construct a candidate split from pinned public MIST code, train matched classical
  baselines, and evaluate them and the released fine-tuned MIST-28M predictor on the same candidate
  reconstructed test rows
- Current state: locally downloaded to an ignored private cache, technically audited, and used for
  the locked classical evaluation plus one released-MIST candidate-test inference; source rows,
  row-level predictions, and weights are not bundled or redistributed

This card covers the exact locally hashed DeepChem-hosted object routed by pinned public MIST code.
It does not automatically cover another QM9 archive, conversion, mirror, or future object at the
same URL.

## Source and provenance

Public MIST commit
[`62ec2ed605021cb16d5e329b48e4280d27c151b7`](https://github.com/BattModels/mist/tree/62ec2ed605021cb16d5e329b48e4280d27c151b7)
routes QM9 to:

`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv`

Audited immutable source evidence is:

- [MIST MoleculeNet loader](https://github.com/BattModels/mist/blob/62ec2ed605021cb16d5e329b48e4280d27c151b7/electrolyte_fm/data_modules/molnet_dataset.py),
  SHA-256 `c55c89792c5a7f706831037129059b14a9e8ab4178c0236cfa4a0040c32ad5aa`;
- [MIST split utilities](https://github.com/BattModels/mist/blob/62ec2ed605021cb16d5e329b48e4280d27c151b7/electrolyte_fm/data_modules/utils.py),
  SHA-256 `60652b02681f3442c60ce2a283126d3ca5de7cc038258fbfb2830744a88225ff`;
- [MoleculeNet task declarations](https://github.com/BattModels/mist/blob/62ec2ed605021cb16d5e329b48e4280d27c151b7/submit/moleculenet_tasks.libsonnet),
  SHA-256 `f719bdbe68d36ad46fa103a3bf01b39a9cedda4309cb64a36ec01714eeb6306e`;
- [QM9 channel metadata](https://github.com/BattModels/mist/blob/62ec2ed605021cb16d5e329b48e4280d27c151b7/opt/package/datasets/qm9.yaml),
  SHA-256 `9f3673c447638e751b5cb98f624a2df07fc61c7f03345979b9e524fd23a62c96`;
- [QM9 project data page](https://quantum-machine.org/datasets/);
- [original QM9 data paper](https://doi.org/10.1038/sdata.2014.22).

This public commit is the basis of our candidate reconstruction. It is not claimed to be the unknown
historical commit used to train the checkpoint.

### HTTP HEAD observations

HEAD metadata was observed on 2026-07-10 before download:

| Field | Observed value |
|---|---|
| Content length | `29,856,825` bytes |
| ETag | `84d1e24e955bf96ed6b2986687119ad9-4` |
| Last-Modified | `2020-07-10` |

The multipart ETag is not an MD5 or SHA-256 guarantee. These are transport metadata observations,
not locally verified facts about parsed content.

### Upstream declarations

The pinned task configuration and channel metadata declare `133,885` rows and the 12 ordered targets
below. The pinned MIST sources route the CSV but do not declare `mol_id`; `mol_id` and `smiles` were
protocol expectations, not upstream claims. Local parsing subsequently observed both exact columns.

### Local Phase 1 verification

An atomic GET completed at `2026-07-10T21:33:19.123399+00:00` with HTTP `200`, final URL equal to the
requested URL, `Content-Type: text/csv`, `29,856,825` bytes, ETag
`"84d1e24e955bf96ed6b2986687119ad9-4"`, and
`Last-Modified: Fri, 10 Jul 2020 07:00:04 GMT`. The locally calculated body SHA-256 is:

`3e668f8c34e4bc392a90d417a50a5eed3b64b842a817a633024bdc054c68ccb4`

Strict streaming parsing verified the exact 21-column header, `133,885` rows, nonempty unique
`mol_id`, nonempty raw SMILES, and finite numeric values for all 12 target columns. The observed
header canonical-JSON SHA-256 is
`b866cddb483b1a3950bac9fe63eec5771560413846040ae9f397f9afbc93de4f`. Full retrieval and source
manifests remain in the ignored local Phase 1 artifact directory; the raw CSV remains outside Git.
The retrieval-record SHA-256 is
`d31af26bdf4c7f841987d244cb48035d94d3955df16ec7bab52a72177988db30`. A subsequent hardened
cache-only verification rechecked the byte count and SHA-256 before and after processing, used a
private immutable snapshot for parsing, and made no HTTP request. Its final source-manifest SHA-256
is `5cbfdca28a6a27289af4304ecfad71224821de013617f40f056f0cd4a921a4b9`.

## Rights and redistribution

QM9 has an openly accessible research history, but that fact alone does not establish the precise
license of every transformed or mirrored file. The DeepChem S3 object does not present an explicit
license beside the CSV, and this repository has not completed a rights chain from the original QM9
release through that processed object.

Until that review is complete:

- keep the raw CSV outside Git under a local ignored data directory;
- do not copy source rows into fixtures, documentation, or the static demo;
- do not commit a split table containing SMILES or target values;
- do not claim that this repository's MIT license applies to QM9 data;
- review rights separately before publishing row-level predictions or other derived artifacts.

The preferred reproducibility mechanism is a verified local retrieval script plus immutable
metadata, not redistribution of the third-party file.

The checkpoint has a separate rights boundary. Hugging Face metadata reports Apache-2.0, while the
model card states research use only, no redistribution without permission, and no commercial use
without a licensing agreement. Phase 3 applied the stricter model-card terms: local research
inference only, no weight redistribution, and no claim that the repository's MIT license covers the
checkpoint. This resolves the boundary used for the local run; it does not authorize publication of
third-party row-level artifacts, commercial use, or redistribution.

## Locally observed and required schema

The exact locally observed header is:

```text
mol_id,smiles,A,B,C,mu,alpha,homo,lumo,gap,r2,zpve,u0,u298,h298,g298,cv,
u0_atom,u298_atom,h298_atom,g298_atom
```

Its roles are:

- stable identity and representation: `mol_id`, `smiles`;
- rotational/geometric source fields that may be preserved but are not benchmark targets: `A`,
  `B`, `C`;
- ordered benchmark targets: `mu`, `alpha`, `homo`, `lumo`, `gap`, `r2`, `zpve`, `u0`, `u298`,
  `h298`, `g298`, `cv`.

The four atomization-energy columns are preserved in the private source but are not silently
promoted into features or labels. Phase 1 established:

- exactly `133,885` rows;
- nonempty, unique `mol_id` values;
- parseable, nonempty raw `smiles` values;
- all 12 target columns in the exact upstream-declared order;
- finite numeric values for every target;
- no implicit unit conversion, imputation, row sorting, row filtering, or duplicate collapse.

Future missing, unexpected, renamed, duplicated, or reordered columns close the execution gate until
reviewed.

## Targets, checkpoint unit strings, and display units

| Order | Name | Definition | Exact checkpoint unit string | Display unit |
|---:|---|---|---|---|
| 1 | `mu` | dipole moment | `debye` | D |
| 2 | `alpha` | isotropic polarizability | `cubic bohr` | bohr^3 |
| 3 | `homo` | highest occupied molecular orbital energy | `hartree` | hartree |
| 4 | `lumo` | lowest unoccupied molecular orbital energy | `hartree` | hartree |
| 5 | `gap` | HOMO-LUMO gap | `hartree` | hartree |
| 6 | `r2` | electronic spatial extent | `square bohr` | bohr^2 |
| 7 | `zpve` | zero-point vibrational energy | `hartree` | hartree |
| 8 | `u0` | internal energy at 0 K | `hartree` | hartree |
| 9 | `u298` | internal energy at 298.15 K | `hartree` | hartree |
| 10 | `h298` | enthalpy at 298.15 K | `hartree` | hartree |
| 11 | `g298` | free energy at 298.15 K | `hartree` | hartree |
| 12 | `cv` | heat capacity at 298.15 K | `calorie / mole / kelvin` | cal/(mol K) |

The validation contract uses the exact checkpoint strings, not the display abbreviations. Phase 3
verified at the pinned revision that the model's `predict` path returns these native value scales.
The benchmark applied no second inverse transform and did not substitute eV for hartree.

These labels are quantum-chemistry calculations, not laboratory measurements. In particular, they
are not experimental redox potentials, electrolyte measurements, or cell-performance data. QM9 is
also restricted to small organic molecules containing its supported elements and does not by itself
represent electrolyte formulations, electrode interfaces, salts, or manufacturing processes.

## Stable row identity

Source order is part of the data identity because the candidate split algorithm operates on row
positions. After validating the untouched CSV, derive:

```text
source_row_index = zero-based data-row position after the header
record_id = "qm9:" + six-digit source_row_index + ":" + mol_id
source_smiles = exact source cell, unchanged
```

`source_row_index`, `mol_id`, `record_id`, and `source_smiles` are immutable. A separate
`canonical_smiles` may be computed with a pinned RDKit version for duplicate auditing and ECFP, but
it must not replace `source_smiles`, reorder rows, or become the MIST inference input.

Primary MIST inference uses the raw CSV `smiles` values and only the preprocessing performed by the
reviewed pinned MIST/tokenizer implementation. No manual canonicalization or alternative SMILES
generation is allowed on that path.

## Candidate split reconstructed from public MIST code

The model card reports an 80/10/10 random split. Pinned public MIST commit
`62ec2ed605021cb16d5e329b48e4280d27c151b7` performs:

```python
train_other = dataset.train_test_split(test_size=0.2, seed=42)
validation_test = train_other["test"].train_test_split(test_size=0.5, seed=42)
```

Reconstruction conditions:

- project core environment: Python `3.12.12`, NumPy `2.5.1`, pandas `2.3.3`, RDKit `2026.3.3`, SciPy
  `1.18.0`, and scikit-learn `1.9.0`, plus an independently frozen Datasets `3.2.0` / NumPy `2.5.1`
  reference environment;
- exact source byte content and parsed row order;
- default `shuffle=True`;
- seed `42` in both calls;
- no canonicalization, deduplication, sorting, filtering, or index reset before either call.

Phase 1 locally verified the row count and produced:

| Partition | Rows |
|---|---:|
| Train | 107,108 |
| Validation | 13,388 |
| Test | 13,389 |

The local NumPy reconstruction and real isolated `datasets==3.2.0` execution matched in exact
emitted order for every row. The source-ordered assignment TSV SHA-256 is
`7cd0a0970a4643812ce263a40dfa27130216536b2c9096f69b556570b80ad5de`; the header-free assignment
row stream SHA-256 is `2be8065e3ab7089144c8e649977ecc6a60b63872e47db64430e43613456254cc`.

The checkpoint did not publish a row-ID split manifest, and the evidence does not prove which exact
historical random permutation, NumPy version, or wider environment was used for the released
training run. Therefore this is a **candidate split reconstructed from public MIST code**, never an
official, publisher-certified, or exact historical split. The final report must preserve this
uncertainty even if our pinned environment reproduces the derived counts.

The split manifest must contain only immutable row IDs/indices, split names, reconstruction
configuration, and hashes required for audit. Do not use the existing redox split allocator: it is a
different grouped algorithm with different default fractions.

## Duplicate and leakage audit

The complete candidate split is the primary cohort because the goal is to evaluate the released
checkpoint under the best-supported public-code reconstruction of its task. Before inference,
independently audit exact molecular identities:

1. parse every `source_smiles` with a pinned RDKit version;
2. derive canonical isomeric SMILES without changing source rows;
3. count within-split duplicate identities;
4. identify train-validation, train-test, and validation-test identity overlap;
5. write row-level overlap reason codes and a hashed summary before predictions are visible.

The secondary duplicate-clean test cohort excludes identities occurring in train or validation and
keeps only the lowest source row for each remaining test identity. The same frozen subset applies to
every model. No row may be removed because one method predicts it poorly.

This audit addresses exact identity only. It does not establish scaffold novelty, chemical-family
novelty, lack of similarity leakage, or lack of overlap with the MIST pretraining corpus.

The completed RDKit `2026.03.3` audit canonicalized all `133,885` rows with zero parse failures. It
found `133,798` unique canonical identities and `87` duplicate identities covering `174` rows.
Cross-split overlap comprises `15` train-test identities (`15` rows on each side), `15`
train-validation identities, and `3` validation-test identities. The duplicate-clean test retains
`13,370` of `13,389` rows; `18` rows are excluded for identity overlap with train/validation and one
higher-source-index within-test duplicate is excluded. Hashed artifacts contain no raw or canonical
SMILES and no target values.
The retained and excluded emitted-index streams have SHA-256
`fecbe1890c8c1eb7bbace00c7a7a390a9ab0d5a719cf311d38d0266f3e609189` and
`0a732f197d7f53b0dca43a9509571ed0c0520bf881c2fcd067f35f99ff33a32f`.

## Phase 2 local classical use

The local classical run used every authenticated row to construct the frozen ECFP4 feature matrix,
then loaded labels only for the train and validation indices until a durable selection lock had
been written. The exactly-once test contained `13,389` rows; the duplicate-clean reporting subset
contained `13,370`. The selected Ridge model was `alpha-10`, with full/duplicate-clean aggregate
normalized MAE `0.3700485581` / `0.3700476436` across all 12 targets.

All six random-forest candidates were subsequently run as a validation-only supplement. The best
validation candidate was `fraction-0.25-leaf-2` (`0.3582335181`), but it was not evaluated on test:
the completed lock forbids a second test access. No MIST model or weights were used. Raw source,
features, and row-level targets/predictions remain ignored local artifacts. On 2026-07-11, the
repository owner authorized only the aggregate report and static results view for publication; no
raw or row-level artifact is included in that scope.

The exact execution protocol is preserved as an ignored snapshot with SHA-256
`0c4f89123f1483d28fcd83970db2ca304f378ed254398f2e1c4774ece533c496`. The post-run observation in
the live TOML intentionally differs from that snapshot and did not trigger another test evaluation.

## Phase 3 local released-MIST use and data observations

The fixed MIST revision `65ceeed479609e9dcaef04e687556e2b39e25f23` received the exact raw
CSV SMILES for the frozen candidate test rows. A separate smoke used 64 train and 64 validation
rows and no test rows. After it passed, one durable reservation authorized exactly one inference on
all `13,389` test rows; the run had no drops, failures, or retries. The duplicate-clean result is a
reporting subset of `13,370` rows, not a retrained model or a second inference.

The aggregate mean normalized MAE was `0.09506432328592` on the complete cohort and
`0.09510356184623144` on the duplicate-clean cohort. The corresponding already locked Ridge values
were `0.37004855807165865` and `0.37004764362542336`. Per-target normalized MAE is:

| Target | Complete test | Duplicate-clean test |
|---|---:|---:|
| `mu` | `0.3131496840012973` | `0.3133405893259606` |
| `alpha` | `0.07518364589980743` | `0.07520754339233288` |
| `homo` | `0.18731326685646793` | `0.18744039642444454` |
| `lumo` | `0.10725373781215485` | `0.1072762701981668` |
| `gap` | `0.1289060650358369` | `0.1289336077536175` |
| `r2` | `0.12161575691054431` | `0.1217259360520844` |
| `zpve` | `0.02829935158129112` | `0.028288178625567453` |
| `u0` | `0.019864602153685986` | `0.019845344261378805` |
| `u298` | `0.025986585140578305` | `0.02598269516355258` |
| `h298` | `0.03715774632534351` | `0.03714757438471613` |
| `g298` | `0.04090269902416906` | `0.04089481960909268` |
| `cv` | `0.05513873868986337` | `0.055159786963863096` |

The full native-unit MAE, RMSE, and R² tables, exact execution commands, environment/model hashes,
runtime record, and locked-Ridge comparison are in the related
[`process document`](qm9_28m_benchmark_process.md). The Phase 3 run record has SHA-256
`5ca43007476bbf0b182f90be43beabab30434ecb8d143753d5f0764f53d908a0`; predictions and metrics
have SHA-256 `c3b7abf994f870f6066f0f890ea1c4d01ce10061b2ee0af115f920a28a5dcc6f` and
`fe84b07b329039c2540b2e7cf23da2eb92b13f1f469a5096f5bdf40e0a0da2f3`.

These observations do not certify the candidate split as the checkpoint's historical split. They
also do not establish scaffold/family novelty, absence of pretraining overlap, performance on
experimental data, or relevance to full battery cells. No uncertainty method was preregistered, so
the numbers are preliminary point estimates.

## Preprocessing and missing-data policy

- Preserve raw SMILES and original targets verbatim in the private source layer.
- Derive numerical arrays only after schema and finite-value validation.
- Use scikit-learn `1.9.0` `StandardScaler(copy=True, with_mean=True, with_std=True)` on the
  12-column training target matrix with population variance `ddof=0`; zero variance stops the run.
  Freeze and hash its parameters/statistics, then inverse-transform predictions exactly once.
- Use training statistics to normalize the validation selection metric.
- Do not impute a missing target; a missing required value stops the run.
- Do not drop a MIST tokenization failure or a baseline failure; a row failure stops the matched-row
  comparison until the cause and a preregistered uniform policy are reviewed.
- Verify the released model's target order, exact checkpoint unit strings, and inverse transform
  before scoring.
- Pin the tokenizer repository and revision explicitly. The reviewed prediction path performs an
  internal tokenizer lookup that does not itself guarantee the same revision as the model unless the
  caller enforces it.

## Quality and acceptance checklist

Before Phase 1 can close, record all of the following:

- [x] final retrieval URL, timestamp, HTTP metadata, byte count, and SHA-256;
- [x] public MIST commit and the four audited source-file SHA-256 values;
- [x] exact project reconstruction versions and hashed core/reference environment records, explicitly not
  presented as historical training versions;
- [x] exact header, row count, stable-ID uniqueness, and target finite-value checks;
- [x] source-order and immutable row-identity fingerprints;
- [x] candidate split algorithm, derived counts, assignment SHA-256, and exact Datasets comparison;
- [x] duplicate/identity-overlap audit and secondary-cohort SHA-256;
- [ ] applicable dataset publication rights and permitted derived-artifact handling;
- [x] stricter checkpoint model-card restrictions applied to the local research-only run, with no
  weight redistribution;
- [x] one-shot released-MIST inference, failure log, artifact hashes, and full/clean metrics;
- [x] documented limitations and unresolved historical-randomization uncertainty.

The unchecked dataset-rights item blocks publication or redistribution of row-level derived
artifacts. The explicitly authorized local research run has completed, but it does not resolve that
legal/provenance gate for broader use.
