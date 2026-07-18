# QM9 fixed-MIST-split repeated evaluation v2

Status: frozen protocol for planned v2 execution; no v2 test result is reported here

Protocol date: 2026-07-18

Author: Lu Zhang (张鲁), independent research

## Scientific question

On the same candidate QM9 split used for the completed released-MIST observation, how stable are
the locally trained traditional models across training seeds, and how do their errors compare with
the one fixed set of released-MIST predictions?

This is a repeated **training** evaluation, not a repeated split evaluation. The train,
validation, and test rows never move. The released fine-tuned MIST checkpoint is not retrained,
retuned, or run on a new random or scaffold split.

## Completed v1 evidence versus planned v2 evidence

The following are completed v1 observations, not v2 results:

- the candidate split reconstructed from pinned public MIST code contains 107,108 train, 13,388
  validation, and 13,389 test rows;
- the released `mist-models/mist-26.9M-kkgx0omx-qm9` checkpoint at revision
  `65ceeed479609e9dcaef04e687556e2b39e25f23` was used for one fixed inference observation;
- the v1 full-test mean normalized MAEs were 0.087400 for the traditional-only ensemble,
  0.094336 for XGBoost, 0.095064 for MIST, 0.100364 for MLP, and 0.149839 for engineered Ridge;
- v1 selected count-ECFP4 plus 17 global descriptors, the XGBoost `deep_slow` recipe, and the MLP
  `wide_low_dropout` recipe using validation labels only; and
- the v1 all-model ensemble was an engineering supplement, not the primary comparison.

V2 has not produced evidence until all five seed artifacts pass the gate and `summary.json` is
written. V2 will add repeated-seed traditional results, paired-row uncertainty, fixed-test
structural-novelty strata, runtime accounting, and MLP learning-curve monitoring. No v1 point
estimate may be relabeled as a v2 repeated result.

## Critical-step Plan -> Execute -> Review gates

Every critical step has a written plan before execution and a recorded review afterward. A later
step cannot be treated as scientifically authorized merely because its code ran successfully.

| Checkpoint | Plan | Required review |
|---|---|---|
| Input boundary | Freeze identities, membership, row order, dependencies, and the no-test-label boundary. | Automated provenance/leakage checks pass before fitting; lead reviews the private preflight. |
| Selection freeze | Train all five seeds and make validation-only decisions. | Review seed completeness, scalers, selected rounds/epochs, loss anomalies, ensemble weights, and prediction hashes. |
| Test unlock | Permit one test-label read from the immutable global-freeze hash. | Recompute and verify the gate, artifact trail, and zero prior test reads before authorization. |
| Publication | Produce only preregistered metrics, intervals, runtime, and novelty strata. | Independent scientific review checks claims against artifacts before the article, site, or headline is updated. |

The preflight and run manifest expose these checkpoints as `critical_reviews`. Automated reviews
are evidence checks, not a substitute for the independent publication review. A completed run has
`publication_ready=false` until that final review is performed outside the training runner.

## Frozen inputs and identities

The runner must reject the run before fitting if any frozen identity differs.

| Input | Frozen contract |
|---|---|
| QM9 source | 133,885-row authenticated CSV; SHA-256 `3e668f8c34e4bc392a90d417a50a5eed3b64b842a817a633024bdc054c68ccb4` |
| Split source | `results/qm9-phase1-v2/datasets_reference.json`; SHA-256 `3dc738ad00acc90db9ddbf4a075df9e7687b6fcd08c7788ba4080451c56dab0c` |
| Ordered train membership | 107,108 rows; membership SHA-256 `738b05901e5f279ed84e87690250d9f7064b3f5edc3582fbf8e2b0ce07713781` |
| Ordered validation membership | 13,388 rows; membership SHA-256 `b27c56bef1233178f3e3844760182b92a06163daacc131131ac88641e757baf1` |
| Ordered test membership | 13,389 rows; membership SHA-256 `a124fa0953a4c343e6ae3b83f7d7ac7ea963714c17605338aba07c5ca237b9dd` |
| Feature artifact | 133,885 × 2,065 sparse matrix; file SHA-256 `ddaaff5608faa3428ee2720fca173e24dc4db90399852c1363db55268ea33810` |
| Feature schema | 2,048 count-ECFP4 values plus the 17 v1 global descriptors; no v2 feature search |
| Scaffold groups | 133,885 row-aligned groups; file SHA-256 `037e27550db7b8dcc5df33d898258cb9c9033268e7daacce24dbd10b4f9f4151` |
| Target order | `mu`, `alpha`, `homo`, `lumo`, `gap`, `r2`, `zpve`, `u0`, `u298`, `h298`, `g298`, `cv` |
| Fixed MIST test predictions | 13,389 rows in the frozen test order; source artifact SHA-256 `c3b7abf994f870f6066f0f890ea1c4d01ce10061b2ee0af115f920a28a5dcc6f` |
| Fixed MIST validation predictions | Required only for the supplemental all-model ensemble; exact row-order and file hash must be frozen in `manifest.json` |

The prediction arrays must have shape `[rows, 12]`, native checkpoint target units, a numeric
non-object dtype, and finite values. Every imported MIST array must carry the exact source-row
index array. Row count, order, indices, target order, checkpoint ID, checkpoint revision, and file
SHA-256 are verified before use. The existing MIST predictions are inference-only inputs: v2 must
not alter MIST weights, preprocessing, target scaling, loss, or predictions.

The five predetermined traditional training seeds are:

```text
20260713, 20260729, 20260811, 20260823, 20260907
```

No seed may be replaced because its result is poor or its runtime is inconvenient. A failed seed is
reported as failed and the primary five-seed result remains incomplete until it is resolved under
the same protocol.

## Models, fitting, and validation-only decisions

Each seed trains engineered Ridge, XGBoost, and MLP on exactly the frozen 107,108 training rows.
The validation rows remain separate. The feature representation and model recipe families are
inherited from v1 and frozen before v2; v2 is not another broad hyperparameter or feature search.

- **Engineered Ridge:** fit the frozen v1 recipe with `alpha=10`. Its optimization is effectively
  deterministic, but it is executed and recorded in every seed cell so the artifact contract is
  uniform.
- **XGBoost:** use the frozen v1 `deep_slow` recipe: `max_depth=10`, `learning_rate=0.03`,
  `n_estimators=1100`, `subsample=0.85`, `colsample_bytree=0.90`, and
  `min_child_weight=1.0`. The seed controls the stochastic training route. Validation labels may
  select the per-target early-stopping iteration with `early_stopping_rounds=35`; they may not
  select a different hyperparameter family or exceed 1,100 estimators.
- **MLP:** use the v1 `wide_low_dropout` winner: hidden dimensions `[768, 384]`, dropout `0.02`,
  learning rate `0.0007`, weight decay `0.00001`, and batch size `512`. Fit input scaling and 12
  independent target scalers on training rows only. Train on standardized targets, invert the
  target transform before scoring, and use validation-only early stopping with an 80-epoch cap,
  patience 10, and minimum improvement `0.0001`. These monitoring-budget values are the corrected
  v2 contract established before test access; they replace v1's shorter exploratory epoch budget
  without reopening architecture or optimizer search. A seed must restore its best checkpoint
  rather than use the final epoch merely because it ran last.

For MLP monitoring, `training_loss` is the optimizer loss in standardized-target space and
`validation_error` must name its exact definition in the seed artifact. The preferred outer curve
is validation mean normalized MAE in native units; if the implementation records sklearn's inner
early-stopping curve, it must be labeled `1 - validation_R2` and must not be presented as MAE.
Nonfinite loss, an empty or misaligned curve, nonfinite predictions, or failure to restore the best
checkpoint is an abnormal stop condition before the test gate. A sustained validation-error rise
after the best epoch (at least two consecutive increases) is marked as a possible-overfitting
warning. A warning is retained in the report; it is not permission to inspect test performance and
revise training.

For each seed, validation predictions select nonnegative weights summing to one for:

1. the **traditional-only ensemble**, using engineered Ridge, XGBoost, and MLP; and
2. the **supplemental all-model ensemble**, using those three models plus fixed MIST validation
   predictions.

The optimization objective is the unweighted mean of the 12 validation target MAEs after division
by the frozen training-target population standard deviations. Ensemble weights are seed-specific
because the traditional validation predictions are seed-specific. The traditional-only ensemble
is primary. The all-model ensemble must be labeled supplemental and is omitted, with a machine-
readable reason, if authenticated MIST validation predictions are unavailable. Test predictions or
test labels can never select epochs, model parameters, feature columns, scalers, anomaly
thresholds, novelty thresholds, or ensemble weights.

Models are not refit on train-plus-validation after the validation decisions. This keeps the
actual seed model that produced the validation evidence and its selected checkpoint intact, and it
matches the v1 train/validation boundary used to construct the comparison.

## Global test-access gate

The gate is global, not seed-by-seed. Before any test target is loaded, all five seeds must have
completed training and validation selection. `manifest.json` must contain a canonical selection
freeze covering:

- all frozen input identities and ordered split hashes;
- code revision, dirty-state declaration, dependency/runtime snapshot, configuration hash, and
  the five-seed list;
- exact feature schema and model recipes;
- training-only scaler identities;
- selected XGBoost iterations and MLP epochs for all 12 targets/seeds as applicable;
- all five traditional ensemble weight vectors;
- all five supplemental ensemble weight vectors, or one explicit omission reason;
- MLP curve integrity and anomaly/warning status for every seed; and
- hashes of all validation prediction arrays, including fixed MIST validation predictions when
  the supplemental layer is enabled.

The canonical selection-freeze subtree is hashed, written atomically, fsynced, and then read back
and verified. Only that hash authorizes one test-label access. The loader must expose no test
labels to fitting or validation-selection functions. The manifest event sequence records
`input-verified`, five `seed-selection-frozen` events, `global-test-gate-frozen`, five
`seed-test-complete` events, and `summary-frozen`.

V2 is deliberately fail-closed and does not resume an incomplete run. A new run requires a new,
nonexistent output directory. If an output directory contains an incomplete manifest, failed seed,
partial prediction set, or a gate without the final summary event, the runner rejects it and
requires a different output directory; it never overwrites, continues, or reauthorizes that gate.
The incomplete directory is retained as failure evidence. A complete output directory may be
reused only as a read-only completed result after the runner verifies the config, input, code,
selection-gate, event-sequence, and every recorded artifact hash and byte size. Any missing or
changed artifact rejects reuse.

After the gate, each frozen model predicts the test features in the exact fixed test-row order.
The test labels are loaded once to calculate the predetermined metrics. No test result may trigger
a retry with a new seed, altered epoch, weight, threshold, feature, or model recipe.

## Metric and uncertainty contract

Let `s_j` be the population standard deviation (`ddof=0`) of target `j` on the fixed training rows.
It is calculated once without test or validation labels. For model `m` and target `j`:

```text
MAE_j(m)  = mean_i |prediction_ij(m) - truth_ij|
NMAE_j(m) = MAE_j(m) / s_j
MNMAE(m)  = unweighted mean of NMAE_j(m) over all 12 targets
```

Each model also reports native-unit RMSE and R² for every target. The primary scalar is MNMAE;
lower is better. No target is weighted by its numerical scale, sample variance, or perceived
importance.

### Seed variation

For each traditional method, calculate all metrics separately for each of the five seeds. Report
the arithmetic mean and sample standard deviation (`ddof=1`) across the five seed-level values.
The seed standard deviation describes observed training stochasticity on this fixed split. Ridge's
standard deviation may be zero; it must not be replaced by an artificial nonzero value.

### Fixed MIST uncertainty

MIST has one frozen prediction per test row, not five training runs. It therefore receives no
"seed standard deviation." Its uncertainty is a paired-row percentile bootstrap over the fixed
test cohort: 2,000 resamples, 95% interval, using one frozen bootstrap seed recorded in the
manifest. Each replicate samples 13,389 row positions with replacement and uses the same sampled
positions for all 12 targets and every compared model. The interval describes sensitivity to the
composition of this candidate test cohort, not MIST fine-tuning randomness, checkpoint
uncertainty, or performance on a new chemical population.

### Paired deltas versus MIST

For traditional method `m`, first take the arithmetic mean of its five predictions at each fixed
`[row, target]` position. This produces one explicitly labeled **seed-averaged predictor**; it does
not add rows and it is not substituted for the separately reported mean and standard deviation of
the five seed-level metrics. Within each bootstrap replicate, score this seed-averaged prediction
and the fixed MIST prediction on identical sampled row positions, then subtract MIST from the
traditional result.

The reported paired point delta and percentile interval are therefore
`seed-averaged-predictor traditional MNMAE - fixed MIST MNMAE`; negative favors the traditional
predictor. Per-target paired deltas use the same sampled rows without the across-target mean. The
5 × 13,389 predictions must never be treated as independent observations. Because averaging
predictions before taking absolute errors can outperform the mean individual-seed score, this
paired result is reported as a seed-ensemble comparison and never mislabeled as the performance of
a typical single seed.

The row-bootstrap interval and the across-seed standard deviation are reported separately because
they represent different sources of variation. They are not combined into one ambiguous error
bar. A paired interval excluding zero is descriptive evidence on this fixed reconstructed split,
not proof of general superiority.

## Fixed-test structural novelty analysis

Structural novelty is a subgroup analysis of the same 13,389 test rows. It is not a new test split
and never changes model selection.

Use the authenticated, row-aligned chirality-aware Bemis-Murcko scaffold identifiers to assign
every fixed test row to exactly one of two frozen cohorts:

- `seen_scaffold`: the identifier occurs in the fixed training rows; or
- `unseen_scaffold`: the identifier does not occur in the fixed training rows.

For acyclic molecules, use the full canonical structure as the identifier rather than grouping
every acyclic molecule under an empty scaffold. Hash the scaffold artifact, the ordered test-cohort
labels, and each cohort's ordered source-row indices before the test gate. The two cohorts must be
disjoint and together cover all 13,389 fixed test rows. V2 does not promise or report nearest-train
Tanimoto bins; adding a similarity cache or thresholds requires a later protocol version.

For every stratum, report the membership rule, row count, row-index hash, each method's MNMAE and
12 target NMAEs, traditional seed mean/sample standard deviation, fixed-MIST point estimate, and
paired seed-averaged-predictor-minus-MIST deltas. Empty strata remain visible with `status: empty`;
they are not merged after inspection. Stratum results are descriptive and cannot feed back into
features, thresholds, hyperparameters, epochs, or ensemble weights.

## Required artifact contract

All JSON is UTF-8, finite-valued, key-sorted, indented, newline-terminated, and atomically replaced.
All NPY files use `allow_pickle=False`, shape `[13_389, 12]`, native target units, and frozen test
row order. `manifest.json` records SHA-256 and byte size for every artifact.

### `manifest.json`

Required top-level content:

- schema and protocol IDs plus `planned`/`complete` scientific status;
- frozen source, split-membership, feature/schema, target-order, MIST checkpoint/prediction, code,
  config, runtime, and dependency identities;
- seed list and bootstrap settings;
- validation-selection freeze and global test-gate SHA-256;
- exact test-access count and gate event sequence;
- artifact hashes for every seed JSON, prediction NPY, scaffold/cohort cache, summary, and loss
  monitor;
- all-model-ensemble availability or explicit omission reason; and
- completion/failure records without deleting failed attempts.

### `seeds/<seed>.json`

Each of the five files records:

- seed, fixed split hashes, target order, and status;
- Ridge, XGBoost, and MLP frozen recipe plus selected validation checkpoint/iteration;
- train-only input/target scaler provenance;
- validation metrics and validation prediction hashes;
- MLP epoch curves, best epoch, restored-checkpoint confirmation, and anomaly/warning flags;
- traditional ensemble validation weights and objective;
- supplemental all-model weights/objective or omission reason;
- test metrics for every available method and all 12 targets;
- training, validation inference, test inference, and total wall time by method, with rows/second and
  the process RSS high-water mark labeled as process-level rather than additive model memory; and
- hashes of the seed's test prediction arrays.

### `predictions/<seed>-<model>.npy`

Required model keys are `engineered_ridge`, `xgboost`, `mlp`, and `traditional_ensemble`.
`all_model_ensemble` is additionally required only when authenticated fixed MIST validation
predictions enabled its validation-selected weights. MIST itself remains a separately hashed fixed
input rather than being copied five times and falsely presented as seed-specific output.

### `summary.json`

Required content:

- schema `qm9-fixed-mist-split-v2-summary-v1`, completion status, target order, row counts, and all
  input/gate identities;
- for every traditional method and every target: five seed values, arithmetic mean, sample standard
  deviation, minimum, and maximum for MAE, NMAE, RMSE, and R²; plus the same seed aggregation for
  MNMAE;
- fixed MIST native metrics, MNMAE, and 2,000-resample paired-row 95% intervals for aggregate and
  all 12 target NMAEs, with no seed-variance field;
- seed-averaged-predictor traditional-minus-MIST paired point deltas and 95% row-bootstrap
  intervals for MNMAE and all 12 targets, clearly separated from single-seed summary statistics;
- primary/supplemental labels that keep the traditional-only ensemble primary and the all-model
  ensemble supplemental;
- the complete structural-novelty strata contract above;
- per-seed/per-model runtime, aggregate runtime mean/sample standard deviation/sum, inference
  throughput, and the separately labeled historical fixed-MIST inference runtime; and
- anomaly, warning, failed-seed, omitted-artifact, and limitation summaries.

### `loss-monitor.html`

This is a local, self-contained report generated only from the five seed JSON files. It shows, for
each MLP seed, the training-loss and correctly labeled validation-error curves, best/restored epoch,
early-stop reason, validation MNMAE, runtime, and `normal`/`warning`/`abnormal` status. A rise after
the best validation epoch is visibly marked. The report contains no raw molecules, labels, or
predictions and is not itself a selection interface.

## Failure policy and honest limitations

The run stops before test access if an input identity, row order, shape, finite-value check, scaler
boundary, validation-selection record, MLP curve integrity check, or global gate check fails. After
test access, a defect may invalidate the run, but test results cannot be used to repair a scientific
choice in place; a revised protocol/version is required.

V2 supports a narrow claim. The split is a candidate reconstruction from public MIST code because
the checkpoint publisher did not release the historical row-ID split manifest. The test rows may
overlap MIST's pretraining corpus, and the study does not isolate the causal effect of pretraining,
LoRA, head-only training, or full fine-tuning. QM9 targets are quantum-chemistry calculations, not
laboratory measurements. Five traditional seeds measure limited optimizer/initialization variation
on one fixed split; they do not measure split uncertainty. The paired row bootstrap conditions on
the same finite test cohort and does not create MIST training-seed uncertainty. Novelty strata are
post-fit descriptive subgroups, not external validation. Runtime comparisons span different
software/hardware routes and must not be treated as controlled efficiency benchmarks.

A true new random or scaffold split is outside v2. It would require split-specific MIST fine-tuning
with auditable training provenance before MIST could be evaluated there. The released fine-tuned
checkpoint must never be moved onto such a split and reported as leakage-free evidence.
