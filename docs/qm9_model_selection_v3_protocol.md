# QM9 post-specified model selection v3 protocol

Status: protocol being implemented; no private v3 selection or v3 result has run

Protocol date: 2026-07-18

Author: Lu Zhang (张鲁), independent research

## Scope and historical boundary

V3 is a new, post-specified model-selection study. It is **not** a v2.1 governance patch and it
does not change, resume, or complete the formal v2 run. Every formal v2 output directory is
read-only; v2 must remain before Stage B unless its own protocol is independently resumed later.

The candidate reconstructed QM9 validation and test cohorts were already examined during v1, and
the project team knows their historical scores. Therefore neither cohort is blind in the ordinary
scientific sense. V3 may enforce fresh procedural access gates, but it must never describe the
outer validation or test result as prospectively blind, untouched, or unbiased. V3 results remain
post-specified evidence on one historically observed candidate split.

V3 answers a narrower engineering question:

> Within the fixed training membership, which preregistered molecular representation and
> traditional model pipeline is most stable under identity-group inner cross-validation and a
> scaffold-held-out stress analysis, and does that frozen choice confirm once on the historical
> outer validation cohort before one governed test evaluation?

The released fine-tuned MIST predictions are a fixed reference. MIST weights, loss, scaling,
preprocessing, and predictions are not tuned by v3 and never participate in inner model selection.

## Non-negotiable boundaries

- Use the existing authenticated QM9 source and candidate reconstructed train/validation/test row
  identities; do not create a new MIST comparison split.
- Perform all exploratory data analysis, feature comparison, hyperparameter search, early stopping,
  and ensemble construction inside the fixed training membership.
- Keep canonically identical molecules in the same inner fold.
- Use scaffold-held-out folds only as a train-only stress analysis, not as a replacement test set.
- Open the historical outer validation labels exactly once after the inner selection freeze. They
  confirm or invalidate the frozen pipeline but cannot select a replacement.
- Open historical test labels at most once, only after outer-confirmation approval. No test result
  may trigger a same-protocol retry.
- Preserve every failed or incomplete directory. Only explicitly authorized, hash-bound stage
  transitions may continue; every other retry uses a fresh directory.
- Never relabel v1 or v2 artifacts as v3 evidence.

## Candidate feature representations

All candidates are deterministic views of the authenticated, row-aligned v1 molecular feature
artifact. They contain no labels, predictions, split names, or scaffold indicators.

| ID | Representation | Frozen transformation |
|---|---|---|
| `all_raw` | Count-ECFP4 plus 17 descriptors | All 2,065 columns unchanged |
| `descriptor_scaled_only` | Raw count-ECFP4 plus 17 descriptors | Standardize descriptor columns on the inner-fit rows; despite its ID, it retains all ECFP columns |
| `all_sparse_scaled` | Count-ECFP4 plus 17 descriptors | Standardize all columns without centering, fitted on inner-fit rows |
| `log_count_descriptor_scaled` | Count-ECFP4 plus 17 descriptors | Apply `log1p` to nonnegative ECFP counts and inner-fit standardization to descriptors |
| `drop_constant_descriptor_scaled` | Count-ECFP4 plus nonconstant descriptors | Drop inner-fit constant descriptor columns and standardize retained descriptors |

Variance removal, learned feature selection, target-guided filtering, new descriptors, alternative
fingerprint radii/lengths, embeddings, and MIST representations are outside v3. Any later addition
requires a new protocol version before outer-label access.

## Candidate model families

The machine-readable config must enumerate every trial before the private selection run. The
following families are permitted:

1. **Ridge:** independently regularized multi-target linear regression with a log-spaced alpha
   grid. Inputs are scaled using the inner-training fold only.
2. **XGBoost:** bounded depth, learning-rate, child-weight, row/column subsampling, regularization,
   and estimator-cap candidates; early stopping uses only the corresponding inner-validation fold.
3. **MLP:** bounded hidden-width/depth, dropout, learning-rate, and weight-decay candidates. Input
   and all 12 target scalers fit only on inner-training labels; the best inner-validation checkpoint
   is restored.
4. **Traditional ensemble:** a nonnegative, sum-to-one blend of frozen finalists fitted only from
   train-membership out-of-fold predictions. It is evaluated as a candidate output, not tuned on
   outer validation or test labels.

MIST and any ensemble containing MIST are excluded from v3 selection. A later supplemental
comparison may score the immutable MIST artifact after test unlock, but it cannot alter the chosen
traditional pipeline.

## Train-only resampling contract

### Primary identity-group inner CV

Canonical molecular identity is the grouping unit. All rows for one identity stay in one fold.
The config freezes fold count, fold-assignment seed, ordered row indices, group hashes, and target
order before any candidate runs. The primary selection metric is mean normalized MAE (MNMAE),
using target population standard deviations fitted on each inner-training fold.

### Scaffold stress analysis

A separate Bemis-Murcko group assignment creates train-only scaffold-held-out folds. Acyclic
molecules use their frozen full canonical structure identity rather than one empty scaffold. This
analysis measures robustness under a harder partition; it does not change the outer cohorts and
does not claim new-population generalization.

Candidate ranking is lexicographic and frozen in the config:

1. lowest mean primary identity-group inner-CV MNMAE;
2. within the configured one-standard-error set, lower scaffold-stress MNMAE;
3. then lower across-fold dispersion;
4. then lower declared complexity/runtime tier; and
5. finally a stable candidate ID tie-break.

The exact one-standard-error calculation and every tie-break must be machine-readable before the
private selection run. No outer validation/test value may affect ranking.

## Budgeted search waves

Budget limits are maximums, not targets that must be exhausted. Failed trials remain visible and
cannot be silently replaced.

| Wave | Purpose | Maximum budget | Promotion rule |
|---|---|---:|---|
| W0 | Synthetic contract/governance smoke | One tiny fixture per family | Schema, state, determinism, and failure-path checks only; never scientific evidence |
| W1 (`scaler-ablation`) | Broad train-only screen | Exactly 41 feature/model/config tuples; 3 identity-group folds; 1 training seed; at most 123 candidate-pipeline fits | Promote at most 12 by frozen primary rank, including at least two complete candidates from each of Ridge, XGBoost, and MLP when available |
| W2 (`feature-screen`) | Train-only refinement | At most 12 candidates from the hash-validated W1 lock; 5 identity-group folds; 1 training seed; scaffold stress only for the identity one-SE set; at most 120 candidate-pipeline fits | Promote at most three using the frozen identity-primary, one-SE, scaffold-stress, dispersion, complexity, runtime, and stable-ID rule |
| W3 (`full-selection`) | Robustness and ensemble confirmation | Two or three finalists from the hash-validated W2 lock; 5 identity folds and 5 scaffold folds; 5 training seeds; exactly 150 fits for three finalists | Compare the frozen base finalists and one train-only nonnegative sum-to-one OOF ensemble; freeze one primary result |
| W4 | One outer-validation confirmation | Frozen winner(s) only; exactly one label access | Confirm or stop; never return to W1-W3 under v3 |

The config must also freeze per-family trial counts, wall-time ceilings, CPU/GPU policy, failure
handling, and the definition of a complete trial. Compute exhaustion may reduce evidence and stop
the study, but it cannot justify selecting the best partial result or adding replacement trials.

W2 requires the preceding W1 `selection_lock.json`; W3 requires the preceding W2 lock. Each lock
must validate its own checksum and artifact hashes and must bind the same authenticated input
identity, fixed-training row order, config hash, and wave lineage. A lock from another data or
feature artifact is invalid even if its config hash and preceding-wave name match.

The current public smoke does not execute the real XGBoost or MLP training backends:
XGBoost-shaped candidates use a clearly labeled Extra-Trees surrogate, and MLP monitoring uses a
deterministic synthetic curve. These are state-machine tests only and cannot promote a candidate,
consume a search budget as scientific evidence, or be called model comparison results.

## One outer-validation confirmation

After W3, write and hash a selection freeze containing the candidate registry, folds, all trial
outcomes, chosen feature/model/config, ensemble rule, preprocessing/scalers, seeds, prediction
hashes, anomaly decisions, and the zero outer-validation/test read counters. An independent
selection reviewer must approve that exact state.

Only then may W4 load outer-validation labels once. Before access, the config must define the
confirmation checks: finite/aligned predictions, all 12 target metrics, the expected interval
derived only from inner-CV evidence, and any failure threshold. W4 can return `CONFIRMED` or
`NOT_CONFIRMED`; it cannot choose another candidate, tune a threshold, change weights, or rerun.
If not confirmed, v3 stops without test access. A revised study must use a new protocol lineage.

Because this outer cohort was historically observed in v1/v2 work, `CONFIRMED` means procedural
consistency with the post-specified contract, not an unbiased replication.

## G0-G7 Plan -> Execute -> Review gates

| Gate | Plan | Execute artifact | Independent review |
|---|---|---|---|
| G0: historical disclosure | Freeze v3 scope, known v1/v2 exposure, candidate registry, budgets, and allowed claims. | Protocol/config/repository hashes. | Reject any blind/unseen language or hidden post-hoc candidate. |
| G1: input and EDA boundary | Freeze identities, folds, feature/scaffold artifacts, train-only EDA, and label counters. | Private preflight and EDA audit with zero outer/test reads. | Verify hashes, row order, grouping, label scope, and leakage controls. |
| G2: feature screen | Freeze W1 candidates and budget. | Complete W1 ledger and inner-fold predictions. | Check deterministic transforms, fold-local fitting, failures, and promotion rule. |
| G3: model refinement | Freeze W2/W3 candidates, seeds, caps, early stop, anomaly, and ranking rules. | Complete refinement/stress artifacts and recursive Why records. | Verify no selective retry, correct scaling, restored checkpoints, and frozen ranking. |
| G4: selection freeze | Freeze one pipeline and optional traditional ensemble before outer-label access. | Atomic/fsynced manifest and selection-freeze hash. | Bound approval verifies all inner evidence and zero outer/test reads. |
| G5: outer confirmation | Predeclare confirmation range and failure action. | One outer-validation read and confirmation report. | Approve test unlock only if the unchanged pipeline is confirmed; never reselect. |
| G6: test exactly once | Predeclare metrics, MIST reference use, bootstrap, subgroups, and claims. | One test read, immutable predictions, summary, and event log. | Verify `test_label_reads=1`, formulas, pairing, hashes, and no feedback. |
| G7: publication | Predeclare primary/supplemental tables, limitations, and wording. | Publication candidate plus full inventory/checksums. | Separate reviewer approves a hash-bound result before any builder/site/article consumes it. |

Automated checks cannot self-approve a human gate. Execution and review identities must be
recorded. A reviewer must not approve work they executed. Missing evidence yields
`DEFER_FOR_EVIDENCE`; a contract/leakage violation yields `REJECT` and invalidates the lineage.

## Test exactly-once and publication review

After a confirmed W4, a distinct approval binds the unchanged outer-confirmation manifest,
selection freeze, code/config hashes, and intended test event. G6 authorizes exactly one test-label
read. The loader and event log must reject a second read. Test predictions are generated by the
already frozen candidate; no refit, seed replacement, calibration, threshold change, or ensemble
weight change is allowed.

G6 ends with `AWAITING_PUBLICATION_REVIEW` and `publication_ready=false`. The publication reviewer
checks every artifact hash, all 12 targets, aggregate metrics, seed/fold summaries, bootstrap and
paired comparisons, scaffold stress/subgroups, runtime, failures, anomalies, and limitations.
Only a separate hash-bound publication transition may set `PUBLICATION_APPROVED` and allow a
v3-aware results builder to consume the run.

## Recursive Why record

Every failed invariant, abnormal loss, promotion surprise, incomplete trial, confirmation failure,
requested retry, or metric discrepancy receives a record before corrective work:

```yaml
issue_id: WHY-YYYYMMDD-NNN
gate: G0-G7
test_or_outer_labels_accessed: false
fact:
  observation: "what was directly observed"
  evidence_hashes: []
hypotheses:
  - statement: "possible cause"
    supporting_evidence: []
    contradicting_evidence: []
falsification:
  next_check: "read-only or train-only check"
  predicted_result_if_hypothesis_is_wrong: ""
root_cause:
  status: supported | unknown
  statement: "deepest evidence-backed cause"
  confidence: high | medium | low
correction:
  containment: "fail closed and preserve output"
  proposed_change: ""
  changes_scientific_contract: false
  requires_new_protocol_version: false
  fresh_output_required: true
review:
  independent_reviewer: ""
  decision: pending | approve | reject
  evidence_hashes: []
```

Why analysis recurses only while evidence supports another causal question. It may branch and may
end at `unknown`; it must not force a convenient story. If a proposed correction changes a
candidate, feature, fold, metric, threshold, ranking rule, budget, or label boundary, v3 ends and a
new protocol version is required. After outer/test access, no scientific choice can be repaired in
place.

## Reporting and limitations

No v3 output is a scientific result until W1-W4 are complete, G4 and G5 approvals bind the frozen
artifacts, G6 reads test labels exactly once, G7 publication review passes, and final checksums and
the builder gate verify. Smoke, EDA, preflight, partial search, inner-CV leaderboards, outer
confirmation alone, and any unapproved test output are not scientific results.

Allowed claims are descriptive: model-selection stability inside the fixed training membership,
scaffold-stress behavior inside that membership, one post-specified confirmation on the historical
outer validation cohort, and one governed evaluation on the historical test cohort. V3 cannot
claim blind replication, causal pretraining benefit, generalization to a new chemical population,
experimental battery performance, or MIST training uncertainty. The candidate split remains a
reconstruction from public code, and historical validation/test exposure creates unavoidable
optimism risk that procedural gating cannot erase.
