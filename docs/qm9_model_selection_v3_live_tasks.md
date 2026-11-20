# QM9 post-specified model selection v3 live tasks

Status date: 2026-07-19

## Objective

Implement the train-only model-selection protocol in
[`qm9_model_selection_v3_protocol.md`](qm9_model_selection_v3_protocol.md), then stop at each
independent gate. V3 is separate from v2 and does not make the historically observed outer
validation or test cohorts blind again.

## Current status

- [x] Independent v2/v2.1 documentation QA intake completed.
- [x] Need for a separate post-specified v3 lineage recorded.
- [x] Draft v3 protocol, G0-G7 gates, candidate registry, budget waves, and recursive Why template
  documented.
- [x] Public smoke and authenticated train-only wave CLI/config/artifact interfaces documented.
- [x] Fold-local recipe, identity/scaffold fold, real Ridge/XGBoost/Torch MLP, strict-authentication,
  lock-integrity, and explicit-smoke code paths implemented.
- [x] Independent protocol and implementation review rounds 1-4 plus final W1 CLI review completed.
- [x] Synthetic governance/search smoke completed (29 focused tests plus public smoke).
- [x] Private input/EDA preflight completed without outer-validation or test-label access.
- [x] Private W1 started from commit `8a52139` and was terminated at the frozen 43,200-second
  wall-clock budget without completing.
- [ ] Private W1-W3 train-only model selection completed.
- [ ] One outer-validation confirmation executed.
- [ ] Test-unlock approval issued.
- [ ] V3 test labels read exactly once.
- [ ] Publication review completed.

## Current QA decision and W1 outcome

Final scoped QA decision: `APPROVE_PRIVATE_W1`. The W1 CLI now persists a governed failure manifest,
recursive Why record, checksum-valid failure lock, zero-fit evidence, and false outer/test read
flags when a forbidden prior lock is supplied. The full 29-test synthetic suite and ruff pass.
That approval was consumed by the run in
`results/qm9-model-selection-v3-w1-8a52139-20260718T234500Z`. An external watchdog sent `SIGTERM`
at 43,201 elapsed seconds because the wave had not completed. The runner had written no durable
candidate, curve, ledger, checkpoint, manifest, or failure artifact. The empty/incomplete directory
is preserved. No partial model ranking exists and none may be reconstructed from resource logs.

Current decision: `REJECT_INCOMPLETE_RUNTIME`. W2, W3, outer confirmation, test access,
publication, and every scientific/model-ranking claim remain prohibited.

## Evidence boundary

- V1 already observed the fixed candidate validation and test scores.
- V2 governance and preflight work is complete, but the formal v2 five-seed directory is read-only
  and must not proceed to Stage B as part of v3.
- The first private W1 attempt ran for 12 hours but did not finish or produce a selection artifact.
- The W1 process remained alive, but missing live loss/progress telemetry made liveness insufficient
  evidence of candidate-level progress.
- No outer-validation or test labels have been read by the v3 runner.
- No v3 result exists, and no website/article/model-ranking claim may be updated from this work.

## Frozen implementation targets

- Identity-group inner CV is the primary train-only selection route.
- Scaffold-held-out CV is a train-only stress analysis.
- Feature candidates are `all_raw`, `descriptor_scaled_only`, `all_sparse_scaled`,
  `log_count_descriptor_scaled`, and `drop_constant_descriptor_scaled`.
- Model families are Ridge, XGBoost, MLP, and a future traditional-only OOF ensemble. The public
  smoke uses a labeled Extra-Trees surrogate for XGBoost candidates and a synthetic MLP curve; it
  is not real model training or a comparison result.
- Search proceeds through W0 smoke; W1 `scaler-ablation` (41 candidates, 3 identity folds, 1 seed,
  123-fit ceiling); W2 `feature-screen` (at most 12 W1 candidates, 5 identity folds plus one-SE
  scaffold stress, 1 seed, 120-fit ceiling); W3 `full-selection` (at most 3 W2 finalists, 5
  identity folds, 5 scaffold folds, 5 seeds, 150-fit ceiling, OOF ensemble); W4 one
  outer-validation confirmation; then separately approved test and publication gates.
- MIST is a fixed reference and never participates in v3 model selection.

## Next implementation gate: profiling and v3.1 observability correction

Next governed gate:

1. preserve the failed W1 directory and watchdog/resource evidence; do not resume or infer a rank;
2. run one fold and one real MLP configuration with stage timers for sparse slicing, `toarray`,
   tensor creation, host-to-device transfer, forward, backward, optimizer, validation, and
   checkpointing;
3. measure RSS, GPU allocated/reserved/peak memory, utilization, temperature, I/O, and epoch time;
4. choose CPU cache, GPU cache, chunk cache, memmap, or unchanged streaming only from profiling;
5. prove any execution-only optimization preserves the first batch, loss trajectory, and
   predictions within frozen tolerances;
6. implement append-only heartbeat/loss/progress telemetry, signal-safe failure evidence, durable
   candidate checkpoints, continuous wall/memory/temperature/disk gates, and complete provenance;
7. exercise the real Torch/XGBoost backends in an instrumented smoke before another long run; and
8. create and independently approve a new v3.1 lineage before rerunning W1. Do not start W2.

## Durable rules

- Do not resume, mutate, publish, or use the formal v2 directory for v3.
- Do not call the outer validation or test result blind, unseen, prospective, or unbiased.
- Do not inspect outer-validation labels during W1-W3.
- Do not use the one outer confirmation to choose a fallback candidate.
- Do not read test labels without a bound approval after a confirmed outer-validation result.
- Do not rerun after outer/test access under the same protocol because a score is disappointing.
- Preserve incomplete and failed directories plus recursive Why records.
- Treat missing or stale telemetry as unhealthy; process liveness alone is not progress evidence.
- Do not approve a run expected to exceed ten minutes without real-backend live observability and
  a measured peak-memory gate.
- Treat smoke, preflight, EDA, inner-CV ranking, and unapproved outputs as engineering evidence only.
