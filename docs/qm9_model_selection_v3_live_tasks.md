# QM9 post-specified model selection v3 live tasks

Status date: 2026-07-18

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
- [ ] Private input/EDA preflight completed.
- [ ] Private W1-W3 train-only model selection run started.
- [ ] One outer-validation confirmation executed.
- [ ] Test-unlock approval issued.
- [ ] V3 test labels read exactly once.
- [ ] Publication review completed.

## Current QA decision

Final scoped QA decision: `APPROVE_PRIVATE_W1`. The W1 CLI now persists a governed failure manifest,
recursive Why record, checksum-valid failure lock, zero-fit evidence, and false outer/test read
flags when a forbidden prior lock is supplied. The full 29-test synthetic suite and ruff pass.
Approval is limited to executing W1 `scaler-ablation` once in a fresh permanent output directory
with the exact reviewed config and authenticated four-artifact input set. It does not approve W2,
W3, outer confirmation, test access, publication, or any scientific/model-ranking claim.

## Evidence boundary

- V1 already observed the fixed candidate validation and test scores.
- V2 governance and preflight work is complete, but the formal v2 five-seed directory is read-only
  and must not proceed to Stage B as part of v3.
- V3 implementation is currently in progress.
- No private v3 feature/model selection has run.
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

## Next implementation gate

Next governed gate:

1. choose a fresh permanent W1 output directory and record the exact command/config/repository and
   authenticated input hashes before execution;
2. execute W1 `scaler-ablation` without `--prior-selection-lock` and preserve the complete output;
3. independently verify the 123-fit ledger, all 41 complete candidates, source/fold provenance,
   failure/anomaly evidence, family-coverage promotion, zero outer/test reads, and lock checksums;
4. issue a hash-bound W1 artifact review before considering W2; and
5. do not start W2 under this W1-only approval.

## Durable rules

- Do not resume, mutate, publish, or use the formal v2 directory for v3.
- Do not call the outer validation or test result blind, unseen, prospective, or unbiased.
- Do not inspect outer-validation labels during W1-W3.
- Do not use the one outer confirmation to choose a fallback candidate.
- Do not read test labels without a bound approval after a confirmed outer-validation result.
- Do not rerun after outer/test access under the same protocol because a score is disappointing.
- Preserve incomplete and failed directories plus recursive Why records.
- Treat smoke, preflight, EDA, inner-CV ranking, and unapproved outputs as engineering evidence only.
