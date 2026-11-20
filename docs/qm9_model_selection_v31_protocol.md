# QM9 model selection v3.1 execution-governance protocol

Status: the fixed v5 bounded CPU synthetic smoke and its read-only verification are complete and
accepted as engineering evidence only. V1-v4 remain immutable evidence. A medium train-only
profile is proposed below but is not prepared, reviewed, authorized or executed. No private W1,
W2, GPU profile, ranking, ensemble, promotion or scientific use is approved.

The completed v5 lineage is fail-closed and read-only at `/tmp/qm9-v31-bounded-smoke-v5`, run ID
`qm9-v31-bounded-smoke-cpu-v5`, runner identity `qm9-v31-bounded-runner-v5`, and v5 lifecycle
schemas. Its one-use execute and verify authorizations are consumed. The prior v1-v4 paths are
explicitly rejected for reuse. The accepted integration-v3 result and fixed-v5 result are
engineering-only seam evidence; neither is a fixed-train capacity result or a scientific result.

Protocol date: 2026-07-19

Author: Lu Zhang (张鲁), independent research

## Why v3.1 exists

The first private v3 W1 attempt used clean commit `8a52139` and output directory
`results/qm9-model-selection-v3-w1-8a52139-20260718T234500Z`. It reached the frozen 43,200-second
budget before all 123 fits completed. An external watchdog sent `SIGTERM` at 43,201 seconds. The
runner left no durable heartbeat, candidate record, curve, partial ledger, checkpoint, manifest, or
failure artifact. The empty directory is preserved. No outer-validation or test labels were read.

This is `REJECT_INCOMPLETE_RUNTIME`, not a negative model result. It supplies no candidate ranking,
promotion, completed-fit count, or evidence identifying the slow stage. The supported root cause is
a governance defect: v3 approved a long run without durable live observability and signal-safe
failure evidence. The computational cause remains `unknown`. Sparse-to-dense conversion, repeated
transforms, serial 12-target XGBoost, validation, transfer, optimizer, memory pressure, and other
stages remain competing hypotheses until real-backend profiling falsifies them.

## 2026-07-19 Wave 1 component QA

The initial observability, guard, checkpoint and profiled-backend components passed their bounded
unit suite, but independent QA returned `REQUEST_CHANGES`. Passing component tests does not satisfy
P0-P7. Reproduced gaps included mixed run IDs accepted in one event chain, no governed truncated-
tail reopen, checkpoint semantic metadata tamper accepted, and a non-12-target arbitrary record
accepted by the direct completed-checkpoint boundary.

Remediation must add and independently verify:

- stable start/run binding on every event and a reviewed event vocabulary/schema;
- durable truncated-tail quarantine before append and parent-directory fsync after replace;
- semantic checkpoint metadata integrity and strict plan+ledger completed-only replay;
- immutable, mutually exclusive failure/final terminal states and partial-ranking tripwire;
- authenticated hashes of actual cached data plus a frozen profile/choice binding;
- complete measured memory/runtime feasibility rather than a feature-size estimate;
- governed real smoke for every backend and intended hardware path; and
- no approval for a skipped or unexecuted CPU/GPU path.

## 2026-07-19 integration remediation re-review

The second independent review accepted the component fixes for run/event binding, truncated-tail
recovery, directory durability, checkpoint semantic validation, completed-checkpoint schemas,
cache-content identity, frozen cache choice, CPU reference equivalence and the exercised CUDA cache
path. The focused suite passed, but the decision remained `REQUEST_CHANGES` because the components
do not yet form one governed execution.

Integration remediation must establish these additional invariants:

- exactly one owner appends reserve/start/complete/fail and terminal transitions;
- the supervisor derives completion and replay identities from the frozen plan and its actual
  durable ledger; callers and backends cannot supply authoritative ledger hashes;
- a successful worker exit is incomplete until the supervisor authenticates the required
  checkpoint and appends exactly one completion;
- memory/runtime feasibility artifacts and device identity are startup hard gates, not advisory
  hashes or report fields;
- sidecar sensors externally observe the child/process group and fail closed for missing required
  RSS, GPU allocated/reserved, temperature or disk evidence; and
- a multi-trial supervisor returns after each successful trial and creates `final/` only after all
  planned trials authenticate and reconcile.

Unit/component evidence cannot close these items. A synthetic/public seam test must run the real
supervisor, sidecar, dispatcher, checkpoint store, replay gate and terminal state together for
Ridge, XGBoost and MLP. Until a third independent review accepts that path, no governed smoke or
private W1 is authorized.

## 2026-07-19 third independent review

The third review verified the integrated supervisor seams and issued
`APPROVE_BOUNDED_TRAIN_ONLY_PROFILE_SMOKE`. The supervisor now owns all transitions, derives the
completion contract from the frozen plan and durable START row, authenticates the checkpoint before
one completion, performs strict replay before finalization and whenever ranking is requested, and
refuses stale/unapproved feasibility artifacts. Multi-trial execution cannot finalize early. The
external sidecar observes the child process group and passed real-child RSS, disk,
GPU-sensor-unavailable, signal and hard-kill tests. Real Ridge/XGBoost/MLP passed the common
supervisor path, and Ridge additionally passed the real subprocess worker/sidecar path.

The only approved next execution is one fresh bounded CPU profile smoke using public/train-only
rows, zero fixed outer/test reads, short reviewed limits and real subprocess workers for all three
families. It is engineering evidence only and may not rank or promote candidates. XGBoost/MLP still
need their own subprocess evidence. GPU remains unapproved because the external NVML process-used
memory value is a conservative proxy copied to allocated/reserved fields, not distinct framework
telemetry. Another independent review is required before private W1.

The subsequent execution pre-review approved that one smoke through an exact
`prepare -> external review -> authorize -> execute` lifecycle. The approval is recorded in
`/tmp/qm9_v31_bounded_smoke_prereview.md` (SHA-256
`4425b6627f2e292e512b9206eaf51fc643b4f3495721f498bd0037f54cdd8976`). It is not a normal
feasibility approval: the post-profile proposal must remain
`INSUFFICIENT_OBSERVATIONS/NOT_APPROVED`, and a separate post-smoke review is mandatory before any
private W1 decision.

That v1 execution approval was consumed by a failed `prepare` claim. The runner located `uv.lock`
using an incorrect fixed parent depth after creating the one-use base. V1 is preserved as a partial
preflight lineage and is permanently ineligible for reuse or authorization; no worker, model fit,
private label or scientific result occurred. Any v2 attempt requires semantic repository-root and
lock discovery before claim, a durable fsynced failure artifact for every post-claim exception,
focused failure tests, a new base/run ID and a separate independent approval.

The v2 remediation is implemented but not execution-approved. A unique ancestor walk requires the
expected package, regular non-symlink markers and exact parsed project identity, then requires the
same root from `git rev-parse --show-toplevel`. Before base creation it freezes root identity/path,
runner identity/path/hash, pyproject and lock lstat mode/size/hash, and the complete code-state hash.
Immediately after base creation it writes an exact, exclusive and fsynced `prepare-claim.json`.
Success writes an exact `PREPARE_COMPLETE.json` binding the attempt, claim, complete lstat preflight
inventory, pre-complete base tree and profile; unexpected base-level entries fail closed. Failure
writes an exact `PREPARE_FAILED.json` binding the attempt, claim,
stable failure code, last checkpoint, partial lstat tree and explicit nonresume/nonauthorize/
nonexecute flags. Claim-write failure has its own exact durable terminal, and even terminal-marker
write failure permanently burns the base. Failure absence uses `lstat`, so dangling symlinks cannot
bypass it. Review, authorization, binding and execution require claim → complete, absence of
failure, and exact current replay of the full repository provenance. The run-start schema is an
explicit v2 constant, and every v1 base request fails as `V1_INCIDENT_INCOMPLETE`. Safe focused
tests pass (27 profile-smoke tests with one integration test deselected and 15 supervisor tests),
but no v2 prepare, integration smoke or model process has run. A new independent hash-bound v2
read-only audit and execution review remain mandatory.

### V3 post-run reauthentication amendment

V2 completed its bounded computations under its frozen execution identity, but its official
post-run validator rejected the expected final `profile-summary.json` and `post-profile/` paths.
Its formal disposition is `REJECT_POST_RUN_REAUTH_GAP`: diagnostic engineering evidence only, not
an accepted authenticated smoke and not W1/general-feasibility evidence. V2 remains byte-for-byte
read-only. Later source changes cannot be represented as the v2 execution code, even when an
independent evidence audit can still verify its immutable files. V3 therefore uses a new run ID,
v3 bounded schemas and the fresh fixed base `/tmp/qm9-v31-bounded-smoke-v3`.

V3 writes and semantically validates the exact summary and `NOT_APPROVED` proposal before the
profile terminal. The bounded v3 terminal binds both files' physical and semantic hashes. After the
terminal, an exact `POSTRUN_MANIFEST.json` binds run start, ledger, terminal, summary, proposal and
the complete pre-manifest base lstat inventory. Public `assert_profile_complete_postrun()` and CLI
`--stage verify` then reauthenticate the frozen execution binding, lifecycle, authorization
consumption, run start, ledger, observations, events, checkpoints, replay, terminal, summary,
proposal, false boundaries and exact final tree. Verification records the current audit-code hash
separately and does not require it to equal the historical execution-code hash.

The v3 verifier uses the production `CheckpointStore.load_completed` and `ReplayContract`, so
checkpoint replay follows canonical filesystem-path order and authenticates NPZ manifests,
predictions and latest pointers. It also authenticates worker results/profiles, MLP loss/anomaly
records, resource telemetry, the exact final `run_complete` progress/terminal equality and every
terminal exclusion flag. The terminal binds worker, resource, receipt, summary and proposal hashes.

The reviewed v3 implementation later passed 275 non-integration tests before its one authorized
execution. That execution completed all three bounded workers and strict checkpoint replay, then
failed before terminal creation because its preterminal exact-tree check removed
`run/final/manifest.json` from the final contract but incorrectly retained the not-yet-created
`run/final/` parent directory. The same contract also reconstructed the cache-manifest filename with
plain `canonical_sha256(identity)`, predicting `identity-71f177...json`, while the producer's
schema-aware `cache_identity_sha256()` emitted `identity-a7a879...json`. V3 is therefore
`REJECT_INCOMPLETE_PRETERMINAL_FINAL_TREE_CONTRACT`: immutable, nonresumable diagnostic engineering
evidence only. It grants no feasibility, ranking, promotion, ensemble, scientific or publication
permission. The absence of a durable failure terminal for this post-claim finalization exception is
a second governance defect.

Future remediation must use a fresh lineage and separate exact before/after contracts for every
filesystem transition. Leaf artifacts and parent directories must be added or removed together
unless an earlier authenticated phase explicitly creates the parent. Hash-addressed paths must use
the producer's exported schema-aware identity helper, not a locally reconstructed hash. The
production seam test must
start without `run/final/`, traverse summary/proposal -> preterminal check -> terminal ->
`run_complete` -> stream close -> success lock, and inject a failure at every seam to prove a durable
failure terminal is written. V3 must not be repaired, resumed, backfilled or reclassified.

## Execution-only boundary

V3.1 preserves the v3 scientific question and, unless a separately reviewed amendment says
otherwise, preserves:

- authenticated QM9 fixed train/outer/test memberships and ordered source rows;
- the five feature recipes and complete candidate tuples;
- identity/scaffold folds, fold seed, training seeds and target order;
- MNMAE formulas, one-standard-error rule, tie-breaks and family-coverage promotion;
- Ridge, XGBoost and dense MLP mathematics, preprocessing, optimizer and early-stop semantics;
- W1/W2/W3 fit ceilings and the traditional OOF ensemble contract;
- protected-label boundaries and G0-G7 approvals.

Caching, memmap, tensor reuse, scheduling, telemetry and checkpoint I/O are execution-only only
after the equivalence gate in P5 passes. A change to features, model mathematics, candidate count,
folds, seeds, target scaling, metrics, ranking, early stopping, fit ceiling or wall budget is a
scientific-contract change. It requires an explicit v3.1 amendment and new reviewed hashes before
any private run. Sparse-native neural architecture is not an execution-only repair.

## Medium train-only profiling amendment

This amendment defines the smallest next capacity experiment. Its sole question is whether the
real QM9 feature, 12-target, model, loss-monitoring, checkpoint and subprocess paths fit a bounded
CPU envelope with durable evidence. It is **not W1**, does not compare model quality and cannot
select a candidate. Diagnostic train-only metrics cannot be converted into a winner, ensemble,
promotion, paper result or W1 approval.

### Frozen workload contract

- Input membership is the authenticated fixed-train cohort only (`107,108` source rows). The
  runner may load fixed-train SMILES, the real 2,065-column representation (2,048 count ECFP4
  columns plus 17 descriptors), and all 12 fixed-train targets. It must not construct an
  outer-validation or test target loader.
- Select a target of 8,192 rows without consulting targets: sort complete canonical-identity
  groups by `SHA256("20260713" || canonical_identity)` and append whole groups until the target is
  first reached or exceeded. Persist the exact ordered row, identity-group, feature and target
  source hashes. The actual row count may therefore be slightly above 8,192.
- The value 8,192 is deliberate: it is a power-of-two medium workload, approximately 7.6% of the
  107,108-row fixed train, large enough to exercise realistic 2,065-column dense conversion,
  minibatching, 12 serial XGBoost targets and checkpoint costs, while remaining small enough for a
  single bounded CPU review cycle. It is a capacity probe, not a statistically representative
  benchmark. Any model-quality claim would require a separate scientific amendment.
- Make one deterministic identity-group 90/10 train/tune partition within the selected rows.
  Neither membership nor the split may depend on target values. Use fold/sampling seed `20260713`
  and training seed `20260713`; one seed is sufficient for execution profiling but supplies no
  seed-variance evidence.
- Run exactly one predeclared probe per family, without search: Ridge
  `descriptor_scaled_only/alpha=1.0`; XGBoost `shallow_fast` with depth 4, learning rate 0.08,
  subsample 0.9, column sample 0.9, at most 200 rounds and patience 35, fitting the 12 targets
  serially; and MLP `compact` with hidden widths `[384, 192]`, dropout 0.05, learning rate 0.001,
  weight decay `1e-5`, batch size 512, at most 20 epochs, patience 10 and minimum delta `1e-4`.
  These are existing capped v3 probes, not newly selected candidates. Fixed thread policy,
  precision and deterministic settings must be hash-bound.
- Ridge, XGBoost and MLP each end independently as `PASS`, `FAIL` or `INCOMPLETE`. Do not invoke
  candidate ranking, one-standard-error selection, family coverage, ensemble fitting, selection
  lock or promotion code.

The one-probe design isolates execution cost. Ridge exercises sparse preprocessing and a linear
solve; shallow XGBoost exercises the expensive 12-estimator serial path without the 650-1,100-round
W1 range; compact MLP exercises dense conversion, 12-target scaling, optimizer state, loss curves,
best-checkpoint restore and early stopping without the 80-epoch W1 cap. Changing these values after
preparation burns the lineage. Using their scores for comparison changes this amendment from
execution-only to scientific and requires a new protocol version.

### Frozen budget and metric gates

The proposed outer wall limit is 2,400 seconds. The child workload deadline is 2,280 seconds and
the last 120 seconds are reserved exclusively for checkpoint/failure flush, tree reconciliation and
terminal artifacts. Soft family budgets are Ridge 180 seconds, XGBoost 1,200 seconds and MLP 720
seconds, with 180 seconds of shared loading/orchestration allowance. A soft-family overrun is a
warning and stops scheduling the next family unless the supervisor proves the finalization reserve
remains intact; the 2,400-second limit is hard and cannot be extended in place.

The cgroup memory hard limit is 4 GiB. Profile acceptance requires observed aggregate peak RSS at
or below 3.2 GiB, retaining the P4 minimum 20% headroom. This ceiling covers the roughly 128 MiB
float64 or 64 MiB float32 dense 8,192-by-2,065 matrix itself plus sparse inputs, train/tune copies,
12 targets, tree/optimizer workspaces, subprocess/runtime overhead, telemetry and two checkpoint
generations; the measured process-group peak, rather than this estimate, controls the decision.
Require at least 5 GiB free disk before claim and cap produced artifacts at 1 GiB. Missing required
RSS/disk/heartbeat telemetry fails closed.

Execution metrics are stage wall time, CPU time, aggregate/process RSS, disk use, checkpoint time
and size, XGBoost per-target best iteration, and MLP epoch-level standardized train MSE, tune MNMAE,
12 per-target NMAEs, learning rate, gradient/parameter norms, best/restored epoch, stale count and
stop reason. Ridge/XGBoost/MLP tune MNMAE is a finite-value sanity signal only. A family is abnormal
for any nonfinite input/scaler/target/loss/gradient/parameter/prediction, invalid target shape/order,
missing checkpoint, failed best-state restore, protected-label tripwire, hard resource breach or
unexplained telemetry gap. Two consecutive MLP tune-loss increases are visibly marked as a warning;
they do not change the frozen stopping rule. Existing frozen anomaly thresholds remain diagnostic
and cannot trigger a replacement configuration.

At run start, every checkpoint and both terminal paths must assert:

```json
{
  "fixed_train_target_reads": "allowed_and_counted",
  "outer_validation_target_reads": 0,
  "test_target_reads": 0,
  "protected_label_reads": 0,
  "ranking_allowed": false,
  "ensemble_allowed": false,
  "promotion_allowed": false,
  "w1_allowed": false,
  "scientific_result": false,
  "publication_ready": false
}
```

### Independent pre-execution review checklist

Passing automated schema, hash, unit, integration or dry-run checks is necessary evidence, but is
**not independent approval**. The executor cannot self-approve, and a program-generated
`approved=true` field has no authority without a separately identified reviewer decision.

Before any preparation, the reviewer checks that the proposed base is fresh and permanent, all
older lineages are rejected, the command is prepare-only, and no output path aliases or symlinks an
older base. Preparation then writes and fsyncs the immutable plan and evidence and **must stop**:
it cannot create an authorization, nonce claim, executable run state, worker, model process or
label read. After preparation, an independent read-only reviewer must verify:

1. exact medium membership algorithm, ordered sample/group hashes and train-only label boundary;
2. exactly three trial keys and the frozen model, seed, feature, target, thread and precision
   contract above;
3. protocol, plan, code state, dependency lock, config, authenticated input, device identity and
   budget hashes;
4. fresh output identity, exact command, prepare claim/completion chain and absence of failure or
   executable state;
5. memory worksheet, measured/synthetic antecedent evidence, deadline/finalization reserve and
   required sensors;
6. all negative tests below and exact false-boundary assertions; and
7. reviewer independence, decision expiry, one-use scope and an explicit statement that the
   decision authorizes only this medium execute, never W1 or scientific use.

Any approval must bind the canonical hashes of **plan, protocol, code state, config, dependency
lock, authenticated input inventory, sampled membership, device identity and complete budget**.
It also binds the exact prepare claim/completion, reviewer identity, command, cwd, output path,
expiry and one-use nonce. A mismatch or later file/device/budget change invalidates approval; it may
not be patched or broadened in place.

### Required negative tests

Before approval, focused tests must demonstrate fail-closed behavior for at least:

- outer-validation/test target path, loader, split-name or counter access;
- target-informed sample selection, identity-group split leakage, row reorder and duplicate key;
- one missing, extra or changed trial; hyperparameter, feature recipe, target order, seed, thread or
  precision drift;
- plan, protocol, code, config, lock, input, sample, device or budget hash mismatch;
- stale/foreign reviewer, expired approval, reused nonce, wrong output, wrong cwd, wrong exact
  command and prepare-plus-execute chaining;
- prepare attempting authorization, execution, model spawn, nonce claim or protected-label read;
- nonfinite features/scalers/targets/losses/gradients/parameters/predictions and wrong 12-target
  prediction shape/order;
- missing RSS/disk/heartbeat sensor, stale heartbeat, telemetry discontinuity, disk floor, RSS
  warning/hard limit, soft deadline, finalization reserve and hard timeout;
- SIGINT/SIGTERM and hard kill during preprocessing, native XGBoost, MLP epoch and checkpoint write;
- truncated/tampered ledger, event, checkpoint, latest pointer, loss curve, worker result, terminal
  or tree inventory; crash after reserve/start/complete and terminal write failure;
- partial completion, mixed run/start hashes, same-directory rerun, extra artifact, surviving child
  or nonempty cgroup; and
- any call to ranking, ensemble, selection lock, promotion, W1 unlock, scientific-result or
  publication path, including after one or two successful families.

Each negative test must end incomplete or failed, preserve diagnostics, retain zero protected-label
reads and produce no authorization or promotion. Mocks may test local schema branches, but real
subprocess/signal/resource seams require a bounded forward test before approval.

### Real lifecycle CLI templates and mandatory stops

The implemented production entry point is the script below. These commands document the real
interfaces and mandatory stage boundaries; they are **not** current authority to prepare or execute
the private medium lineage. Placeholder paths and hashes must be replaced only by separately
reviewed, exact physical bindings.

```bash
uv run python scripts/run_qm9_model_selection_v31_medium_profile.py \
  --stage prepare \
  --output /tmp/qm9-v31-medium-profile-v1 \
  --identity qm9-v31-medium-runner-v1 \
  --qm9-csv /ABSOLUTE/AUTHENTICATED/qm9.csv \
  --feature-matrix /ABSOLUTE/AUTHENTICATED/features.npz \
  --feature-manifest /ABSOLUTE/AUTHENTICATED/feature-manifest.json \
  --phase1-dir /ABSOLUTE/AUTHENTICATED/phase1

uv run python scripts/run_qm9_model_selection_v31_medium_profile.py \
  --stage review \
  --output /tmp/qm9-v31-medium-profile-v1 \
  --decision-artifact /ABSOLUTE/EXTERNAL/review.json \
  --expected-sha256 REVIEW_FILE_SHA256

uv run python scripts/run_qm9_model_selection_v31_medium_profile.py \
  --stage authorize \
  --output /tmp/qm9-v31-medium-profile-v1 \
  --decision-artifact /ABSOLUTE/EXTERNAL/authorization.json \
  --expected-sha256 AUTHORIZATION_FILE_SHA256

uv run python scripts/run_qm9_model_selection_v31_medium_profile.py \
  --stage execute \
  --output /tmp/qm9-v31-medium-profile-v1 \
  --expected-sha256 INTERNAL_AUTHORIZED_FILE_SHA256

uv run python scripts/run_qm9_model_selection_v31_medium_profile.py \
  --stage verify \
  --output /tmp/qm9-v31-medium-profile-v1
```

Every invocation must stop after its named stage. The implementation rejects combined
`prepare+authorize` or `prepare+execute` operation. After preparation, return immutable hashes to an
independent reviewer; after authorization, stop again. Execution needs its own fresh, scope-bound
approval. This protocol text grants neither preparation nor execution.

### Post-execution independent review

An execute exit code of zero and automatic public verify do not accept the result. A second
independent read-only review must reauthenticate the approval consumption, exact start binding,
three reserve/start/complete-or-fail chains, telemetry continuity, resource peaks, checkpoint
replay, MLP curve/anomaly decisions, XGBoost 12-target rounds, final tree, empty process group and
cgroup, zero protected-label reads and every false boundary. Acceptance requires all three trials
complete exactly once, finite outputs, best-state replay, total workload at or below 2,280 seconds,
finalization within 2,400 seconds, peak RSS at or below 3.2 GiB and no unexplained anomaly.

Failure remains valid engineering evidence but cannot be repaired, resumed, reranked or called a
model result. Success establishes only medium CPU feasibility for the exact bound contract. It
does not approve W1 automatically; a separate capacity projection and independent W1 decision are
still mandatory.

The public-fixed lineage remains blocked until the following replay contract is implemented and
covered by negative tests. This section is a requirement, not a claim about current evidence:

- The post-run reviewer identity and reviewer-tool identity must both differ from the runner,
  pre-execution reviewer and authorizer identities/tools. Parse the postreview timestamp as an
  aware UTC instant and require `terminal <= outer-finished <= POSTRUN-created <= postreview-issued`.
  A malformed, backdated or missing timestamp fails closed.
- The evidence archive must be self-contained. The copied external review and authorization must
  match the physical and semantic hashes recorded by `REVIEWED.json`, `AUTHORIZED.json`, the
  authorization-consumption receipt and `POSTRUN.json`; validation must not depend on the original
  absolute decision paths still existing. After extraction, replay the lifecycle, one-use nonce,
  outer evidence, POSTRUN and independent postreview against the archived paths. Reject duplicate
  member names, non-regular-file/non-directory members, extras, missing members, links and unsafe
  lexical paths before extraction.
- XGBoost replay must select only authenticated `xgb_iteration` events for its exact trial and
  reconstruct 12 target streams. Targets must appear exactly in order `0..11`; rounds for each
  target must start at 1, increase contiguously without duplicates, contain a finite evaluation
  value, and agree with the profile's recorded best/terminal round under the frozen early-stop
  semantics. Missing, extra, reordered or nonfinite events reject POSTRUN.
- Ridge must explicitly replay the non-iterative contract: no epoch or XGBoost-iteration events,
  exactly the required fit/forward and prediction stage boundaries in legal order, finite timing,
  one authenticated completion/checkpoint and an explicit `curve_type=non-iterative` with empty
  curves. The absence of an epoch curve is by design, not missing telemetry.

As of 2026-07-21, the archive extractor still performs inventory-only validation, XGBoost/Ridge
raw-event replay is not complete, and postreview identity/chronology separation is not complete.
Therefore these requirements are execution-authorization blockers.

### 2026-07-20 public synthetic outer CLI seam

The explicitly opt-in public fixture completed the real five-stage CLI path
`prepare -> review -> authorize -> execute -> verify` under run ID
`qm9-v31-medium-profile-public-synthetic-v1`. The exact invocation was:

```bash
env -u VIRTUAL_ENV \
  -u QM9_MEDIUM_PUBLIC_SYNTHETIC_SEAM \
  -u RUN_QM9_MEDIUM_INTEGRATION \
  RUN_QM9_MEDIUM_PUBLIC_OUTER_E2E=1 \
  uv run pytest -q \
  tests/qm9/test_model_selection_v31_public_outer_e2e.py::test_public_synthetic_full_outer_cli_e2e
```

Its fixed base was `/tmp/qm9-v31-medium-profile-public-synthetic-v1`; its outer envelope was
`/tmp/qm9-v31-medium-profile-public-synthetic-launch-envelope-v1`. Before execution an independent
review confirmed both paths were absent, the exact command/cwd/code/dependency/protocol hashes,
cgroup-v2 memory controller and user-systemd scope availability, at least 5 GiB free `/tmp`, zero
private inputs and one-use scope. The approval physical SHA-256 was
`a6963d8e81454c6fd4cf8cf3ee26dc668ea57793db1e928aa782d5bab95ece99`.

The test passed once in 8.40 seconds. The outer harness itself reported child elapsed time
1.6309699339908548 seconds, aggregate peak RSS 119,193,600 bytes, verified 4 GiB cgroup limit,
empty cgroup and process group, no OOM, clean status append and no termination action. Every label
counter was zero and all private/protected/ranking/ensemble/promotion/scientific/publication
boundaries remained false. `POSTRUN.json` was `POSTRUN_ACCEPTED` with physical SHA-256
`3183ea508ad6b92d8372c316a88add97924021e5ac40584876150b6b43489cb4`.

`POSTRUN_ACCEPTED` is an automatic disposition, not independent acceptance: its contract keeps
`automatic_disposition_only=true` and `independent_post_review_approved=false`. A separate
read-only postreview replayed the exact wrapper, bound files, repository identity, authorization
nonce/claim/consumption chronology, terminal tree, status/result hashes and resource/label gates.
That postreview issued `GO_PUBLIC_SYNTHETIC_OUTER_E2E_SEAM_ONLY`; its physical SHA-256 was
`60f91332fde9b9a8be8a6d306dcea86f7dd8d9bce633fd0c0d5a025f432eaa47`.

The immutable evidence archive is
`/tmp/qm9-v31-public-outer-e2e-evidence-v1.tar.gz` (mode `0444`, 103,995 bytes), SHA-256
`8294eb3dfb2b871bb924380c52abf631b69b4176bbd9ac38f66e5dea30ce656a`.

This seam used a zero-label, zero-model public synthetic marker workload. It validates only the
top-level CLI, outer harness/cgroup and POSTRUN integration. It did not run the Ridge, XGBoost or
MLP workers and does not establish model or medium-profile feasibility. The existing three-backend
integration reaches the direct sidecar path but not this outer CLI path; the two tests are
complementary, not interchangeable. No real three-backend outer E2E, private preparation, private
execution, W1, ranking, promotion, ensemble, scientific result or publication claim is approved.
The public run ID is consumed and immutable; it must not be deleted, retried, resumed or reused.

## P0-P7 governed plan

### P0 — freeze the failed v3 lineage

Before implementation review:

1. preserve the empty v3 W1 directory and external watchdog evidence read-only;
2. inventory and hash the exact v3 command, config, commit, dependency lock, authenticated inputs,
   hardware and any surviving external resource records;
3. record candidate/fold/epoch, completed fits and computational root cause as `unknown`;
4. prohibit reconstruction of a partial rank or selective continuation; and
5. close the v3 attempt as `REJECT_INCOMPLETE_RUNTIME` with W2-W7 locked.

### P1 — immutable start and append-only state

Before feature loading or the first fit, write and fsync `run_start.json`. It binds:

- schema, run ID, protocol/config/code/dependency hashes and exact command;
- commit, dirty state and patch hash;
- authenticated input, source-row, split, fold and scaffold hashes;
- OS, CPU, RAM, GPU, driver/CUDA/runtime, thread policy, precision and determinism settings;
- complete ordered trial registry and expected candidate/fold/seed keys;
- fit, memory, temperature, disk and soft/hard deadline limits;
- zero protected-label counters, `scientific_result=false` and `publication_ready=false`.

Reserve each trial durably in `ledger/trials.jsonl` before preprocessing or model work. Append start,
progress, completion or failure events; never rewrite history. A nonempty output directory rejects a
new invocation. V3's empty directory is never resumed or backfilled.

The supervisor is the sole owner of ledger and terminal transitions. It derives completion identity
from the immutable plan, prediction contract and durable state, authenticates the checkpoint, and
then appends exactly one completion. A backend-emitted completion, caller-supplied ledger hash or
zero worker exit code cannot authorize success.

Compute one canonical `run_start_sha256` after the immutable start artifact is written. Every event,
ledger row, cache decision, checkpoint and terminal lock must contain that same digest. A mixed
`run_id` or start digest invalidates the chain.

### P2 — exact event and artifact contract

Every JSONL event contains:

```json
{
  "schema_version": "qm9-model-selection-v31-event-v1",
  "sequence": 1,
  "timestamp_utc": "ISO-8601",
  "monotonic_seconds": 0.0,
  "run_id": "...",
  "event": "run_start|heartbeat|trial_reserved|trial_start|stage_timing|epoch_end|target_end|warning|checkpoint|trial_complete|trial_failure|signal|terminal",
  "wave": "W1",
  "candidate_id": null,
  "strategy": null,
  "fold_id": null,
  "training_seed": null,
  "epoch": null,
  "target": null,
  "progress": {"completed_fits": 0, "total_fits": 123},
  "budgets": {"wall_remaining_seconds": 43200, "fit_remaining": 123},
  "label_reads": {"outer_validation": 0, "test": 0},
  "scientific_result": false,
  "publication_ready": false
}
```

Required layout:

```text
run_start.json
telemetry/events.jsonl
telemetry/resources.jsonl
ledger/trials.jsonl
checkpoints/<candidate>/<strategy>/<fold>/<seed>/latest.json
checkpoints/<candidate>/<strategy>/<fold>/<seed>/best.*
failure/failure.json
failure/recursive_why.json
failure/selection_lock.json
final/...
```

`telemetry/`, `ledger/` and `checkpoints/` are inspectable engineering evidence. `failure/` and
`final/` are mutually exclusive terminal states. A truncated final JSONL line is quarantined; prior
valid events remain readable. Only a complete, independently reviewed `final/` can rank candidates.

The implementation must reject event names or fields outside the reviewed schema. Before append
after an interrupted write, validate the complete prefix, move the damaged tail to a hash-recorded
quarantine file, fsync repaired/quarantine files and their parent directories, then continue from
the last valid event hash. No event may be appended after `terminal`.

### P3 — heartbeat, resources and soft/hard deadlines

The v3.1 config must freeze these W1 timing gates before approval:

- heartbeat interval: at most 30 seconds;
- unhealthy heartbeat age: greater than 60 seconds;
- hard wall deadline: 43,200 elapsed seconds;
- internal soft-drain deadline: 42,600 elapsed seconds;
- sidecar escalation: request graceful termination at 43,140 seconds if no terminal state exists;
- absolute kill boundary: 43,200 seconds.

The 600-second drain window is provisional until a forward test measures checkpoint/failure flush
time; it may be increased only before W1 approval and must move the soft deadline earlier, not the
hard boundary later. The trainer stops scheduling at the soft deadline, reaches the next safe
point, flushes telemetry/ledger/checkpoint/failure state and exits incomplete. The sidecar writes an
external signal record even if Python is blocked in a native library.

The independent heartbeat reports process liveness separately from last training progress and last
metric time. The sidecar continuously enforces heartbeat age, RSS, GPU allocated/reserved/peak
memory, GPU temperature, disk floor and hard wall time. Missing sensors are `unavailable`, never
zero. These sensors observe the child or complete child process group externally, not the sidecar
itself; a required unavailable sensor fails closed. Every stop preserves zero outer/test label-read
counters.

### P4 — real-backend profiling and memory gate

No optimization is selected before stage profiling. Run bounded train-only profiling using the
real feature recipe, backend, optimizer, precision, batch size, validation and checkpoint paths.
Measure separately:

- source slicing and fold/recipe transformation;
- sparse-to-dense conversion and cache lookup;
- tensor construction and host-to-device transfer;
- forward, backward and optimizer;
- validation/prediction and per-target metric computation;
- checkpoint copy, hash and write;
- each XGBoost target estimator and Ridge fit/predict;
- CPU time, RSS, GPU utilization/temperature/allocated/reserved/peak memory, I/O and disk.

The memory worksheet separately accounts for raw/cached train and tune inputs, targets, parameters,
gradients, Adam states, activations, dense temporaries/workspaces, CUDA reservation, predictions,
Python copies, telemetry buffers and two checkpoint generations. The approved ceiling must retain
at least 20% measured headroom.

Report estimated and measured baseline/incremental/peak values separately. Reset GPU peak counters
around each option, record unavailable sensors explicitly, and reconcile estimate versus measure.
Feature-cache byte count alone is not a memory model.

Estimate W1 runtime from representative real units by family/recipe/complexity tier. The
conservative projection is the sum of profiled p95-equivalent trial durations, transformations and
checkpoint costs plus 20% orchestration margin. It must be at most 80% of the 42,600-second soft
deadline: **34,080 seconds**. Otherwise W1 authorization is refused and v3.1 must be amended before
private execution.

The supervisor loads the exact hash-bound memory and runtime artifacts and device identity before
creating executable run state. Any missing/mismatched artifact or `approved=false` decision refuses
startup and trial reservation.

### P5 — execution-equivalence gate

Compare the unoptimized reference path with every proposed cache, reuse or scheduling path on the
same train-only rows, folds, candidates and seeds. Require:

- identical row, group, batch and target order plus identical trial keys;
- bitwise-identical sparse indices/indptr and deterministic transformed feature values;
- identical fold-local scaler statistics and provenance hashes;
- exact Ridge/XGBoost best-iteration and prediction equality on the reviewed environment;
- exact first-batch MLP tensors and target standardization;
- MLP train/tune/per-target curves and predictions within `rtol=1e-6`, `atol=1e-7`;
- identical best/restored epoch, early-stop reason and anomaly decisions; and
- identical aggregate metrics, promotion set and ranking order.

Any failure is a scientific change, not an execution-only optimization. The optimized path cannot
enter a long run until its equivalence artifact and hashes receive independent approval.

Cache identity must bind the authenticated source rows, strategy/fold/recipe transformation and
semantic hashes of the actual transformed train/tune matrices. Placeholder identities are
forbidden. The full profile and deterministic option decision are hash-frozen in `run_start.json`;
the backend rejects any caller-selected option that differs.

Equivalence approval is device-specific. CPU evidence cannot approve CUDA, and a skipped hardware
test is missing evidence. Each intended device records deterministic settings and passes the exact
reference tolerances.

### P6 — real-backend observability and failure forward tests

The instrumented smoke must exercise real Ridge, XGBoost and Torch paths. It must expose live:

- candidate, strategy, fold, seed, epoch or XGBoost target and completed/total fits;
- standardized training loss, tune MNMAE and 12 per-target metrics;
- learning rate, gradient/parameter norms, best/restored epoch, stale count and stop reason;
- all P4 stage timings, resources, ETA method and budget burn-down;
- validation-increase/anomaly marks and label-read counters.

Forward tests must cover: a >65-second unit with independent heartbeat, SIGINT/SIGTERM during MLP
and a native XGBoost call, soft/hard deadline, stale heartbeat, disk/RSS/GPU-memory/temperature
limits, nonfinite feature/scaler/target/loss/gradient/parameter/prediction, crash after trial reserve,
truncated JSONL recovery, same-directory rerun, provenance tamper, label-access tripwire and
partial-ranking tripwire. All abnormal cases end `complete=false`, retain diagnostics, and produce
no promotion.

Each Ridge, XGBoost and Torch smoke must traverse the same supervisor: immutable start, durable
trial reservation, independent heartbeat/sidecar, continuous guard, checkpoint, completed or
failed ledger event, and immutable terminal lock. A recording double or surrogate does not satisfy
this gate.

The integration smoke must prove the seams, not merely invoke the components in one test module.
For a multi-trial plan, a successful worker returns to the scheduling supervisor; it cannot finalize
the run per trial. `final/` is created only after every planned trial has exactly one authenticated
checkpoint and completion transition and strict replay succeeds.

### P7 — independent approval and terminal review

Separate executor and reviewer identities are mandatory. Before private W1, the reviewer must bind:

- v3.1 protocol/config/code/dependency/input/hardware hashes;
- exact command and fresh permanent output directory;
- P0 failure inventory;
- P4 memory/runtime feasibility evidence;
- P5 equivalence evidence;
- P6 real-backend smoke and every required failure test;
- zero outer/test reads and W1-only scope.

Current decision is `NO_W1_APPROVAL`. After an authorized W1, a separate reviewer verifies all 123
trial keys, 41 complete candidates, ledger/event continuity, folds, curves, anomalies, budgets,
source provenance, promotion, terminal lock and zero protected-label reads. W2 cannot start under a
W1 execution approval.

Completed-only replay must reconcile exactly one immutable completed artifact for every planned
trial key with exactly one completed ledger event and the same start/config/input/cache/environment
bindings. Exact prediction shape/order/status and semantic metadata hashes are mandatory. Extras,
duplicates, missing keys, partial runs, terminal failure, or metadata tamper make replay ineligible.

## Recursive Why v3.1

Each abnormal event creates a root record with child hypotheses:

```yaml
schema_version: qm9-model-training-recursive-why-v31
issue_id: WHY-YYYYMMDD-NNN
lineage: qm9-model-selection-v3.1
gate: P0-P7
fact:
  observation: "direct observation without interpretation"
  evidence_hashes: []
  missing_evidence: []
hypotheses:
  - id: H1
    statement: "competing mechanism"
    status: supported | falsified | unknown
    falsification_test: "smallest controlled check"
    evidence_hashes: []
root_causes:
  governance: {status: supported | unknown, statement: ""}
  computational: {status: supported | unknown, statement: ""}
containment: []
correction:
  proposed_change: ""
  execution_only: false
  scientific_contract_change: false
  fresh_output_required: true
recurrence_prevention:
  forward_test_ids: []
label_reads: {outer_validation: 0, test: 0}
review:
  executor: ""
  independent_reviewer: ""
  decision: pending | approve | reject | defer_for_evidence
  bound_hashes: []
```

Never force the computational root cause to match the governance cause. Unknown evidence remains
`unknown`. A correction cannot be called execution-only until P5 passes.

## Claims and current status

V3.1 now has a completed and independently verified fixed-v5 bounded CPU synthetic engineering
smoke and the completed public synthetic outer CLI seam recorded above. The fixed-v5 workload
establishes lifecycle, subprocess, telemetry, checkpoint and replay seams; the public outer seam
establishes the top-level CLI, outer cgroup and POSTRUN seam. Neither establishes real fixed-train
capacity or a scientific model comparison. The real medium train-only lineage has not been
prepared, reviewed, authorized or executed. No v3.1 private fit, real three-backend outer E2E or W1
approval exists. The only scientific-result statement remains: **the v3 W1 attempt was incomplete
and produced no v3 model-selection result.**

## Public fixed three-backend preflight lineage

Before private profiling, v3.1 requires one public-synthetic run using the unchanged production
Ridge, XGBoost and MLP candidates. Its identity is
`qm9-v31-public-fixed-three-backend-cpu-v1`; its purpose is engineering evidence only.

The bundle is frozen before review. It has at least 8,192 rows selected as complete unequal
identity groups, 2,048 deterministic sparse binary ECFP-like columns, 17 dense descriptors and 12
finite feature-related targets with different scales. Validation reconstructs groups, split,
features, targets and the unique `PlannedTrial` payloads from constants. A self-consistent rewritten
manifest is rejected. No QM9, private, protected, outer-validation or test target is available.

External reviewer and authorizer identities must be distinct from the runner and each other. The
dedicated `public-fixed-v1` profile binds the run/purpose, wrapper argv, authorization, bundle,
adapter, CLI, worker, supervisor, harness, protocol, dependency lock and repository code state.
All workers are direct descendants of one authoritative 4-GiB outer cgroup.

Atomic heartbeat snapshots expose model-progress sequence, stage, target/round or epoch, metrics
and label counters. The sidecar separately observes heartbeat age, progress age, RSS and disk.
The current implementation uses one 20-second stale threshold for both heartbeat and model
progress, so a healthy long native call can be terminated as `STALE_MODEL_PROGRESS`. Execution is
blocked until liveness staleness and per-family/per-stage progress staleness are separate reviewed
values and a greater-than-65-second blocking-stage test proves that a continuing independent
heartbeat does not fabricate metric progress or cause a false stop.

POSTRUN currently replays the three ledgers, worker profiles, checkpoints and the MLP
curve/anomaly result, then binds the outer cgroup peak. XGBoost/Ridge raw-event replay and
self-contained archive replay remain pending under the contract above. Automatic acceptance
remains `scientific_result=false` and `publication_ready=false`; independent post-review is
required before evidence archival.

This preflight is not model selection, W1, capacity approval or a scientific result. It currently
awaits second independent pre-execution review; no public three-backend model process has run.
