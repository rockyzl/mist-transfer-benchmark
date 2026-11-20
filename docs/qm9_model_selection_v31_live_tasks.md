# QM9 model selection v3.1 live tasks

Status date: 2026-07-20

Current decision: `PUBLIC_SYNTHETIC_OUTER_E2E_ACCEPTED_ENGINEERING_ONLY; PRODUCTION/PRIVATE NO-GO`

Engineering status: `PUBLIC OUTER SEAM ACCEPTED — NO MODELS/PRIVATE W1/GPU/RANKING`

Public-fixed three-backend status on 2026-07-21: `NO-GO`. Remaining hard gates are self-contained
archive/lifecycle replay, independent postreview identity and chronology, authenticated XGBoost
12-target plus Ridge non-iterative event replay, and separate heartbeat-versus-progress stale
policies with a long-blocking-stage seam test. The focused component suite does not authorize the
run, and this document grants no execution authority.

## Evidence boundary

- V3 W1 at commit `8a52139` ended incomplete after 43,201 seconds and left an empty preserved
  output directory.
- It produced no candidate rank, promotion, durable fit count or scientific result.
- Missing observability is a supported governance cause; the computational bottleneck is unknown.
- No outer-validation or test labels were read by that attempt.
- The synthetic integration-v3 seam and the fixed v5 lineage reached accepted engineering-only
  success terminals. Fixed v3 failed its preterminal contract, and fixed v4 never executed a
  worker.
- Formal W1, private-label access and GPU execution have not run.
- V1-v4 evidence remains immutable. The fixed v5 lineage at `/tmp/qm9-v31-bounded-smoke-v5` was
  executed and independently reauthenticated; it is now immutable engineering evidence rather
  than authority for another run. V4 is explicitly read-only and rejected by the runner.
- The medium runner and public outer-seam test are implemented. The separately governed production
  profile in the [medium profiling brief](qm9_model_selection_v31_medium_profile_brief.md) has not
  been authorized, prepared or executed.

## 2026-07-20 public synthetic outer E2E acceptance

One separately authorized, explicitly opt-in five-stage public synthetic invocation completed with
`1 passed in 8.40s`. Its preserved lineage then passed an independent read-only post-run review and
is recorded as `PUBLIC_SYNTHETIC_OUTER_E2E_ACCEPTED_ENGINEERING_ONLY`.

The invocation exercised the real `prepare -> review -> authorize -> execute -> verify` CLI seam,
bounded outer wrapper, cgroup collection and exact POSTRUN replay. POSTRUN was accepted with no
reasons; outer elapsed time was 1.631 seconds, aggregate cgroup peak was 119,193,600 bytes, the
child exited zero, and the process group and cgroup were empty. All six label-read counters were
zero and all private/protected/ranking/ensemble/promotion/W1/scientific/publication boundaries
remained false.

The workload was deliberately a zero-label, zero-model public synthetic marker. It did not run
Ridge, XGBoost or MLP, did not produce predictions or metrics, and supplies no model, ranking,
feasibility, capacity or scientific evidence. The real three-backend medium integration has not
run. Production/private preparation and execution remain `NO-GO` and require a fresh independent
decision and one-use authorization.

The durable evidence archive is `/tmp/qm9-v31-public-outer-e2e-evidence-v1.tar.gz`, SHA-256
`8294eb3dfb2b871bb924380c52abf631b69b4176bbd9ac38f66e5dea30ce656a`. The independent
post-review record is `/tmp/qm9_v31_public_outer_e2e_postreview_v1.json`, SHA-256
`60f91332fde9b9a8be8a6d306dcea86f7dd8d9bce633fd0c0d5a025f432eaa47`. The completed
public lineage, envelope and decision artifacts are immutable engineering evidence, not authority
for a retry or a production run.

## 2026-07-19 v5 identity migration

The bounded runner, lifecycle schemas, default runner identity and exact-base guard were migrated
to v5 before execution. That migration itself was code/test work only. V5 was subsequently
prepared, independently reviewed, authorized once, executed, read-only verified and accepted as an
engineering-only smoke, as recorded below. Historical v1-v4 artifacts were not modified.

## P0-P7 status

- [ ] **P0 failure freeze:** inventory/hash the v3 directory, command, watchdog evidence, inputs,
  config, dependency and hardware provenance; record unknown fields explicitly.
- [x] **P1 start/ledger:** implement immutable `run_start.json`, append-only trial reservations and
  fresh-output enforcement.
- [x] **P2 telemetry/artifacts:** implement exact event schema and separate telemetry, ledger,
  checkpoints, failure and final namespaces.
- [ ] **P3 safety:** implement independent <=30-second heartbeat, sidecar, continuous resource
  gates, 42,600-second soft drain, 43,140-second escalation and 43,200-second hard boundary.
- [ ] **P4 profile/capacity:** run bounded real-backend stage and memory profiles; require >=20%
  memory headroom and <=34,080-second conservative W1 projection.
- [ ] **P5 equivalence:** prove each cache/reuse/scheduling optimization meets the frozen numerical
  and decision-equivalence contract.
- [ ] **P6 forward tests:** complete real Ridge/XGBoost/Torch observability smoke and all abnormal
  termination, resource, nonfinite, provenance, label and partial-ranking tests.
- [ ] **P7 approval:** independent reviewer binds exact hashes/command/output and issues a W1-only
  decision.

## Independent QA remediation tracker

- [x] Bind the event stream and trial ledger to one immutable `run_start_sha256`; reject mixed run
  IDs. Checkpoint/supervisor completion integration remains tracked separately below.
- [x] Enforce the reviewed event names/schema and immutable terminal event.
- [x] Quarantine/hash a truncated JSONL tail before repaired append.
- [x] Fsync parent directories after every atomic create/replace covered by the v3.1 tests.
- [x] Validate checkpoint semantic metadata and exact prediction shape/status/trial identity.
- [x] Reconcile replay against exact plan keys, completed ledger events and successful terminal;
  repeat strict replay whenever ranking is requested after finalization.
- [x] Bind cache identity to actual transformed matrix hashes and authenticated row/fold/recipe.
- [x] Freeze the profile and deterministic cache choice; reject arbitrary MLP backend option.
- [x] Implement the complete memory/runtime worksheet, immutable decision artifact, device/hash
  binding and 34,080-second startup/reservation refusal. Actual profile values remain P4 evidence.
- [x] Add a true child-process-group sidecar plus RSS/disk/sensor-unavailable, signal/escalation and
  absolute-hard-stop subprocess tests.
- [ ] Run governed real smoke for Ridge, XGBoost and Torch on every intended hardware path.
- [x] Treat skipped CUDA/other hardware as missing evidence, never approval.

## 2026-07-19 remediation re-review

The focused v3.1 suite passes (39 tests), including the CUDA cache-equivalence test on the reviewed
CUDA device. Ruff, `git diff --check`, and the generic governance-skill validator also pass. The
event-chain, truncation-recovery, parent-fsync, checkpoint-tamper, completion-schema, cache-content
identity, frozen cache-choice, CPU equivalence and CUDA equivalence remediations are accepted as
component evidence.

The re-review remains `REQUEST_CHANGES` because the components are not yet one governed execution:

- `RunSupervisor` does not construct `CheckpointStore`, derive `CompletionContract` from its frozen
  plan and durable ledger, require a completed checkpoint before ledger completion, or call strict
  replay before authorizing the terminal ranking. A backend can accept a caller-invented ledger hash
  and emit `trial_complete` while the supervisor ledger is still only `start`.
- `ExternalWorkerSidecar` enforces heartbeat and elapsed-time deadlines, but does not independently
  sample/enforce worker RSS, GPU allocated/reserved memory, GPU temperature, or disk floor. Its
  heartbeat events contain no resource sample.
- The memory worksheet and runtime projection functions exist, but their `approved` decisions are
  not bound into `RunSupervisor`; a projection with `approved=false` can still reserve work.
- The all-family test runs real Ridge/XGBoost/MLP functions with component hooks, but not through one
  supervisor/sidecar/start-ledger-checkpoint-terminal path. The current sidecar success test runs a
  trivial Python worker rather than a real model backend.

Integration work now in progress must add:

- [x] one supervisor-owned reserve/start/complete/fail/terminal state machine;
- [x] supervisor-derived completion/replay identities from the exact plan and actual durable ledger;
- [x] authenticated checkpoint required after child exit and before exactly one completion event;
- [x] hash/device-bound memory and runtime startup refusal gates;
- [x] external child/process-group RSS/disk enforcement plus fail-closed unavailable GPU sensors;
- [x] multi-trial scheduling that finalizes only after all planned trials reconcile; and
- [x] one real Ridge/XGBoost/MLP synthetic/public supervisor seam test plus a real Ridge subprocess
  sidecar/checkpoint/replay/terminal test.

## 2026-07-19 third independent review

Decision: `APPROVE_BOUNDED_TRAIN_ONLY_PROFILE_SMOKE`.

The third review verified one completion owner, supervisor-derived completion identity, checkpoint
authentication before completion, strict replay before final and at every later ranking request,
feasibility refusal/recheck, no early multi-trial final, external RSS/disk/GPU-sensor-unavailable
failure, signal/hard kill, all-family real supervisor execution, and a real Ridge subprocess worker.
The focused suite passed 49 tests; the full repository passed 231 tests with two optional dependency
skips. Ruff, `git diff --check`, and skill validation passed.

The approval is limited to one fresh, bounded **CPU-only** profiling smoke:

- public or train-only rows only; zero fixed outer-validation/test-label reads;
- predeclared <=4096-row/<=4096-feature worker fixtures and short per-trial deadlines;
- one real subprocess-sidecar trial for each Ridge, XGBoost and MLP so the smoke itself closes the
  remaining XGBoost/MLP subprocess evidence gap;
- immutable approved feasibility/device/input/code/config/profile bindings before reservation;
- no candidate ranking, promotion, ensemble decision, scientific claim, or publication output; and
- preserve the complete output for another independent review before any private W1 decision.

Residual limitations: XGBoost and MLP have passed the real supervisor path but not yet their own
subprocess-sidecar test. The external GPU measurement maps NVML process-used memory to both
allocated and reserved fields; it is conservative containment evidence, not distinct PyTorch
allocated/reserved telemetry. Therefore CUDA/GPU profiling, private W1, W2-W7, ranking and promotion
remain prohibited.

Historical second-review restriction is superseded only by the bounded CPU profile-smoke approval
above. It does not authorize private W1.

## 2026-07-19 bounded-smoke execution pre-review

Decision: `APPROVE_EXECUTION` for exactly one fresh run
`qm9-v31-bounded-smoke-cpu-v1` under the three-artifact
`bounded_profile_only` bootstrap contract. The independent decision is preserved at
`/tmp/qm9_v31_bounded_smoke_prereview.md` with physical SHA-256
`4425b6627f2e292e512b9206eaf51fc643b4f3495721f498bd0037f54cdd8976`.

The final focused review passed 11 non-integration profile-smoke tests and 49 core v3.1 tests;
Ruff, `git diff --check` and generic training-governance skill validation passed. The fixed output
path remained absent. The implementation now binds staged/unstaged/tracked/untracked code,
reviewed preflight bytes and types, external reviewer hashes, authorization chronology, atomic
one-use claims, initial and per-trial resource evidence, reviewed required memory, the physical and
semantic run start, durable ledger replay, checkpoints and a profile-only terminal.

Execution remains conditional on `prepare` exiting before any run output, a separately created
external `independent-review.json`, a fresh short-lived authorization, exact out-of-band review/auth
hashes and reviewer identity, CPU/synthetic resource gates and preservation of the complete output
for post-smoke review. Private W1, GPU, retry/resume, ranking, promotion, ensemble selection,
scientific claims and publication remain prohibited.

## 2026-07-19 v1 prepare incident

The approved v1 prepare atomically claimed `/tmp/qm9-v31-bounded-smoke-v1` and then failed because
the runner searched for `uv.lock` at a hard-coded `parents[4]`; the repository lock is at
`parents[3]`. The preserved directory contains partial preflight/cache artifacts only—no run,
attempt, review, authorization, worker, checkpoint, ledger, terminal or protected-label access.

V1 is permanently burned and must not be deleted, reused, resumed or authorized. The independent
incident record is `/tmp/qm9_v31_bounded_smoke_v1_prepare_incident.md`. A v2 lineage may be reviewed
only after semantic repository-root/lock discovery happens before claim, every post-claim exception
durably records a terminal preparation failure, focused tests pass and a new v2 plan/hash receives
a separate independent approval.

## 2026-07-19 v2 remediation implementation

Status: `IMPLEMENTED_NOT_APPROVED`; no prepare or smoke was run.

The v2 runner uses run ID `qm9-v31-bounded-smoke-cpu-v2` and the fresh fixed base
`/tmp/qm9-v31-bounded-smoke-v2`. Before claiming that base it resolves the Git root semantically,
selects exactly one package/marker ancestor, checks the parsed project identity and matching Git
top-level, verifies regular non-symlink `pyproject.toml` and `uv.lock` files, and freezes full lstat,
path, identity and code-state provenance. The base lifecycle is now exact
`prepare-claim.json -> PREPARE_COMPLETE.json`; failures bind the attempt/claim, stable failure code,
last checkpoint and partial lstat tree in `PREPARE_FAILED.json`. Review, authorization and execution
require success, absence of failure and full repository provenance replay. The generic run-start is
now an explicit v2 schema, and v1 requests fail with `V1_INCIDENT_INCOMPLETE`.

The success terminal also freezes the pre-complete exact base lstat tree, not only `preflight/`.
Only the complete terminal and staged review/authorization files may appear later; `attempt/` and
`run/` are admitted only during the explicit execute transition. Unexpected base files and symlinks
fail closed. Failure absence uses `lstat`/lexists semantics, including dangling symlinks. A failed
claim write produces an exact `PREPARE_CLAIM_FAILED.json`; if any failure-terminal write itself
fails, the existing base remains permanently burned and reuse still refuses.

Safe verification passed: Ruff; 27 non-integration profile-smoke tests with one integration test
deselected; 15 supervisor tests; and `git diff --check`. Tests cover wrong repository root, missing
lock, symlink lock, ambiguous nested markers and wrong project identity without claiming the base;
they also cover claim/success ordering, missing success, failure blocking, provenance tamper,
failure-artifact durability and non-reuse. The burned v1 directory was not read, changed or reused.
Before v2 can run, another read-only audit and independent hash-bound execution decision are
required; the v1 approval cannot authorize v2.

## 2026-07-19 v3 post-run reauthentication implementation and incident

Status: `REJECT_INCOMPLETE_PRETERMINAL_FINAL_TREE_CONTRACT`; v3 is preserved and permanently
nonresumable.

V2 disposition is `REJECT_POST_RUN_REAUTH_GAP`. Its immutable model, checkpoint, ledger, telemetry
and curve files remain useful diagnostic engineering evidence, but the lineage is not an accepted
governed smoke because the public post-run reauthentication path rejected its own expected final
summary/proposal outputs. V2 grants no W1, general-training, ranking, promotion, ensemble,
scientific or publication permission and must remain byte-for-byte unchanged.

V3 moved summary/proposal creation before the bounded terminal and implemented a final exact-tree
success lock. Its authorized bounded execution completed Ridge, XGBoost and MLP, produced nine
valid reserve/start/complete ledger rows, ended telemetry at sequence 14 with `trial_complete`, and
strictly replayed three completed checkpoints. Peak observed worker RSS was 653,422,592 bytes and
minimum free disk was 737,287,778,304 bytes. Protected-label reads remained zero and every
scientific/ranking boundary remained false.

The run then failed at the preterminal exact-tree assertion. `_expected_final_path_types()` included
both `run/final/` and `run/final/manifest.json`; the preterminal filter removed only the manifest.
Because `finalize_profile_success()` had not yet created either entry, the required parent directory
was missing. Independently, the contract used plain `canonical_sha256(identity)` and expected cache
path `identity-71f177...json`; the production schema-aware `cache_identity_sha256()` created
`identity-a7a879...json`. The preterminal check therefore had both an expected-only parent and a
cache hash-domain mismatch and deterministically failed. No `run_complete`, success terminal,
failure terminal or `POSTRUN_MANIFEST.json` exists. The summary and `NOT_APPROVED` proposal are
diagnostic only; they do not turn the lineage into a successful smoke. The missing durable failure
terminal for this post-claim exception is an additional governance gap.

V3 must remain byte-for-byte preserved: no retry, resume, repair, backfill or verification claim.
A fresh successor must define exact before/after path-type contracts with consistent parent
directory treatment, derive cache paths through the producer's schema-aware identity helper, and
add production-path tests starting from absent future directories. It must also catch
finalization-seam exceptions and durably record a mutually exclusive failure terminal.

V1 and v2 paths are explicitly refused for mutation. V2 remains valid only as frozen historical
evidence under its original code identity; this v3 code cannot retroactively become its executor.

## 2026-07-19 v4 authorization and integration-seam gate

Status: `AUTHORIZED_NOT_EXECUTED`; current execute decision:
`NO-GO_MONITOR_CONTRACT_UNBOUND`.

The fresh v4 lineage completed prepare, independent review and its bounded authorization stage.
The authorization hard stop was respected: no v4 attempt/run directory, worker, checkpoint,
terminal, ranking or scientific result was created. Authorization is therefore provenance and
scope evidence only; it is not permission to execute.

A separate integration-only plan and initial approval were created for the exact opt-in pytest
node that uses a fresh pytest `tmp_path`, CPU-only synthetic 96-by-64 data and one Ridge, XGBoost
and MLP subprocess worker. That initial approval did **not** become executable permission. Review
found that the proposed invocation did not bind an external harness that continuously enforces
both approved limits over the aggregate pytest process tree:

- 60-second wall-clock hard limit from pytest process start through complete process-tree exit; and
- 1.5-GiB aggregate RSS ceiling across pytest and every descendant, with process-tree termination
  and durable evidence on breach.

The integration command was not run. Its state is `NO-GO_MONITOR_CONTRACT_UNBOUND`, not a failed
model smoke and not scientific evidence. The team is implementing a bounded wrapper/harness that
must bind the exact command and code state, monitor the complete process tree, enforce both hard
limits, report peak aggregate RSS and wall time, and preserve fresh-`tmp_path` isolation. The
harness and its tests require a new independent review and a fresh, unexpired integration-only
approval before one invocation. It must not read from or write to the fixed v4 lineage.

Formal W1/general training, private or protected labels, outer/test labels, CUDA/GPU, candidate
ranking, promotion, ensemble selection and scientific/publication claims have not run and remain
prohibited. Passing the future bounded integration seam would provide engineering evidence only;
it would still require a later independent execute-only review before fixed-v4 execution.

## 2026-07-19 bounded-harness second independent review

Decision: `SECOND_REVIEW_REQUEST_CHANGES`; execute status remains
`NO-GO_MONITOR_CONTRACT_UNBOUND`.

Focused harness unit tests and the explicitly non-integration profile-smoke tests passed, but they
do not yet prove the required fail-closed process-tree envelope. The second review identified eight
blocking contract classes:

1. the configured hard wall currently begins graceful termination at the limit, so process-tree
   exit can occur after the approved 60-second boundary;
2. monitoring is anchored to the leader PID and can lose reparented or still-running descendants
   after the leader exits;
3. termination grace waits for the leader rather than proving that the complete process group has
   exited, so a descendant that ignores `SIGTERM` may escape escalation;
4. unavailable or failed RSS sampling can be represented as zero usage instead of a fail-closed
   `SENSOR_UNAVAILABLE` outcome;
5. the `<=1`-second observable-status cadence is not a strict elapsed-time guarantee and is not
   maintained through the termination grace period;
6. the immutable result records the harness-file hash but does not independently bind and verify
   the reviewed repository code-state and working directory before spawn;
7. termination evidence lacks sent-signal timestamps, escalation history, survivor identities and
   a final process-group-empty assertion; and
8. spawn, sampling or telemetry-write exceptions can escape before an immutable failure result is
   durably created.

A targeted fix round is in progress. It must add forward unit coverage for leader-first exit,
`SIGTERM`-ignoring descendants, unavailable sensors, strict status cadence and hard deadline,
repository/cwd binding, durable monitor failures and final no-survivor evidence. The repaired
code-state then requires another independent review and a fresh integration-only approval.

The integration test, fixed-v4 execute and formal W1 have still not run. Private/protected labels,
outer/test labels and GPU/CUDA also remain untouched and prohibited.

## 2026-07-19 integration-v2 diagnostic and terminal-observability remediation

Status: inner diagnostic `PASS`; outer decision
`REJECT_OBSERVABILITY_TERMINAL_CONTRADICTION`. The one-use integration-v2 authorization was
consumed by its single invocation and cannot authorize a retry.

The bounded integration-v2 invocation completed the CPU-only synthetic Ridge, XGBoost and MLP
diagnostic. It produced three valid 24-by-12 checkpoints, exactly nine ledger rows
(reserve/start/complete for each model), replayed all three completions, and closed with
`PROFILE_COMPLETE`. Telemetry was contiguous through sequence 15 and ended at `run_complete`.
Inner/outer/test labels were unavailable to the workers, protected-label reads were zero, and every
ranking, promotion, scientific-result and publication boundary remained false. The immutable
bounded result SHA-256 is
`b7b1a1f8c49d1c2acb89e7556e863126f16d36c168d2bbbd5e21de2cd8715e05`; the status JSONL
SHA-256 is `a40f4617ffcb929960e54867021a3e6e5ae269e8da2477ac95bbc7d7f91dd463`.
Elapsed time was 11.1914 seconds, peak aggregate RSS was 857,313,280 bytes against the
1,610,612,736-byte limit, the maximum observed status gap was 0.561905 seconds, and both the
process group and cgroup were empty at collection.

The outer evidence was nevertheless rejected because the then-current wrapper appended
contradictory terminal states (`failed/CHILD_EXIT` followed by `complete/CHILD_EXIT`) and recorded
spurious TERM/KILL cleanup actions after a verified clean exit with no survivors. Thus the model
path passed only as an engineering diagnostic; the invocation is not an accepted governed smoke
and supplies no scientific result or execution permission.

The terminal-observability repair is implemented and its current code state has passed an
independent read-only QA: all 21 bounded-process tests pass, including collected/inactive systemd
success, active-cgroup sensor-loss fail-closed behavior and the prior cgroup/escape/deadline cases;
Ruff also passes. A real short-command probe emitted only
`running -> running -> finalizing -> complete`, exactly one terminal, with no TERM/KILL timestamps
or cleanup-action evidence. This validates the repair locally but does not retroactively repair
integration v2 and does not authorize another integration invocation. A fresh code-state binding,
independent review and new one-use authorization are required for any successor integration run;
terminal remediation therefore remains in progress at the governed integration level.

The earlier fixed-v4 authorization expired unconsumed: no v4 execute occurred. Formal W1/general
training, private or protected labels, outer/test labels and CUDA/GPU execution still have not run.

## 2026-07-19 integration-v3 post-run observability audit

Decision: `ACCEPT_ENGINEERING_INTEGRATION_SEAM`; this is not a scientific result and does not
authorize execution of the fixed v4 lineage.

The single-use invocation was bound to plan SHA-256
`ac9e8d3df6e51675d0f01d228cebaf002af3f45df5cd95756076df75e683879c` and approval SHA-256
`8fb70851f28bb1f60123c906a094fae6f14e55926690490ec5251bf4b75f456d`. Both authorization
artifacts remained mode `0444`, and the wrapper started at `2026-07-19T23:08:39.352716Z`, before
the approval expiry. The immutable wrapper result SHA-256 is
`892aafd42ad0626e2c188728d9f6b7d6591f48b2540fa0b3dd660641497124b7`; its bound status SHA-256
is `ffb187e4ee75df5767699e27f41075725b714f750e8321e3c7eca5a14f2f53a0`.

The wrapper completed with `CHILD_EXIT`, exit code zero, in 11.1561 seconds. Peak aggregate RSS was
857,477,120 bytes under the 1,610,612,736-byte cgroup limit, all memory-event counters were zero,
and both the cgroup and process group were empty. The 24 status records were monotonic
(`running -> finalizing -> complete`), contained exactly one terminal, and had a maximum elapsed
gap of 0.520176 seconds. No TERM or KILL timestamp, survivor or cleanup action was recorded. All
command, harness, code-state, cwd and eight bound-file provenance comparisons matched.

The inner synthetic seam completed Ridge, XGBoost and MLP. Each worker result was `COMPLETE` and
each completed checkpoint contained one 24-by-12 prediction array with a matching physical hash.
The ledger contained exactly nine ordered reserve/start/complete entries, checkpoint replay count
was three, and telemetry sequences 1 through 15 formed an unbroken hash chain ending in
`run_complete`. The profile-only terminal was complete, `POSTRUN_MANIFEST.json` reported
`PROFILE_COMPLETE` with its success lock and both streams closed, and all terminal-to-worker,
profile, summary, proposal and authorization physical hashes matched. Protected-label reads were
zero; every outer/test/private-data, ranking, promotion, ensemble, W1, scientific-result and
publication permission remained false. The post-profile proposal remained `NOT_APPROVED` with
`INSUFFICIENT_OBSERVATIONS`, as required for this one-observation engineering smoke.

The integration-v3 authorization is consumed and cannot be retried. The next governed step is a
new, fresh v5 fixed lineage: create and freeze its plan and evidence root, independently review its
exact code/provenance/resource contract, then seek a new execute-only authorization. This accepted
integration seam alone grants no permission to reuse or mutate fixed v4, inspect private or
protected labels, run GPU/CUDA, start formal W1, rank candidates or make scientific claims.

## 2026-07-19 fixed-v5 bounded execute post-run audit

Decision: `ACCEPT_FIXED_V5_ENGINEERING_SMOKE`; public verify has not yet run and scientific use
remains prohibited.

The one-use corrected execute ran through the external cgroup harness. Its immutable result
SHA-256 is `d73b40def4c95e77b05f163633b619d6aa8aab60ca5d01d2371cb4da23cab492`; status JSONL
SHA-256 is `92e19ecf2a9e1758e6ef9c2be2394a7ef5182fc17997f2a383cc9b5e4c970e58`.
The wrapper returned `CHILD_EXIT`/zero in 9.14034 seconds, observed peak aggregate RSS of
576,638,976 bytes below the 1,610,612,736-byte limit, and recorded no memory events. Its 20 status
records were monotonic, had a maximum gap of 0.525894 seconds and exactly one `complete` terminal.
Both process group and cgroup were empty; no TERM/KILL, survivor or cleanup action was recorded;
all command, harness, code-state, cwd and bound-file comparisons matched.

The fixed v5 lineage completed one Ridge, XGBoost and MLP worker. Each result and completed
checkpoint is `COMPLETE`, has a 24-by-12 prediction array and matches its physical and prediction
hash. The ledger contains exactly nine reserve/start/complete rows; the event stream has contiguous
sequences 1 through 15, an intact hash chain and final `run_complete` payload equal to the v5
terminal. Checkpoint replay count is three. MLP train standardized MSE decreased from 1.08710 to
1.06698 and inner-tune MNMAE decreased from 0.80623 to 0.80334; its anomaly status is `normal`,
with zero consecutive validation increases, no warnings and no anomalies.

The terminal is complete, profile-only and schema v5. The exact pre-POSTRUN tree matches the
embedded inventory SHA-256
`9dac6ea70cd990a60748a35f6ab4d8ef9281c953aceb3db879155187713063ac`.
`POSTRUN_MANIFEST.json` has SHA-256
`4d19d937872fac3b72bf0ef73b475984e56d85df16f3f0188d12c207a3f66fe6`, status
`PROFILE_COMPLETE`, `success_lock=true`, and both ledger and telemetry streams closed. All
terminal/POSTRUN hashes for run start, ledger, events, summary, proposal, receipt, worker results
and profiles reconcile. Protected-label reads are zero and all 291 observed private/test,
ranking, promotion, ensemble, W1/general-training, scientific-result and publication boundary
fields are false. The proposal remains `NOT_APPROVED/INSUFFICIENT_OBSERVATIONS`, as required for
one engineering observation per family.

The execute authorization is consumed and cannot be retried. The next gate is one separately
approved, read-only `--stage verify` invocation binding the final artifact hashes and exact
review/authorization identities. Verify may emit only an external report; it may not alter the
fixed lineage. A later independent post-verify review is still required before any next-stage
decision. Private/protected/outer/test labels, GPU/CUDA, W1, ranking and scientific claims remain
prohibited.

## 2026-07-19 fixed-v5 final post-verify review

Decision: `ACCEPT_FIXED_V5_VERIFIED_ENGINEERING_SMOKE`; the bounded engineering smoke and its
read-only verification are complete. This does not approve W1, ranking or scientific use.

The one-use verify command produced only the external mode-`0444` report
`/tmp/qm9_v31_bounded_smoke_v5_verify_report_v1.json`, SHA-256
`f5a66ad3ca6e4f235e4e7b52566870151492ab9e8969364bdc2d5b0f51beb03d`.
It reports `evidence_reauthenticated=true`, `execution_identity_matches_current_code=true`, audit
code SHA-256 `4f4c40085293e2347d00368fbe510051ef3de10f9f1267c48f5717a8b0b70f62`,
`scientific_result=false` and `publication_ready=false`. Its run-start, ledger, summary and
proposal hashes exactly match the fixed lineage, and its proposal decision remains
`NOT_APPROVED`.

Independent post-verify inspection confirmed that verify did not mutate the fixed lineage. The
current tree excluding the allowed later `POSTRUN_MANIFEST.json` still exactly matches the embedded
inventory and SHA-256
`9dac6ea70cd990a60748a35f6ab4d8ef9281c953aceb3db879155187713063ac`;
the POSTRUN file remains
`4d19d937872fac3b72bf0ef73b475984e56d85df16f3f0188d12c207a3f66fe6`.
Outer result/status, terminal, ledger, telemetry, three worker results/profiles, three 24-by-12
checkpoints, replay count, MLP loss/anomaly evidence, zero protected-label reads and all 291 false
governance boundary fields remain identical to the accepted postexecute audit.

The verify authorization is consumed and cannot be retried. The engineering result establishes
that the fixed v5 production subprocess path can complete under the reviewed CPU synthetic,
60-second and 1.5-GiB envelope with coherent immutable evidence. It supplies no candidate ranking,
scientific comparison or publication result. Any move toward formal W1 requires a new independent
feasibility/capacity decision, fresh permanent lineage, exact budget and label-access contract; no
current artifact authorizes private/protected/outer/test-label access, GPU/CUDA, W1/general
training, ranking, promotion, ensemble selection or scientific claims.

## Wave 1 implementation status

The v3 bounded CPU smoke did not satisfy the success-terminal contract. The later fixed v5 bounded
CPU smoke did satisfy its reviewed success and read-only verification contracts. Neither run is W1,
and neither supplies a candidate rank, promotion, ensemble decision or scientific result. All fixed
v5 evidence remains `scientific_result=false` and `publication_ready=false`.

The next capacity step is the frozen 8,192-row train-only medium profile described in the
[medium profiling brief](qm9_model_selection_v31_medium_profile_brief.md). Its runner is
implemented, and the separate zero-model public outer seam is accepted as engineering evidence.
The production profile still requires a fresh lineage, remaining real three-backend integration
evidence, independent review and one-use authorization; it has not been prepared or executed. Only
a later hash-bound decision may use accepted production engineering evidence to consider private
W1.

## Required review artifacts

1. v3 failure inventory and recursive Why record;
2. reviewed v3.1 config and schema hashes;
3. real-backend profiling and stage-time report;
4. explicit memory worksheet and measured peaks;
5. conservative runtime projection;
6. optimization-equivalence report;
7. heartbeat/resource/curve telemetry samples;
8. signal, deadline, stale-heartbeat, resource and partial-ranking forward-test report;
9. exact provenance inventory; and
10. independent W1 approval binding a fresh permanent directory.

## Durable prohibitions

- Do not resume, mutate or backfill the empty v3 W1 directory.
- Do not infer a runtime root cause or candidate rank from liveness/resource anecdotes.
- Do not introduce sparse neural architecture as an execution-only change.
- Do not start private W1 from implementation progress or a surrogate smoke.
- Do not start W2 from a W1 execution approval.
- Do not inspect outer/test labels during profiling, smoke or W1.
- Do not publish inner-CV, partial, failed or unreviewed output as a scientific result.

See the [v3.1 protocol](qm9_model_selection_v31_protocol.md) and
[reproducibility status](reproducibility.md#qm9-model-selection-v31).

## Public fixed three-backend gate

- [x] Deterministic complete-group public bundle: 2,065 inputs and 12 scaled targets.
- [x] One runtime/bundle plan hash for the three production candidates.
- [x] Public-specific review/auth contract and `public-fixed-v1` profile.
- [x] One outer cgroup with direct worker descendants.
- [x] Atomic stage/loss/progress heartbeat plus RSS/disk monitoring.
- [x] POSTRUN replay of profiles, ledger, checkpoints and raw MLP curves.
- [x] Independent post-review and deterministic archive validation.
- [x] Focused tests and non-integration regression suite.
- [ ] Second independent pre-execution GO review.
- [ ] Permanent bundle and PREPARED artifact.
- [ ] Independent review plus distinct one-use authorization.
- [ ] One monitored Ridge/XGBoost/MLP execution.
- [ ] Independent post-review and archive.

Implementation and tests do not authorize unchecked stages. Private QM9 targets and the full
five-seed experiment remain untouched.
