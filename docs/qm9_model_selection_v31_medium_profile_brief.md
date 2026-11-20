# QM9 model selection v3.1 medium profiling brief

Status date: 2026-07-20

Decision: `PUBLIC_SYNTHETIC_OUTER_E2E_ACCEPTED_ENGINEERING_ONLY — PRODUCTION/PRIVATE NO-GO`

Purpose: measure engineering feasibility and observability at a meaningful intermediate scale. This
profile is not model selection, a ranking exercise, a scientific experiment or a publication
result. It grants no permission for private W1, outer/test-label access, CUDA/GPU execution,
promotion or ensemble selection.

## Public synthetic outer-seam status

The medium-profile lifecycle and outer-wrapper implementation now exist. On 2026-07-20, one
separately authorized, explicitly opt-in public synthetic five-stage E2E invocation completed with
`1 passed in 8.40s` and then passed an independent read-only post-run review. The accepted status is
`PUBLIC_SYNTHETIC_OUTER_E2E_ACCEPTED_ENGINEERING_ONLY`.

That invocation exercised `prepare -> review -> authorize -> execute -> verify`, including the
bounded outer process, cgroup evidence and exact POSTRUN replay. Its outer elapsed time was
1.631 seconds and its aggregate cgroup peak was 119,193,600 bytes. All source, derived,
outer-validation, test, protected and private-evaluation label-read counters were zero. The
workload was a zero-label, zero-model public synthetic marker: Ridge, XGBoost and MLP did not run.

The preserved evidence archive is
`/tmp/qm9-v31-public-outer-e2e-evidence-v1.tar.gz`, SHA-256
`8294eb3dfb2b871bb924380c52abf631b69b4176bbd9ac38f66e5dea30ce656a`. The independent
post-review record is `/tmp/qm9_v31_public_outer_e2e_postreview_v1.json`, SHA-256
`60f91332fde9b9a8be8a6d306dcea86f7dd8d9bce633fd0c0d5a025f432eaa47`. This evidence
validates only lifecycle, provenance, one-use authorization, outer resource containment and
POSTRUN plumbing. It does not validate model execution, predictive quality, real-backend
integration or medium-scale feasibility.

Production/private preparation and execution remain `NO-GO`. The real three-backend medium
integration for Ridge, XGBoost and MLP has not run and requires a fresh lineage, independent review
and separate one-use authorization.

## Frozen scope

- Use a minimum target of 8,192 authenticated fixed-training members and their 12 training targets.
- Select rows deterministically by connectivity group, never by target value. Add each selected
  group in full and stop when the accumulated row count first reaches or exceeds 8,192; persist the
  realized count. Preserve group integrity while constructing the subset, then make one
  deterministic 90/10 split by identity-group count. Persist the realized row fractions because
  unequal group sizes mean the row ratio need not be exactly 90/10. No connectivity group may
  cross that boundary.
- Use the single frozen seed `20260713` for subset tie-breaking, the inner split and model
  stochasticity. Record exact ordered row IDs, group IDs and their hashes before execution.
- Use the authenticated production feature recipe and target order. Fit every feature transform and
  target scaler on the inner-training partition only, then apply it unchanged to inner tune.
- Run exactly one fixed configuration for each family: Ridge, XGBoost and MLP. The three complete
  configuration objects must be reviewed and hash-bound before authorization. There is no
  hyperparameter search, retry-based tuning, cross-validation, ranking, blend or ensemble.
- Run each family once. A failed or incomplete family is evidence about feasibility; it must not be
  silently retried, replaced or interpreted as inferior predictive quality.
- Use a fresh permanent lineage. Do not reuse, resume, mutate, repair or derive authority from the
  verified fixed-v5 directory.

## Frozen resource and time envelope

- CPU only; GPU/CUDA discovery or execution is outside scope.
- Aggregate process-tree hard memory limit: 4 GiB (4,294,967,296 bytes).
- Worker/resource refusal threshold: 3.2 GiB. Crossing it must stop new work and produce a durable,
  non-success terminal rather than attempt to consume the remaining hard-limit margin.
- Total hard wall-clock limit: 2,400 seconds.
- Soft drain begins no later than 2,280 seconds, leaving a fixed 120-second shutdown and evidence
  finalization reserve. No new model unit may start after soft drain.
- Preserve at least 20% measured memory headroom for any later capacity approval. Merely completing
  below the 4-GiB hard limit is not sufficient.
- Freeze working directory, command argv, code state, dependency lock, device/thread settings,
  numeric precision, input hashes, configurations and output root before authorization.

## Label-access gate

- The only permitted labels are the 12 targets for the selected fixed-training members.
- Outer-validation, fixed-test, private, protected and any other evaluation labels must remain
  unavailable to every worker and supervisor component.
- Record label-access counters at start, on every heartbeat or stage transition, and in the final
  terminal. All protected, outer, test and private counters must remain exactly zero.
- Any unavailable counter, unexpected label path, nonzero prohibited counter or target-derived row
  selection is a fail-closed governance event.

## Required measurements

Record total and per-stage wall time and CPU time for data selection, transformation, cache or
tensor construction, fit, validation/prediction, loss computation where applicable, checkpointing
and finalization. Also record completed versus planned units, current stage, elapsed budget, ETA,
process count, baseline/current/peak aggregate RSS, worker peak RSS, free disk floor, read/write
volume where available and artifact bytes.

Model-specific observability must include:

- Ridge: fit and prediction timing, completion state and checkpoint identity. Epoch loss curves are
  not applicable and must be represented as unavailable-by-design, not missing evidence.
- XGBoost: per-target progress, fit/prediction timing and best iteration where the fixed
  configuration exposes it. Do not fabricate an epoch loss curve when the backend does not emit
  one.
- MLP: per-epoch standardized training MSE, inner-tune MNMAE, all 12 per-target tune metrics,
  learning rate, gradient and parameter norms, best epoch, restored checkpoint, stale count,
  validation-increase count, stop reason, warnings and anomaly state. Every validation-loss
  increase must be visibly marked even when it does not yet trigger early stopping.

The evidence stream must expose liveness, progress and metric ages; monotonic lifecycle states;
continuous resource sampling; immutable run-start identity; append-only ledger and telemetry hash
chain; authenticated checkpoints; strict replay; exactly one terminal; closed streams; final
process-group/cgroup emptiness; and cleanup signals only when they were actually issued. The outer
resource/status sampler must remain independent of the model workers.

## Acceptance gates

An independent post-run review may accept this only as engineering feasibility evidence when all
of the following are true:

1. The fresh lineage, exact command, code/config/input/device identities and complete artifact tree
   reauthenticate without mismatch.
2. Ridge, XGBoost and MLP each complete their single planned configuration, every required
   checkpoint authenticates, strict replay succeeds and no planned unit remains reserved or
   running.
3. Status and telemetry remain timely and monotonic, exactly one success terminal exists, streams
   are closed and no process or cgroup survivor remains.
4. No nonfinite value, stale heartbeat, sensor-unavailable condition, resource refusal, hard-limit
   breach, deadline breach or unexplained cleanup action occurred.
5. Peak memory stays below 3.2 GiB and the reviewed worksheet demonstrates at least 20% measured
   memory headroom. The estimate must cover the measured peak; otherwise the worksheet fails.
6. Total execution and evidence finalization complete before 2,280 seconds; the 120-second reserve
   is not consumed as ordinary training time.
7. Protected, outer-validation, fixed-test and private-label counters are present and exactly zero
   throughout.
8. Ranking, promotion, ensemble, W1/general-training, scientific-result and publication-ready
   fields are all false.

Any failed gate yields `NOT_FEASIBLE_PENDING_REVIEW` or a more specific non-success engineering
decision. It does not rank or reject a model family. Successful acceptance may support a later,
separately reviewed capacity decision; it does not itself authorize private W1.

## Remaining public-fixed authorization blockers

As of 2026-07-21, component tests passing is not sufficient to authorize the three-backend outer
run. The following implementation and negative-test evidence is still required:

- independently bind the postreview reviewer and tool identities, and prove their timestamp is
  after terminal, outer completion and POSTRUN creation;
- make the archive self-contained and replay the copied review, authorization, one-use
  consumption, outer evidence, POSTRUN and postreview after extraction; reject duplicate archive
  member names and every unsupported member type;
- reconstruct all 12 XGBoost target/round/evaluation streams from authenticated events and replay
  Ridge's explicit non-iterative stage contract;
- separate process-heartbeat staleness from model-progress staleness so a healthy native call
  longer than 20 seconds is not killed, then test a blocking stage longer than 65 seconds.

These are acceptance specifications, not completed-result claims. Until they pass another
independent read-only review, public-fixed execution remains `NO-GO`.

## Governance sequence and ownership

1. A training engineer maintains the implemented medium-profile runner/config and completes the
   remaining real-backend forward tests without changing the verified fixed-v5 or accepted public
   synthetic lineages.
2. An executor proposes a fresh output root and exact command but does not run it.
3. An independent reviewer freezes the plan, artifact hashes, resource contract and one-use
   authorization.
4. The executor performs at most the authorized invocation and stops.
5. An independent reviewer reauthenticates the preserved output before any status or capacity
   conclusion is recorded.

As of the status date, implementation and the isolated public synthetic outer-seam sequence are
complete. The production 8,192-row sequence has not started: it has not been prepared, reviewed,
authorized or executed. This document freezes its scope; the public synthetic result is not
production execution approval and does not claim that the 8,192-row profile exists or passed.
