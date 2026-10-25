# Aljabr Engineering Roadmap

> **Purpose:** prioritize correctness, measurable performance, and maintainable hardware acceleration for Aljabr's Java tensor/compute infrastructure.
>
> This is a rolling, gate-based roadmap rather than a date-based promise. Work advances when the acceptance criteria for a phase are met. Status below reflects the repository state when this document was prepared; update it as PRs merge.

## Product principles

1. **Correctness before speed.** An optimized kernel or decode path must be numerically validated against a trusted baseline before it is used to justify a performance claim.
2. **Measure the same thing.** Keep prefill, token decode, kernel-only, Java-side, and end-to-end timings separate. Do not compare numbers collected with different timing scopes.
3. **Make runtime behavior observable.** Logs and diagnostics must identify the selected backend/path and explain why a faster path is unavailable.
4. **Preserve safe fallback paths.** Unsupported model features should use a known-correct fallback, not silently bypass an eligibility guard.
5. **Keep contracts explicit.** Native ABI, cache ownership, buffer lifetime, quantization layout, and backend capability assumptions need tests and documentation.
6. **Keep the core modular.** Aljabr provides infrastructure and compute primitives; orchestration and user-facing inference workflows belong in downstream consumers such as Aqli.

## Current baseline

The following work is already present on mainline and should be treated as a foundation, not reimplemented:

- Shared-KV Metal decode reads the cache from the resolved KV-owner layer and avoids writing invalid per-layer cache entries.
- Full-GPU decode eligibility diagnostics explain several reasons the path can be unavailable, with focused unit tests.
- Metal triple-batch performance measurement is opt-in; Gradle can expose benchmark output.
- Metal native-library discovery and ABI handling have received fixes in recent work.

**Immediate review gates:** PR [#33](https://github.com/andalus-ai/aljabr/pull/33) covers triple-batch numerical parity, including the production-relevant batch size 27; PR [#34](https://github.com/andalus-ai/aljabr/pull/34) covers persisting benchmark results to CSV. Confirm their latest CI and Apple Silicon validation before treating those items as complete.

---

## Phase 0 — Establish a trustworthy runtime and baseline

**Priority: P0 · Do this before making claims about full-GPU decode performance**

### Work

- Rebuild and install the actual Aljabr/Aqli runtime through the supported build/install flow. Avoid patching individual class files into a packaged JAR as a normal diagnostic workflow.
- Confirm that the installed artifact contains the same diagnostic code as the source revision being tested.
- Make startup diagnostics unambiguous: backend loaded, native library path/name, decode path selected, and a structured reason when full-GPU decode falls back.
- Verify the installed runtime can be rebuilt reproducibly from a clean checkout; investigate packaging or duplicate-entry failures rather than mutating an uncertain JAR.
- Capture a baseline for a fixed model, prompt, generation length, JVM/runtime configuration, and hardware.

### Acceptance criteria

- A clean build/install produces a valid runtime artifact and passes archive-integrity checks.
- The runtime log clearly reports either **full-GPU decode enabled** or the exact fallback reason. Absence of a log line is not evidence that a path is active.
- The test report records commit SHA, JDK/Gradle version, chip/OS, native library path, model identifier/quantization, and command line.
- Baseline results distinguish prefill latency, per-token decode latency/throughput, and end-to-end latency/throughput.
- The same command and configuration can be rerun to obtain comparable measurements.

### Exit gate

Do not start tuning the full-GPU path until its activation state can be proven from the installed runtime and the correctness baseline is reproducible.

## Phase 1 — Lock down native and batched-kernel correctness

**Priority: P0 · In progress**

### Work

- Complete review and validation of PR #33's Q4_K/Q4_K/Q6_K triple-matvec parity coverage, including batch sizes 4, 8, 15, and 27 where supported.
- Complete review and validation of PR #34's opt-in CSV benchmark output.
- Keep single-row native operations as the reference when testing batched implementations.
- Exercise finite-output checks, relative-error limits, edge dimensions, alignment/stride assumptions, and quantization-block boundaries.
- Preserve shared-KV regression coverage: KV-owner resolution, owner geometry, cache stride, cache writes, and reads must all agree.
- Keep native calls guarded by explicit capability/library checks; tests may skip only when the test's documented native-runtime prerequisites are genuinely unavailable.

### Acceptance criteria

- Batched outputs match repeated single-row outputs within the agreed numerical tolerance (currently the PR #33 proposal uses relative error **2e-3**); explain and approve any change to that tolerance.
- Batch 27 is actually executed on supported Apple Silicon hardware, not merely compiled.
- Repeated benchmark runs produce machine-readable results containing batch size, mean/median/p90 latency, per-item latency, and throughput.
- Correctness tests remain independent from performance thresholds so noisy hardware does not make CI flaky.
- The regular test suite remains safe on hosts without an active Metal device.

## Phase 2 — Make full-GPU decode eligibility correct and explicit

**Priority: P0 · Main functional objective**

### Work

- Use the Phase 0 diagnostics to identify the exact blocker for the target model/runtime; do not infer the blocker from a missing log line.
- Trace eligibility from model metadata and weight registration through layer registration, KV-cache initialization, output-layer registration, and the first native decode step.
- Address per-layer token embeddings deliberately:
  - implement their semantics in full-GPU decode if supported; or
  - retain the guard and use a correct fallback until support is complete.
  - Never remove the guard solely to force the optimized path to activate.
- Validate model-layer indexing, shared-KV owner mapping, RoPE/position handling, attention masks, cache positions, output projection, logits, and token selection against the existing correct path.
- Keep diagnostic reasons specific enough to distinguish unsupported model features from missing weights, native-library unavailability, registration failure, or cache initialization failure.

### Acceptance criteria

- Unit tests cover each eligibility failure branch and the eligible case.
- Registration and initialization failures identify the failing stage/layer without masking the original cause.
- On a supported model, logs prove that the full-GPU path is selected and successfully completes decode steps.
- On unsupported models, fallback is explicit and output remains correct.
- For a fixed prompt and deterministic configuration, compare logits or token outputs with the established reference path. Document any expected numeric differences and tolerance.
- Test at least one model with shared-KV layers and one model exercising the relevant embedding behavior before broadening eligibility.

### Exit gate

The full-GPU route is not considered delivered merely because eligibility returns true or native calls return success. It must pass end-to-end correctness tests and produce coherent output over multiple decode steps.

## Phase 3 — Profile and optimize Metal prefill and decode separately

**Priority: P1 · Begin after Phase 1 correctness gates pass**

### Work

- Establish separate profiles for prompt prefill and single-token decode. Identify whether time is dominated by matvec kernels, attention/SDPA, cache traffic, memory movement, Java/FFM call overhead, synchronization, or launch overhead.
- Use the opt-in batch benchmark and end-to-end baseline together; do not treat synthetic kernel throughput as user-visible generation speed.
- Evaluate optimizations in measured order:
  1. Avoid redundant host/native transitions, buffer registrations, and synchronization.
  2. Improve memory locality and reuse for quantized weights and KV-cache reads.
  3. Tune supported batch sizes and dispatch thresholds from measurements rather than assumptions.
  4. Consider kernel fusion or specialized decode kernels only where profiling demonstrates a bottleneck.
- Preserve stable single-item behavior while extending multi-row prefill and batching.
- Track CPU and Metal fallback costs so dispatch logic does not make small workloads slower.

### Acceptance criteria

- Each optimization PR includes before/after measurements from the same hardware, runtime, model/quantization, prompt, generation length, warm-up policy, and timing scope.
- Report median and tail latency as well as throughput; include repeated runs or dispersion where practical.
- Numerical parity and the full relevant test suite pass.
- Keep or revert an optimization based on measured benefit; avoid performance claims based on a single run.
- Report both prefill and decode effects, including regressions, even when only one phase was targeted.

## Phase 4 — Harden backend contracts and fallback behavior

**Priority: P1**

### Work

- Treat Java-to-native interfaces as explicit contracts: argument order/types, struct layout, alignment, ownership, error codes, and lifetime of native buffers.
- Expand ABI regression tests whenever a native function signature or configuration structure changes.
- Test native-library discovery across documented locations and ensure an explicit user override is either honored or rejected with a clear error; never silently load a stale legacy library.
- Ensure missing/incompatible native libraries fail safely into supported fallback behavior where available.
- Keep the CPU backend as a reliable correctness reference and maintain parity expectations for other hardware backends where equivalent operations exist.
- Document backend capabilities and unsupported operations so dispatch decisions are explainable.

### Acceptance criteria

- ABI-sensitive changes include a test that fails on a Java/native signature or layout mismatch.
- Library discovery tests verify precedence and the selected library identity.
- Error handling preserves the original native failure details and does not turn a failed optimized path into plausible but invalid output.
- Fallback paths have at least one correctness test for each advertised capability boundary.

## Phase 5 — Build a sustainable test and release matrix

**Priority: P1**

### Work

- Keep fast, deterministic Java unit tests separate from hardware-dependent integration tests and opt-in performance benchmarks.
- Run Metal integration tests on Apple Silicon CI; ensure ordinary CI can still compile and run non-Metal tests without a Metal device.
- Add regression tests for discovered bugs at the narrowest useful layer (unit, ABI, kernel parity, model integration, or end-to-end).
- Make benchmark reports and environment metadata accessible as build artifacts.
- Verify clean builds, module boundaries, dependency declarations, and runtime packaging with the documented JDK/Gradle toolchain.
- Maintain troubleshooting instructions for native build/install, library discovery, runtime diagnostics, and benchmark execution.

### Acceptance criteria

- A regression is caught by a repeatable test or documented hardware-specific validation, not only by manual observation.
- Hardware-specific tests are clearly marked and do not create misleading passes by skipping unsupported assertions.
- A clean-checkout validation procedure is documented and succeeds.
- The release artifact and its source revision can be traced unambiguously.

## Phase 6 — Broader infrastructure maturity

**Priority: P2 · After the immediate Metal correctness/performance work**

- Audit backend SPI boundaries and keep hardware-specific details out of core tensor/model contracts.
- Review resource lifecycle and concurrency safety for native handles, model buffers, sessions, and inference memory.
- Improve observability with consistent backend selection, fallback reasons, operation timing hooks, and optional diagnostics that are disabled or low-cost by default.
- Expand model/quantization compatibility through explicit capability tests rather than broad optimistic assumptions.
- Prioritize CUDA, CPU, and other backend work according to real user workloads and measured parity/performance gaps.
- Keep performance reports versioned so regressions can be compared across commits and hardware generations.

These are investigation areas, not a commitment to implement every item; each should be supported by an identified use case and measurable acceptance criteria.

---

## Execution order

1. **Prove the installed runtime and active decode path** (Phase 0).
2. **Close Metal numerical parity and benchmark-reporting gates** (Phase 1; PRs #33 and #34).
3. **Fix the actual full-GPU eligibility blocker without weakening correctness guards** (Phase 2).
4. **Profile, then optimize the measured bottleneck** (Phase 3).
5. **Harden ABI, discovery, fallback, CI, and release reproducibility** (Phases 4–5).
6. **Expand broader architecture and backend maturity** only after the mainline behavior is measurable and stable (Phase 6).

## Definition of done for each performance change

A performance change is complete only when the PR includes:

- The bottleneck and expected mechanism of improvement.
- A correctness regression test or clearly documented hardware validation.
- Comparable before/after measurements and environment metadata.
- Fallback behavior and unsupported cases.
- Relevant unit, integration, and build validation results.
- Documentation updates when behavior, configuration, or troubleshooting changes.

**Non-goal:** maximizing a single benchmark number at the expense of numerical correctness, robust fallback, reproducibility, or the library's modular contracts.
