# Aljabr Developer Tooling Architecture

> **Status:** Proposed architecture and implementation sequence  
> **Scope:** Developer-facing inspection, diagnostics, profiling, numerical validation and interoperability  
> **Boundary:** Tooling is an optional framework capability. It must not become a serving control plane or a mandatory dependency of tensor/model consumers.

## 1. Product objective

Developer tooling should make Aljabr understandable when code is correct, explainable when execution is slow, and actionable when computation fails. It should support the complete development lifecycle:

1. Explore tensors, models, parameters and datasets.
2. Understand shapes, dtypes, layouts, devices and execution plans.
3. Diagnose numerical errors and unsupported operations.
4. Measure performance without misleading results.
5. Observe memory ownership, copies, materialization and host/device transfers.
6. Export findings to ordinary Java diagnostics and established tools.

Tooling is not a collection of unrelated print helpers. It is a small, composable observability and diagnostics foundation with optional adapters and front ends.

## 2. Repository baseline and issues to address

The current repository already contains useful but fragmented pieces:

- `models/aljabr-model-common/.../ModelProfiler.java` profiles a whole model forward call, but estimates activation memory using a fixed depth heuristic and parameter memory as float32 regardless of actual dtype/storage. It does not synchronize asynchronous device work, expose warmup/measurement policy, or report percentiles/provider/kernel breakdown. Its sample uses Python-style named arguments that are not valid Java syntax.
- `spi/aljabr-spi-inference/.../observability/AdapterMetricsRecorder.java` and `AdapterMetricSchema.java` define inference-adapter metrics, not a general tensor/model/scientific diagnostics contract.
- `plugins/ml/autograd/.../MetricsTracker.java` stores scalar histories and exports CSV, while `TensorBoardWriter.java` implements a training-oriented event writer. The legacy autograd project is excluded from the default build, so these classes are not a universal tooling foundation.
- JMH benchmarks exist in provider and algorithm modules, but benchmark harnesses are not the same as runtime profiling or explainability.
- Model metadata and runtime profile records exist in multiple places. Their meaning should not be conflated with measured runtime observations.

These findings are source-level observations, not a claim that the current tools never work. The immediate issue is that they do not yet present one consistent, general-purpose developer experience.

## 3. Design principles

### 3.1 Optional and layered

- The core diagnostics contracts must have no dependency on a specific logging, metrics, tracing, visualization, notebook or serving product.
- Tensor-only consumers should not need model modules; numerical consumers should not need tensors; model inspection should be an optional extension.
- Provider-specific probes are optional implementations discovered through explicit registration/ServiceLoader where appropriate.
- No mandatory background threads, global singleton, agent, network exporter or file output.
- A no-op/disabled path must have negligible overhead and must not allocate per operation unnecessarily.

### 3.2 One semantic model, several surfaces

Expose the same information through:
- Ordinary Java API for programmatic use.
- Fluent scopes/builders for opt-in instrumentation.
- Structured text/JSON/CSV exports where useful.
- JFR events for low-overhead JVM-native analysis.
- Optional adapters for Micrometer/OpenTelemetry and notebook visualization, without coupling core to those libraries.

Do not build a custom notebook kernel or custom dashboard as a prerequisite. JBang, JShell and Jupyter should consume the same published APIs.

### 3.3 Measurements must be honest

Every metric must identify whether it is:
- measured, estimated, sampled or unavailable;
- host-side or device-side;
- synchronous or asynchronous;
- inclusive or exclusive;
- cold-start, warmup or steady-state;
- wall-clock, CPU time or device execution time.

Never label an estimate as actual memory usage. Never compare asynchronous GPU launch latency with synchronized execution latency without clearly stating the difference.

### 3.4 Diagnostics are structured data first

Use typed diagnostic records with stable codes, severity, category, operation/context, relevant tensor metadata, provider context, explanation and remediation. Human-readable messages are a rendering of structured diagnostics, not the only source of truth.

Do not include tensor contents, prompts, user data or model weights in diagnostics by default. Sensitive payload capture must be explicit and opt-in.

## 4. Proposed capability model

### 4.1 Inspection

Inspection is a read-only snapshot of an object at a point in time.

Initial inspectable subjects:
- Tensor: shape, dtype, layout/strides, device, storage kind, contiguity, ownership/lifetime state and gradient metadata when available.
- Model/module: architecture identity, module tree, parameter/buffer counts, parameter dtype/device distribution and declared capabilities.
- Execution: selected provider, operation plan, fallback decisions and unsupported capability reasons.
- Dataset/batch: schema, cardinality when known, shape/type summaries and sampling metadata.

Inspection must not force expensive materialization or read tensor contents unless explicitly requested. Distinguish metadata-only inspection from value sampling.

### 4.2 Diagnostics

Initial diagnostic categories:
- Shape/rank/broadcast/indexing mismatch.
- Dtype conversion/promotion and unsupported dtype/operator combinations.
- Device/provider capability mismatch and fallback.
- Invalid layout/stride/alignment.
- Closed/released storage, invalid lifetime and ownership violations.
- Non-finite values (NaN/Inf), overflow/underflow and numerical instability.
- Autograd graph disconnection, missing gradient and invalid gradient shape.
- Resource exhaustion and allocation/transfer failures.

Each diagnostic should have a stable code, severity, message, explanation, optional remediation, operation ID and structured context. Preserve original causes for debugging while keeping public error types stable.

### 4.3 Profiling

Provide separate layers; do not overload one profiler class:

1. **Benchmarking:** repeatable comparative experiments with explicit warmup, iterations, forks, setup/teardown and environment metadata. Prefer JMH for JVM microbenchmarks.
2. **Runtime profiling:** opt-in timing and counters around operations, modules and model execution.
3. **Execution profiling:** provider/kernel timings, synchronization points, fallback and transfer costs when providers expose them.
4. **Memory profiling:** allocation, retained storage, peak usage, pool hits/misses, copies, materializations and transfer bytes when measurable.
5. **Numerical profiling:** optional min/max/mean/variance, norms and non-finite counts; sampling must be explicit because reductions can synchronize or copy device data.

Reports should include sample count and distribution (at least min/median/mean/max; p95/p99 where sample count makes them meaningful), not just a single average. Include warmup and measurement policy, clock type, synchronization policy, provider/device and runtime/build identity. Keep measurement overhead visible.

### 4.4 Memory and data-movement tracing

Integrate with the memory/storage contracts rather than attempting to infer ownership externally. Events should represent allocation, retain, release, view creation, copy, materialization, host/device transfer, pool reuse and failure. Provide correlation IDs and byte counts where known.

Tracing must be bounded and opt-in. Avoid retaining tensor contents or unbounded event histories. Distinguish logical tensor size from physical allocation size and estimated from measured values.

### 4.5 Numerical validation

Provide composable helpers for:
- all-close / tolerance assertions with absolute and relative tolerance;
- finite-value assertions and NaN/Inf location summaries;
- shape/dtype/layout assertions;
- gradient checking and finite-difference comparison;
- backend parity comparisons using deterministic fixtures where supported;
- reference-vs-optimized kernel validation.

Diagnostics should report mismatch count, maximum absolute/relative error and first mismatch coordinates without dumping full tensor data by default.

## 5. Module boundaries

Do not create one monolithic `aljabr-tools` dependency.

Proposed logical boundaries (names are provisional until checked against existing module conventions and publication metadata):

| Capability | Dependency direction | Notes |
|---|---|---|
| Diagnostics API | standalone, JDK-only | stable codes, severity, context and sink interfaces |
| Observability API | standalone or shared diagnostics artifact | scopes, events, counters, timers and no-op implementation |
| Tensor inspection | depends on tensor API | optional tensor metadata inspector |
| Model inspection | depends on model/module APIs | optional tree and parameter inspector |
| Profiling core | depends only on observability contracts | generic timing/statistics and report model |
| Provider profiling | provider -> profiling SPI | provider-specific device/kernel timing |
| Memory tracing | depends on storage/memory contracts | allocation/copy/transfer lifecycle events |
| Numerical validation | depends on tensor API | assertions, comparison and gradient-check helpers |
| Exporters | optional adapters | JFR, JSON/CSV, Micrometer/OpenTelemetry, TensorBoard if appropriate |
| CLI/notebook presentation | external integration/examples | no CLI or notebook dependency in core |

Avoid a dependency cycle: tensor/model/provider implementations may publish events to a tiny optional sink contract; the inspector/profiler consumes snapshots/events and must not be required by compute implementations.

Before adding artifacts, map existing core/SPI contracts and package conventions. Reuse a suitable existing diagnostics or observability artifact if it is semantically general and dependency-light.

## 6. Java developer experience

Support explicit and fluent styles over the same API.

Illustrative target API (design sketch only; not implemented API):

```java
try (var session = Diagnostics.session()
        .with(TensorInspector.metadata())
        .with(ExecutionProfiler.summary())
        .with(MemoryTrace.allocationsAndTransfers())
        .to(DiagnosticSinks.console())) {
    var result = model.forward(input);
    session.report().print();
}
```

Traditional usage must remain straightforward:

```java
var snapshot = TensorInspector.inspect(tensor);
var report = Profiler.profile(operation, ProfileOptions.defaults());
DiagnosticSinks.console().accept(diagnostic);
```

Annotation-assisted instrumentation may be considered later for module/model boundaries, but must be optional and must not add hidden instrumentation to hot paths. Prefer explicit scopes and generated metadata over reflection-heavy interception.

All scopes must have deterministic close behavior. No global mutable configuration is required. Thread/context propagation must be explicit and safe for asynchronous execution.

## 7. Interoperability

- **JFR:** preferred first JVM-native export surface for structured runtime events.
- **JMH:** preferred microbenchmark harness; do not recreate its benchmark lifecycle.
- **SLF4J:** optional logging adapter, never a mandatory dependency of diagnostics contracts.
- **Micrometer/OpenTelemetry:** optional metrics/tracing bridges, preserving Aljabr semantic attributes.
- **TensorBoard:** optional training visualization adapter; keep training-specific event recording outside the minimal diagnostics core.
- **JSON/CSV:** deterministic, versioned export formats for reports and offline analysis.
- **JShell/JBang/Jupyter:** examples should show scoped sessions, inspection and report export using published artifacts.

## 8. Phased implementation plan

### Phase 0 — Inventory and contracts

- Map existing diagnostic, metric, profiler, model-info and runtime-profile types.
- Identify package/artifact ownership and legacy modules.
- Specify stable diagnostic/event schema and measurement terminology.
- Do not introduce public API until duplicate contracts and dependency direction are understood.

### Phase 1 — Minimal diagnostics foundation

- Add stable diagnostic code/category/severity and immutable context.
- Add sink interface, no-op sink and console renderer.
- Add tests for immutability, safe context handling and deterministic formatting.
- Integrate only at a few high-value tensor validation/provider fallback boundaries.

### Phase 2 — Inspectors and numerical assertions

- Add tensor metadata inspector and numerical assertion helpers.
- Add model/module inspector as a separate optional extension.
- Ensure metadata inspection does not force materialization or device synchronization.
- Add tests for closed resources, non-contiguous tensors, dtypes and optional gradients.

### Phase 3 — Profiling and memory trace

- Replace heuristic memory estimates in existing model profiling with explicit measured/estimated/unavailable fields.
- Add correct warmup/measurement options and provider synchronization hooks.
- Add operation/model scopes, distributions and environment metadata.
- Add storage lifecycle and copy/transfer event hooks.

### Phase 4 — Export and interactive workflows

- Add JFR exporter and deterministic JSON/CSV report output.
- Add optional metrics/tracing adapters only when there is a concrete consumer.
- Publish tested JBang, JShell and Jupyter examples.
- Add examples that can be run using published artifacts outside the repository checkout.

## 9. Acceptance criteria

A tooling capability is ready for stable use only when:
- It is optional and has a tested dependency boundary.
- Its measurement semantics and overhead are documented.
- It does not expose tensor contents or sensitive model inputs by default.
- It behaves correctly under exceptions, cancellation and resource closure.
- Async/device timing states synchronization policy explicitly.
- Memory numbers are labeled measured, estimated or unavailable.
- Tests cover empty, invalid, closed, concurrent and failure cases as appropriate.
- Examples run from published dependencies and close all resources.
- It does not pull AQLI, Tafkir, a native provider, notebook runtime or visualization server into core consumers.

## 10. First implementation recommendation

Start with **Phase 0 followed by a minimal JDK-only diagnostics API**, but first inspect existing core error/diagnostic abstractions and dependency metadata. Do not immediately refactor `ModelProfiler` or move legacy metrics classes: they may have consumers and compatibility constraints. The first code PR should add the smallest stable contract and one integration point, then grow by evidence.


## 11. Auditability and execution evidence

Observability answers what happened and how the framework behaved. Auditability preserves structured evidence of consequential framework decisions and lifecycle transitions. They are related but must remain distinct from ordinary logs, metrics, traces and profiling samples.

### 11.1 Audit event contract

Define an immutable, versioned audit-event contract with at least:
- event/schema version and stable event type;
- event timestamp plus explicit clock/time semantics;
- source/actor classification (caller, Aljabr framework, provider/extension);
- action, target/resource identity and outcome;
- operation, parent and correlation identifiers where available;
- safe, typed context attributes and optional reason/remediation codes.

Audit events should cover meaningful transitions such as provider discovery/initialization/shutdown, compatibility evaluation and fallback, model/format validation decisions, resource lifecycle failures, explicit configuration changes applied by framework APIs, and conformance/validation outcomes. Avoid emitting a record for every low-level arithmetic instruction.

The sink is caller-selected and optional. Aljabr must not silently persist audit history, create a database, or claim compliance, tamper-proof storage, non-repudiation or retention guarantees. Those properties require an explicit sink and product-level security/retention policy. Delivery guarantees (best effort, synchronous, buffered) and overflow behavior must be documented per sink.

### 11.2 Privacy and security

- Default to metadata-only records. Never include tensor values, prompts, user payloads, model weights, credentials, secrets or arbitrary configuration dumps.
- Use typed allowlisted attributes rather than arbitrary object serialization or toString() output.
- Apply redaction before export, not only in a presentation layer.
- Make sensitive capture explicit, narrowly scoped and documented; ensure disabled capture is the default.
- Bound buffers and cardinality. Define backpressure/drop behavior; never block a compute hot path indefinitely on an exporter.
- Treat provider-supplied diagnostic text and attributes as untrusted input. Sanitize control characters and prevent secret leakage through exception messages.

### 11.3 Correlation and causal context

Use explicit immutable execution/correlation context with parent-child relationships across nested operations. Context propagation into asynchronous work must be deliberate and documented; do not depend on an implicit thread-local that breaks across executors or virtual threads. Correlation identifiers are opaque and must not embed user data.

### 11.4 Evidence provenance

Reports must carry enough provenance to interpret findings:
- software/build revision, JDK/runtime and relevant API/schema versions;
- provider identity/version and device/host details when applicable;
- model/format identity as safe metadata, not weight contents;
- operation/input metadata, seed and determinism constraints when relevant;
- measurement unit, clock, scope, sample count and synchronization policy;
- whether each value is measured, estimated, sampled, derived or unavailable.

Correctness evidence, performance evidence and operational audit are separate report sections. A successful benchmark is not proof of numerical correctness; a provider declaration is not proof of compatibility; a passing unit test is not proof of hardware validation.

## 12. Implementation slices and release gates

Keep the implementation incremental and evidence-driven:

1. **Inventory:** locate existing exception, validation, provider-selection, profiling, metrics, memory and model-loading hooks; map their modules, dependencies and consumers before introducing new public types.
2. **Foundation:** add the smallest dependency-light diagnostics/event contracts, stable codes, typed safe context, no-op sink, explicit scope/correlation and deterministic console rendering.
3. **High-value integration:** instrument shape/dtype/layout validation, provider compatibility/fallback, and resource-lifecycle failures first. Add tests at the owning module boundaries.
4. **Audit sink:** add caller-supplied audit sink semantics, bounded delivery and failure policy. Do not introduce implicit persistence.
5. **Numerics and memory:** add optional numerical assertions and storage lifecycle/copy/transfer events, with explicit synchronization and measured/estimated semantics.
6. **Profiling:** add operation/model scopes and distributions; correct legacy heuristic profiler reports only after consumer and compatibility audit.
7. **Exporters:** add JFR and deterministic JSON first; add OpenTelemetry/Micrometer/SLF4J adapters only for concrete consumers.
8. **External usability:** provide examples and consumer tests against published artifacts; ensure diagnostics can be used without AQLI, Tafkir, native providers or dashboards.

Stable release requires tests for immutability, schema compatibility, privacy/redaction, boundedness, concurrency/correlation, no-op overhead, exception and cleanup paths, exporter failure behavior, and supported dependency profiles. Hardware/device measurements additionally require hardware-specific evidence. Until these gates pass, mark the implementation in progress or experimental rather than stable.

## 13. Implementation status

This document defines the target contract and implementation gates. It is not evidence that the APIs, event sinks, exporters or integrations already exist. Track each deliverable separately as planned, in progress, implemented, validated, stable or optimized, and link status to source/tests and the exact validation environment.


## 14. Diagnostics foundation — implementation slice 1

The first implementation increment extends the existing `core:aljabr-diagnostics` artifact. It does not create a parallel diagnostics module or replace the existing `Diagnostic`, `DiagnosticSeverity`, `DiagnosticSink`, or `DiagnosticFormatter` contracts.

### Implemented in this slice

- `DiagnosticContext`: immutable execution context with opaque correlation and parent IDs, operation/component/provider fields, and explicitly selected attributes.
- Context attributes are bounded (32 entries, 64-character keys, 512-character values), deterministically ordered, and immutable after construction.
- Standard context keys are reserved so arbitrary attributes cannot spoof correlation, operation, component or provider identity.
- `Diagnostic.withContext(...)` adapts the typed context to the existing diagnostic representation. A named factory deliberately preserves source compatibility for existing constructor calls, including calls passing a null map.
- `DiagnosticSink.NOOP` is a shared disabled sink and `DiagnosticSink.noop()` returns that shared instance.
- Tests cover context immutability, defensive copying, bounded values, reserved keys, typed-context integration, and no-op sink identity.

### Explicit limitations

- Context construction validates structure and bounds, not whether arbitrary caller-provided values are safe. Callers must allowlist metadata; exporters must redact.
- This slice does not add implicit thread-local context propagation, tracing, audit persistence, exporters, background workers, or automatic tensor-value inspection.
- The existing map-based `Diagnostic` constructor remains available for compatibility. New integrations should prefer `Diagnostic.withContext`.
- No performance claim is made until the disabled-path and allocation behavior are measured.

### Validation status

Source and unit tests have been added. CI/build execution is still required before this increment can be marked validated. The next slice should add explicit diagnostic scopes and lifecycle events only after reviewing the actual operation/provider integration points.


## 15. Execution lifecycle scopes — implementation slice 2

The second increment adds explicit, synchronous execution lifecycle observation to the existing JDK-only diagnostics artifact.

### Implemented contracts

- `ExecutionEvent` is an immutable lifecycle record with STARTED, SUCCEEDED, FAILED and ABANDONED transitions.
- `ExecutionObserver` is a minimal synchronous callback contract with a shared no-op implementation.
- `ExecutionScope` requires explicit correlation and operation metadata, emits STARTED immediately, and emits exactly one terminal event.
- Closing an unfinished scope emits ABANDONED. Closing a completed scope is safe and does not emit a duplicate event.
- Durations use monotonic nanoseconds; event timestamps use wall-clock instants. These clocks must not be conflated.
- Failure events contain the exception class name only. Exception messages and stack traces are not copied into lifecycle events.
- Runtime exceptions thrown by an observer are isolated from computation and retained for inspection through `ExecutionScope.observerFailure()`. Fatal JVM errors are not intercepted.

### Usage

```java
var context = DiagnosticContext.builder()
        .correlationId("trace-123")
        .parentId("model-forward-7")
        .operation("tensor.matmul")
        .component("tensor")
        .provider("cpu")
        .build();

var scope = ExecutionScope.start(context, observer);
try {
    var result = left.matmul(right);
    scope.succeed();
    return result;
} catch (RuntimeException failure) {
    scope.fail(failure);
    throw failure;
} finally {
    scope.close();
}
```

When a computation can fail, the scope should be retained outside the try body or use a small helper so the failure path calls `fail(Throwable)` before rethrowing. A plain close without an explicit terminal call is intentionally reported as ABANDONED, not silently treated as success.

### Boundaries and remaining work

- No implicit ThreadLocal, async propagation, executor, queue, background worker or persistence is introduced. Nested/asynchronous callers construct child contexts explicitly.
- These events are lifecycle observations, not diagnostic findings, immutable audit facts or metric samples. Those require separate contracts and delivery guarantees.
- Observer callbacks run synchronously and should be fast/non-blocking. Runtime exceptions are available through the scope accessor; fatal errors propagate.
- This slice does not yet instrument tensor/provider/resource implementations. Integration points must be selected and tested at their owning module boundaries.
- Unit tests cover transition order, failure metadata privacy, abandoned close, observer failure isolation, required context and duplicate completion.

### Validation status

The first CI run exposed a test assertion that constructed an invalid context without calling `build()`; this assertion has been corrected. The same run also exposed a separate CPU backend compile failure at `CpuBackend.java:470` (`@Override` does not match a supertype), which is outside this diagnostics change and must be investigated independently. The lifecycle slice remains in progress until the new branch's targeted tests and CI complete successfully.


## 16. GGUF execution lifecycle integration — implementation slice 3

The next increment connects the generic lifecycle contract to one real model execution boundary: synchronous GGUF inference.

### Implemented integration

- `GgufInferenceEngine` accepts an optional `ExecutionObserver` through an additive constructor overload. Existing constructors remain source-compatible and use `ExecutionObserver.NOOP`.
- Synchronous `execute(...)` and context-aware `execute(..., ModelExecutionContext)` create a single `gguf.infer` scope around the backend call.
- The context uses an opaque generated correlation ID and the selected backend name. Request prompts, messages, user/session identifiers, model paths and arbitrary request metadata are not copied into lifecycle context.
- Backend success emits SUCCEEDED; runtime failure or fatal error emits FAILED with exception class identity only. The engine preserves its existing inference exception behavior and counters.
- Observer runtime failures remain isolated from inference and are reflected in `executionObserverFailureCount()`.
- Tests cover failure event ordering, safe context metadata and observer-failure isolation.

### Deliberate limits

- This integration covers synchronous inference only. Streaming scopes require lifecycle handling at subscription time and correct terminal handling for completion, failure and cancellation; instrumenting `Multi` construction would produce misleading durations.
- Embedding, model load/unload, provider discovery outside GGUF, tensor primitives, resource allocation and kernel-level profiling remain separate integration points.
- The default observer is a no-op. No logging backend, exporter, persistence, background thread or global observer is introduced.
- A generated correlation ID starts a local observation; applications needing end-to-end correlation should add a separately reviewed explicit context propagation API rather than relying on hidden thread-local state.

### Validation status

Tests have been added but require CI execution. The current full CI job is blocked at the unrelated existing CPU compilation error in `CpuBackend.java:470`; this does not establish that the GGUF tests passed. The lifecycle integration remains in progress until targeted tests and CI complete successfully.


## 17. GGUF streaming lifecycle integration — implementation slice 4

The streaming integration applies the same lifecycle contract to Mutiny `Multi` execution while preserving reactive subscription semantics.

### Implemented behavior

- Lifecycle scopes are created from the `Multi.createFrom().deferred(...)` subscription path, not when the stream is assembled. Each subscription receives its own scope and generated correlation ID.
- Normal upstream completion emits SUCCEEDED; upstream failure emits FAILED with exception class identity only; downstream cancellation closes the scope as ABANDONED.
- Active-inference accounting is decremented from the common termination callback. Failure accounting is incremented for upstream failures and synchronous backend stream-construction failures.
- Context-aware streaming delegates the supplied `ModelExecutionContext` unchanged and uses the same lifecycle behavior.
- The no-op observer path retains a lean chain without lifecycle context or scope allocation.
- Tests cover deferred start, successful completion, failure privacy and cancellation/abandonment.

### Deliberate limits

- The scope measures the subscription lifetime through terminal completion/cancellation, not merely stream assembly or first-item latency. Token-level timing and per-chunk instrumentation are not added.
- A backend that returns a hot/shared `Multi` retains its own source semantics; the engine creates a distinct observation per subscription but does not make a hot source cold.
- This does not yet instrument embeddings, model load/unload, resource allocation, or lower-level tensor/kernel operations.

### Validation status

Source and tests have been added. CI must validate the Mutiny event hooks and cancellation behavior before this slice can be marked validated. No local build or hardware test is claimed.

## 18. GGUF model lifecycle observation — implementation slice 5

The fifth integration slice observes backend creation and cleanup at the model lifecycle boundary.

### Implemented behavior

- A successful provider selection is followed by a `gguf.model.load` execution scope around backend creation and registration. The selected provider ID is attached as the provider identity.
- `unloadModel` emits `gguf.model.unload` around backend close. Engine close emits an unload scope for each backend rather than hiding all cleanup behind one engine-level event.
- Load failures and cleanup failures emit FAILED with exception class identity only. Original exceptions retain their existing propagation behavior.
- The no-op observer path calls the backend action directly without creating context or scope objects.
- Observer RuntimeExceptions remain isolated by `ExecutionScope` and contribute to the existing execution observer failure count.

### Deliberate limits

- Provider selection/probing remains represented by the separate provider-decision observer. The load scope measures backend creation and registration after a provider has been selected.
- Model IDs, paths, model metadata, arbitrary configuration, prompts and weight contents are not copied into lifecycle context.
- This is lifecycle observation, not durable audit persistence or proof of model provenance. Format validation and tensor mapping diagnostics remain separate work.

### Validation status

Tests cover successful load, failed load metadata privacy, and unload behavior. They require CI execution; the current full build is known to be blocked by the unrelated CPU backend compile error at `CpuBackend.java:470`, so no passing validation is claimed for this slice.

## 19. GGUF embedding lifecycle observation — implementation slice 6

The sixth integration slice applies the explicit execution lifecycle contract to GGUF embedding operations.

### Implemented behavior

- Embedding instrumentation uses operation `gguf.embed` and starts only when the deferred Mutiny `Uni` is subscribed.
- Each subscription receives its own correlation ID and execution scope.
- Item emission maps to SUCCEEDED, upstream failure maps to FAILED, and downstream cancellation closes the unfinished scope as ABANDONED.
- Active and failed counters are updated for terminal outcomes; synchronous backend construction failures are handled without leaking active counts.
- The no-op observer path retains direct backend delegation and only existing counter hooks.
- Observer RuntimeExceptions are isolated by `ExecutionScope` and counted after termination.

### Privacy and boundaries

Embedding text, request IDs, model IDs, paths, request parameters and vectors are not copied into diagnostic context. This is operation lifecycle telemetry, not embedding-content logging or durable audit persistence.

### Validation status

Tests cover deferred start, success, failure privacy, cancellation accounting and observer-failure isolation. CI is required to validate the Mutiny cancellation hooks and compilation. The current full build is blocked by the existing CPU backend compile error at `CpuBackend.java:470`; no passing validation is claimed.


## 20. GGUF provider readiness probe lifecycle — implementation slice 7

The seventh GGUF integration slice measures the provider's cheap model-readiness probe separately from provider selection decisions and backend construction.

### Implemented behavior

- AUTO selection wraps each available provider's `probe(modelPath)` call in a `gguf.provider.probe` execution scope.
- Probe success emits SUCCEEDED even when the returned capability is not generation-ready; readiness and selection outcomes remain the responsibility of `ProviderDecisionObserver`.
- Probe exceptions emit FAILED with exception class identity only. Existing AUTO selection behavior is preserved: ordinary probe exceptions are recorded as provider decision failures and selection proceeds to the next candidate.
- Explicit provider selection intentionally bypasses the readiness probe and therefore emits no probe lifecycle scope.
- The no-op execution observer path invokes the provider probe directly without allocating a context or scope.
- Observer RuntimeExceptions remain isolated and contribute to the existing execution-observer failure count.

### Privacy and boundaries

Probe context contains only a generated correlation ID, operation/component identity and provider ID. It excludes model paths, model IDs, capability metadata, probe explanations and exception messages. The lifecycle duration covers the probe call only; it excludes provider ranking, decision publication and backend creation.

### Validation status

Tests cover successful probe ordering relative to model loading and failed-probe privacy. CI must validate compilation and the selection/fallback behavior; no passing validation is claimed until the workflow completes.


## 21. GGUF engine shutdown cleanup reliability — implementation slice 8

Engine shutdown now detaches the loaded-backend registry and marks the engine uninitialized before invoking backend cleanup. It then attempts every captured backend, regardless of earlier close failures.

### Behavior

- Each backend receives its own `gguf.model.unload` lifecycle scope when execution observation is enabled.
- A backend close failure is retained as the primary failure, but does not prevent later backends from being closed or observed.
- Additional cleanup failures are attached to the primary failure as suppressed exceptions.
- The original RuntimeException or Error type is rethrown after all cleanup attempts.
- The no-op observer path uses direct backend close calls but follows the same best-effort cleanup semantics.
- Repeated `close()` calls are harmless after the backend registry has been detached.

### Validation

Tests cover an early backend failure followed by a successful cleanup, lifecycle STARTED/FAILED/SUCCEEDED outcomes, repeated close, and the no-op observer path. CI remains the authority for compile and test validation.


## 22. GGUF asynchronous inference lifecycle semantics — implementation slice 9

The GGUF `infer(...)` and `executeAsync(...)` methods schedule the existing synchronous `execute(...)` operation on Mutiny's worker pool. They do not introduce a separate model-execution boundary.

### Behavior

- The async `Uni` remains lazy: backend execution and the `gguf.infer` lifecycle scope begin only when a subscriber triggers execution.
- The synchronous execution boundary emits the single STARTED and terminal event pair. The async adapter must not create a duplicate scope or correlation ID.
- The lifecycle duration describes the backend execution performed by `execute(...)`, not time spent waiting in the worker-pool queue.
- Cancelling the async subscription is not currently a guarantee that an in-progress backend call is interrupted. Therefore the adapter must not report ABANDONED as if the backend operation had stopped; the synchronous operation records its eventual success or failure.
- A future cancellable async backend contract may add true cancellation observation, but it must preserve exactly-once terminal and active-accounting semantics.

### Validation

A delegation test verifies that async inference is lazy and emits exactly one correlated `gguf.infer` STARTED/SUCCEEDED pair. CI must validate the test on the supported JDK baseline.

## 23. GGUF model registration ownership — implementation slice 10

A model ID is a unique live registration key within one `GgufInferenceEngine`.

### Behavior

- `loadModel` rejects null or blank model IDs and rejects an ID that is already registered.
- Duplicate detection occurs before provider selection, readiness probing, backend creation or lifecycle observation.
- Duplicate registration never silently replaces or closes the currently registered backend.
- Callers must explicitly call `unloadModel(modelId)` before loading another backend under the same ID.
- The existing backend remains registered and usable after a rejected duplicate load.
- This slice deliberately does not add implicit hot replacement or claim safe concurrent unload while inference is in flight. A future replacement API must define draining/reference ownership before it can close an active backend.

### Validation

A focused test verifies that duplicate registration is rejected before provider creation, produces no model-load lifecycle event, preserves the original backend for inference, and closes it only on explicit unload. CI must validate the supported JDK baseline.
