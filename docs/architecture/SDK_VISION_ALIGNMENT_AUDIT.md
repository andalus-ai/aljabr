# Aljabr SDK Vision Alignment Audit

**Audit date:** 2026-11-28  
**Baseline:** `main` source as inspected on this date  
**Method:** targeted inspection of Gradle settings/root conventions, tensor API and implementation, storage contract, diagnostics API, existing capability inventory, distribution audit, research-readiness audit and tensor quickstart.  
**Validation:** source inspection only. No Gradle build, dependency report or tests were executed in this audit.

> This document is a current targeted audit, not a complete recursive source-tree inventory. Findings distinguish directly inspected behavior from hypotheses that require broader call-site or runtime validation.

## 1. Executive summary

Aljabr already has several important pieces of the intended developer experience:

- A public tensor abstraction with shape, dtype, device, layout and backend concepts.
- Explicit storage ownership documentation, retain/release semantics, views/layout metadata and idempotent release in the default eager tensor implementation.
- A dependency-free diagnostics artifact and structured diagnostic records.
- Independently selectable numerics, linear-algebra, tensor, NN and backend modules.
- Optional/experimental build switches for legacy autograd, HAT, storage providers and selected integrations.
- A Java research quickstart that demonstrates explicit compute-runtime and tensor resource lifecycles.

The principal gap is not absence of all foundations. It is that the foundations are not yet consistently surfaced as one coherent, tested SDK experience. Existing architecture and readiness documents also describe different repository snapshots and module paths; the source tree must remain the authority.

## 2. Evidence-backed current state

### 2.1 Tensor API and resource lifecycle — meaningful foundation

Inspected:
- `plugins/ml/aljabr-tensor/.../Tensor.java`
- `plugins/ml/aljabr-tensor/.../DefaultTensor.java`
- `plugins/ml/aljabr-tensor/.../Shape.java`
- `plugins/ml/aljabr-tensor/.../TensorStorage.java`
- `plugins/ml/aljabr-tensor/.../DefaultTensorTest.java`

Confirmed:
- `Tensor` is `AutoCloseable`; it exposes shape, dtype, device, backend and immutable layout metadata.
- `eval()` is documented as returning a borrowed tensor; `contiguous()` is documented as returning a newly owned independent copy.
- `storage()` is documented as returning borrowed storage; callers must retain before extending its lifetime.
- `TensorStorage` defines explicit retain/release ownership references independent of heap, FFM, host or device representation.
- `DefaultTensor` accepts a diagnostic sink for construction validation and uses an atomic release guard. It releases its owned gradient and storage once.
- Tests cover layout validation, storage capacity, ownership transfer on successful construction, ownership retention on failed construction, strided reads and resource release.

Risks / follow-up:
- The interface-level default `Tensor.release()` is a no-op. This preserves compatibility for implementations without owned resources, but a custom implementation that owns resources and forgets to override it can silently leak. Audit every implementation and decide whether to add an explicit lifecycle capability/contract or make release mandatory in a compatibility-aware major-version plan.
- Ownership rules are documented, but must be audited consistently across all operations: which return owned tensors, borrowed views, aliases or copies; whether view storage is retained; and what happens on backend failure.
- The current quickstart uses manual `finally` cleanup. Add examples for try-with-resources and scoped resource composition where the API supports it.
- Do not claim end-to-end zero-copy merely because storage is off-heap or a view is created. Instrument and document each conversion/transfer boundary.

### 2.2 Diagnostics — reusable base exists

Inspected:
- `core/aljabr-diagnostics/build.gradle.kts`
- `core/aljabr-diagnostics/.../Diagnostic.java`
- Tensor construction validation in `DefaultTensor`.

Confirmed:
- The diagnostics artifact declares no runtime dependencies.
- `Diagnostic` is immutable and structured around stable code, severity, category, message, explicitly supplied context and timestamp.
- Context values are strings and are copied into an immutable sorted map.
- Tensor construction can report validation failures to a supplied `DiagnosticSink`; existing constructors remain silent/source-compatible.

Risks / follow-up:
- The diagnostics API is currently demonstrated in tensor validation, not yet proven to be the consistent error/diagnostic surface across backends, model loading, format conversion, memory allocation, graph compilation and provider selection.
- Define the relationship between thrown typed exceptions and emitted diagnostics. Diagnostics should enrich failures, not replace exceptions or allow execution to continue after invalid state.
- Ensure diagnostic context remains bounded and does not automatically capture tensor contents, prompts, weights, credentials or other sensitive payloads.

### 2.3 Build and module graph — modular direction, global build coupling remains

Inspected:
- `settings.gradle.kts`
- Root `build.gradle.kts`
- `plugins/ml/aljabr-tensor/build.gradle.kts`
- Existing `GRADLE_DISTRIBUTION_AUDIT.md`.

Confirmed in current settings/build excerpts:
- Gradle discovers projects under core, SPI, engine, tools, integration, plugins, backend and model areas.
- Legacy plugin core, selected storage adapters, legacy autograd, experimental HAT and legacy accelerator integrations have explicit opt-in switches.
- Diffusion has a dedicated nested model-subsystem discovery path.
- Tensor is currently located at `:plugins:ml:aljabr-tensor` and depends on `:core:aljabr-diagnostics`; CPU backend is test-only in its module build file.
- The root build still applies broad Java toolchain/compiler/test conventions and global dependency substitution/version resolution rules. The prior distribution audit identified global preview/vector flags, Jandex work, publication metadata, Tafkir substitution and Maven Local as items requiring executable validation and scoped remediation.

Risks / follow-up:
- Older docs refer to `:core:aljabr-tensor`, while current settings/build substitution and source evidence use `:plugins:ml:aljabr-tensor`. Reconcile stale module paths in documentation and examples.
- A BOM aligns versions; it must not become a transitive all-modules aggregator.
- Verify actual resolved runtime graphs and published POM/module metadata using clean external Gradle and Maven consumers before claiming profiles are minimal.
- Keep preview/incubator/native-access flags scoped to modules that require them; verify with Gradle tasks before changing the root conventions.

### 2.4 Scientific and ML breadth — existing foundations, important validated gaps

The earlier capability inventory records tensor operations, CPU SIMD kernels, NN layers/losses, tokenizer contracts and a broad model catalogue. It also records missing or partial dedicated capability families such as statistics/inference, general linear-algebra decompositions/solvers, classical ML estimators, scientific solvers and time-series methods.

The current Gradle settings show a growing separation between `core`, `plugins/ml`, `engine`, `backend` and `models`, but module existence or model catalogue size is not proof of algorithm coverage or supported execution.

Required follow-up:
- Refresh the capability inventory from current source and Gradle paths.
- Mark every capability as declared, implemented, tested, validated, stable and optimized separately.
- Do not create a new module for each algorithm. Group cohesive capability families and keep optional adapters independent.

### 2.5 Training boundary — keep foundational work in Aljabr

The research-readiness audit documents legacy autograd code excluded from default builds, stale trainer API dependencies, incomplete gradient capability signaling and optimizer paths that do not yet provide reliable collection-based parameter updates.

This is still the right ownership split:
- Aljabr: tensor, autograd, differentiable operations, NN/parameter contracts, losses, optimizer algorithms and reusable training computation.
- Tafkir: training loops/workflows, dataset orchestration, experiment management and fine-tuning/adaptation.
- AQLI: inference serving and deployment.

Before advertising training as supported, repair build visibility and optimizer/gradient semantics and run deterministic convergence tests against the CPU reference.

## 3. Priority backlog

| Priority | Work item | Why now | Acceptance evidence |
|---|---|---|---|
| P0 | Reconcile documentation and module paths against current Gradle settings | Existing docs contain stale `:core` tensor paths while current source is under `:plugins:ml` | Links and commands resolve; CI runs documented tasks |
| P0 | Tensor ownership/lifecycle conformance audit | Ownership is a differentiator and a source of severe FFM/off-heap defects | Every implementation/operation documents owned/borrowed/view/copy behavior; tests cover close, aliasing and failure paths |
| P0 | Establish one diagnostic/error contract | A diagnostics artifact exists but consistency across layers is unproven | Stable codes, typed exceptions, context policy and tests across tensor/backend/model/format boundaries |
| P1 | Reconcile public API styles | Fluent, traditional and declarative experiences must wrap one semantic core | Examples for explicit and fluent use compile and produce equivalent results; annotations remain optional |
| P1 | Publish a real minimal dependency/profile matrix | Modular source layout is not enough | Clean external consumers and dependency reports for numerics, tensor+CPU, ML and multimodal profiles |
| P1 | Developer tooling baseline | Profiling/inspection should not be an afterthought | Tensor/model inspection, backend explanation, memory/copy trace and operation profiling prototype |
| P1 | Model/algorithm extension testkit | External extension ease is a core platform differentiator | External sample project registers an extension and passes conformance suite |
| P2 | JShell/JBang/Jupyter compatibility matrix | All environments should use the same published APIs | Reproducible tested examples for each entry point and version |
| P2 | Scientific/data capability gap closure | Broad scientific scope must be built on reusable foundations | Prioritized statistics, probability, datasets and solver milestones with reference tests |
| P2 | Multimodal/diffusion capability matrix | Diffusion and modalities must be first-class but truthful | Per-model/operator/backend capability declarations and executable reference tests |

## 4. Recommended next implementation increment

Do not start by redesigning the entire Tensor API or creating a new SDK facade. The existing tensor contract already contains useful ownership documentation and the default implementation has a release guard.

The next code increment should be a **Tensor Lifecycle and Ownership Conformance** workstream:

1. Inventory every concrete `Tensor` implementation, wrapper, view, lazy node and backend-produced tensor.
2. Classify each tensor-producing method as newly owned, borrowed, view/alias with retained storage, or materializing copy.
3. Audit `eval()`, `contiguous()`, `reshape()`, `slice()`, `to()`, `cast()`, `grad()`, `setGrad()`, `storage()` and all backend operations for matching implementation behavior.
4. Test close/release idempotency, use-after-release failure, alias survival after source release, storage retain/release balance, failure during construction/operation and gradient ownership.
5. Decide the compatibility strategy for the no-op default `Tensor.release()`; do not silently break external Tensor implementations without an API-versioning plan.
6. Add a concise ownership table and try-with-resources examples to the public Tensor API guide.
7. Run the focused tensor tests and full relevant CPU-backed suite; record the exact JDK/Gradle commands and results.

This is the best first implementation increment because memory safety and lifecycle clarity underpin tensors, datasets, models, zero-copy interop, multimodal pipelines and accelerator backends.

## 5. Guardrails for subsequent work

- Keep API semantics provider-neutral and diagnostics Aljabr-owned.
- Keep third-party types out of stable public APIs; use explicit adapters.
- Keep pure Java/reference paths available as correctness oracles where practical.
- Keep native, accelerator, notebook and dataset connectors opt-in.
- Support traditional, fluent, functional and annotation-assisted styles over the same semantic contracts.
- Make multimodal and diffusion first-class model families without putting their domain semantics into generic Tensor APIs.
- Make scientific math/statistics/data capabilities independently consumable.
- Do not equate source presence, module inclusion, provider declaration or model catalogue entries with validated support.
- Do not claim zero-copy, deterministic execution, backend parity, tree-shaking or minimal packaging without executable evidence.
