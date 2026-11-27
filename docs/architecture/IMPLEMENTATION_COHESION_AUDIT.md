# Implementation Cohesion Audit

**Status:** Source-level triage snapshot  
**Reviewed:** 2026-11-27  
**Base:** `main` at the time of inspection  
**Scope:** Representative Java execution, backend, model-configuration and GGUF/accelerator integration classes

This is a prioritization document, not a claim that every large file is a god class. Generated bindings, numerical kernels and public facades can legitimately be large. Refactoring must follow responsibility boundaries, preserve semantics and be backed by focused tests.

## 1. Review method and limits

The initial pass used the repository tree and fetched source files to record source line counts, method inventories, fields, collaborators and visible responsibility groups. It did not execute Gradle tasks, inspect every call site, measure runtime behavior or establish that all listed files are actively used by published consumers.

Treat the findings as candidates for deeper call-graph and test-coverage review. Before changing a candidate, inspect its full dependency graph, callers, source-set ownership, public API compatibility, module build files and relevant tests.

Generated native/JNA/FFM binding declarations and mechanically generated files are excluded from god-class scoring. Kernel implementation size is not by itself evidence of poor cohesion.

## 2. Findings at a glance

| Candidate | Observed size | Initial classification | Priority |
|---|---:|---|---|
| `GgufMetalBridge.java` | 2,054 lines | Multiple accelerator operation families plus native discovery and decode state | High |
| `ModelConfig.java` | 2,519 lines | Public configuration contract mixed with format ingestion, metadata mapping and architecture resolution | High |
| `MetalComputeBackend.java` | 877 lines | Backend facade mixes operation forwarding, fallback policy, native dispatch and execution observation | Medium |
| `CpuBackend.java` | 1,467 lines | Broad backend implementation; already delegates several cohesive operation families | Medium / monitor |
| `AccelOps.java` | 2,301 lines | Dense optimized numerical kernel collection and native/vector dispatch | Review carefully; size alone is not a defect |
| `GGUFConverter.java` | 1,007 lines | Conversion orchestration, format detection, native FFM callbacks, paths and async lifecycle | Medium / verify ownership |

Line counts are a navigation signal, not a pass/fail threshold.

## 3. High-priority candidates

### 3.1 GGUF Metal bridge

**Source:** `engine/gguf/aljabr-gguf-java/src/main/java/tech/kayys/aljabr/gguf/javabackend/GgufMetalBridge.java`

Observed structure:
- 2,054 lines, static bridge API and substantial static mutable state.
- Reflective/MethodHandle discovery and initialization of the optional Metal binding.
- Quantized matvec variants, pair/triple/batch matvec, fused FFN operations and FlashAttention dispatch.
- Model buffer registration, normalization/RoPE registration, KV-cache synchronization and full decode-step operations.
- Thread-local reusable buffers and long-lived native arena/segment state.

**Why review it:** These concerns all cross the Java-to-Metal boundary, but they represent distinct operation families and lifecycle concerns. The class currently centralizes native discovery, operation dispatch and decode-session state. Static mutable native resources also make lifecycle, concurrency and test isolation important to verify.

**Safe decomposition direction (preserve existing facade):**
1. Extract an internal binding/discovery component that owns optional Metal loading and typed MethodHandle resolution.
2. Group matrix-vector and fused feed-forward operations behind cohesive internal delegates.
3. Move KV-cache, model registration and decode-step state into a scoped decode-session component with explicit initialization and close/reset semantics.
4. Keep `GgufMetalBridge` as a compatibility facade while callers migrate to narrower contracts.

Do not split by arbitrary method count. First map all call sites and establish whether static global state is intentionally process-wide or accidentally shared across model instances. Add tests for unavailable Metal, partial symbol availability, concurrent models, repeated initialization, reset/close and native-call failure before changing lifecycle ownership.

### 3.2 Model configuration contract

**Source:** `spi/aljabr-spi-model/src/main/java/tech/kayys/aljabr/spi/model/ModelConfig.java`

Observed structure:
- 2,519 lines and a large public bean-style configuration surface.
- JSON file loading and fallback extraction from nested `text_config`.
- GGUF metadata ingestion and key/value normalization.
- Architecture-specific defaults and reconciliation (including RoPE and shared-KV behavior).
- Numeric/list conversion helpers, resolved dimension calculations, architecture traits and compatibility getter aliases.

**Why review it:** The public model configuration contract is valuable and should not be fragmented casually. However, ingestion from different formats and architecture-specific interpretation are separate change drivers from the stable configuration value model.

**Safe decomposition direction (preserve public API and Jackson behavior):**
1. Keep `ModelConfig` as the public compatibility/value contract initially.
2. Extract JSON loading and nested-config merge behavior into a package-owned loader.
3. Extract GGUF metadata-to-config mapping into a dedicated mapper in the format-owning module where dependency direction permits; do not make the model SPI depend on GGUF.
4. Move architecture-specific resolution into focused internal resolvers or typed architecture metadata only where the semantics are genuinely reusable.
5. Retain current public factory/accessor methods as delegating compatibility entry points until downstream callers and serialization compatibility are proven.

Before extraction, inspect the full field inventory and Jackson annotations, constructor/default semantics, equality/serialization expectations, and all consumers. Existing tests include Gemma-4 text-config merge coverage in the SPI module, plus model-config tests in GGUF and Safetensors modules; consolidate a contract test matrix without coupling the SPI test suite to optional formats.

## 4. Medium-priority candidates

### 4.1 Metal compute backend

**Source:** `backend/metal/aljabr-backend-metal/src/main/java/tech/kayys/aljabr/backend/metal/MetalComputeBackend.java`

Observed structure:
- 877 lines and 67 declared methods.
- Implements the broad `ComputeBackend` contract.
- Owns runtime initialization/forced-CPU configuration, CPU fallback, output wrapping, host staging, native dispatch, fallback policy and execution observation.
- The constructor already separates runtime policy, per-operation fallback policy and observer configuration.

**Assessment:** This is a backend facade and broad interface implementation, so a large number of forwarding methods is expected. It is not an automatic god-class finding. The likely improvement is to keep policy and dispatch mechanics out of the facade.

**Safe direction:** Consider extracting operation dispatch/fallback reporting and native storage/staging mechanics behind package-private collaborators. Keep the public `ComputeBackend` implementation thin. Do not introduce a second backend-selection policy here; capability declarations and Aljabr provider contracts should remain the shared source of truth, while product-specific serving policy remains outside Aljabr.

Relevant existing seams include `MetalFallbackPolicy`, `MetalExecutionObserver`, `MetalRuntimePolicy` and `MetalTensorStorageAccess`. Reuse and assess these before adding abstractions.

### 4.2 CPU backend

**Source:** `backend/cpu/aljabr-backend-cpu/src/main/java/tech/kayys/aljabr/backend/cpu/CpuBackend.java`

Observed structure:
- 1,467 lines and 75 declared methods.
- Implements the broad `ComputeBackend` and `KVCacheAttentionBackend` contracts.
- Already delegates attention, spatial, reduction and normalization execution to dedicated collaborators.
- Pointwise operations reuse `CpuOps`; tensor storage/layout access is delegated to dedicated access helpers.

**Assessment:** The class is large, but there is evidence of ongoing decomposition. Avoid a wholesale rewrite. Review the remaining operation groups (broadcast/arithmetic, shape/indexing, activation/loss, embedding and transfer) for cohesive extraction opportunities only when the collaborator can own a meaningful set of behavior and tests.

Preserve allocation and tensor-storage ownership semantics. The injected output allocator, pool ownership and backend close behavior are part of the lifecycle contract and must not be duplicated across delegates.

### 4.3 GGUF converter

**Source:** `engine/gguf/aljabr-gguf-converter/src/main/java/tech/kayys/aljabr/converter/GGUFConverter.java`

Observed structure:
- 1,007 lines and 29 declared methods.
- Combines format detection/metadata inspection, conversion path resolution, Safetensors and native conversion paths, FFM callback creation, async/progress APIs and active-conversion cancellation state.

**Why review it:** Conversion orchestration and native callback/resource lifecycle have different failure and testing concerns from format detection and path policy.

**Safe direction:** Map which methods are compatibility API versus internal implementation. Consider extracting path/name resolution, format inspection and native FFM callback/session handling behind focused collaborators. Keep conversion orchestration responsible for coordinating them, not reimplementing their details. Preserve cancellation, progress, callback arena lifetime and error propagation.

## 5. Large files not to split mechanically

### 5.1 Safetensors `AccelOps.java`

The 2,301-line file includes vector/native symbol binding, matrix multiplication, fused gated half-precision linear kernels, shape-specific optimized paths, normalization, activation, image normalization and convolution/group normalization operations.

This is a dense kernel collection, not necessarily an orchestration god class. The relevant question is whether kernel families have independent ownership, dependencies and conformance suites. If splitting, group by operation family and retain a single canonical numerical implementation per operation/capability. Benchmark and compare against reference implementations before and after extraction; do not disturb hot paths merely to reduce file length.

### 5.2 Generated native bindings

Very large generated FFM/JNA declarations (for example generated llama headers) should be regenerated from their source schema/tooling rather than manually decomposed. Keep generated output clearly marked and isolate handwritten adapters from it.

## 6. Recommended implementation order

1. **GGUF Metal bridge:** map callers and native state lifetime; add/identify lifecycle and optional-provider tests; then extract internal collaborators behind the existing facade.
2. **ModelConfig:** establish compatibility/serialization contract tests; separate format ingestion from the public value model without introducing reverse module dependencies.
3. **Metal backend:** use existing policy and storage seams to isolate dispatch/fallback mechanics; retain the backend facade.
4. **GGUF converter:** separate format inspection and FFM callback/session lifecycle from conversion orchestration.
5. **CPU backend and AccelOps:** continue incremental operation-family decomposition only where it improves independent testing, ownership or reuse.

For each change:
- Search and document all callers before moving public or package-visible methods.
- Reuse existing contracts, storage helpers, validators and capability descriptors.
- Keep provider/model/format-specific code in its owning adapter.
- Preserve numerical semantics, fallback constraints, resource ownership and exception behavior.
- Add focused unit tests and provider/format conformance tests as appropriate.
- Run the owning module tests, then relevant downstream consumer tests.
- Do not claim performance improvements without reproducible benchmarks on the relevant hardware.

## 7. Validation status

This document records source inspection only. No Gradle build, test task, benchmark or hardware execution was performed for this audit. File sizes and method inventories are a snapshot and should be refreshed before implementation. The audit does not assert that every listed class violates SOLID; it identifies where deeper responsibility and lifecycle review has the highest expected value.
