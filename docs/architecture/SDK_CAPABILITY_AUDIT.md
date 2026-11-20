# Aljabr SDK Capability and API Audit

> **Status:** Initial repository inventory and source-level review  
> **Baseline:** `main` at the time of this audit (2026-11-21)  
> **Purpose:** Translate the Java-native scientific + AI platform vision into an evidence-based SDK improvement sequence.  
> **Important:** This is not a certification of runtime correctness or production readiness. A module, class, model-family plugin or test file existing in the tree does not prove an end-to-end capability is complete.

## 1. Executive assessment

Aljabr already has substantial foundations: a backend-neutral tensor storage contract, reference-counted storage, FFM arena management, eager and lazy tensor paths, autograd and optimizer modules, CPU/Metal/CUDA/HAT provider projects, a model-family plugin SPI, a large architecture catalogue, format-specific GGUF/Safetensors modules, a Java linear-algebra module, and a modular Gradle build.

The main opportunity is **coherence and productization**, not a wholesale rewrite. The repository has many useful building blocks, but the developer-facing contract is spread across a large number of independently evolved modules. The next work should converge these into one discoverable, documented, tested programming model, with clear capability boundaries and explicit support status.

### Initial findings

| Area | Repository evidence | Initial assessment |
|---|---|---|
| Tensor foundation | `plugins/ml/aljabr-tensor`: `Tensor`, `Shape`, `DType`, eager/lazy execution, tests | Strong base exists; API completeness and cross-provider semantics need a systematic audit |
| Memory and FFM | `TensorStorage`, `ManagedArena`, storage views, allocators/pools and lifecycle tests | Ownership work is real; consolidate contracts and verify race, alias and async-transfer cases |
| Autograd/training primitives | `aljabr-autograd`, `aljabr-autograd-nn`, `aljabr-optim`, `aljabr-nn` | Foundational modules exist; legacy `plugins/ml/autograd` is separately excluded by default and overlaps training/data/metrics concepts |
| Linear algebra | `aljabr-linear-algebra` with decomposition implementations and tests | Useful scientific foundation; relationship to tensor operations and numerical conventions needs clarification |
| Model extensibility | `ModelFamilyPlugin`, descriptor/capability/validation/registry contracts and 261 model Gradle projects | Broad extension ecosystem; discoverability, compatibility matrix and tested execution coverage need stronger product-level presentation |
| Multimodal | model SPI modality types, multimodal request/processor contracts, many modality model projects | Contracts exist; architecture-level capability declarations must be distinguished from end-to-end supported pipelines |
| Diffusion | `StableDiffusionPipeline` appears in model-common | A diffusion-related implementation exists, but no dedicated diffusion/generative module appeared in the initial path inventory; do not claim complete diffusion support |
| Formats | GGUF and Safetensors engine trees, converters, loaders and quantization | Strong specialized format coverage; ensure formats adapt into one canonical tensor/model execution path |
| Scientific statistics/data | no dedicated statistics/probability/dataframe module found by the initial filename inventory; dataset abstractions appear in clustering and legacy autograd | Visible gap in discoverable general-purpose statistics/probability/dataset APIs; validate deeper source inventory before designing new modules |
| Developer tools | model profiler class and benchmark/test utilities exist | Tooling is scattered; no obvious cohesive inspector/profiler product module in the initial inventory |
| Interactive workflows | README describes JBang/JShell; no JBang or notebook module found in initial module/path inventory | Likely documentation/integration gap, not necessarily a missing runtime module |
| Distribution | 315 Gradle build files in recursive repository tree; root settings dynamically discovers projects and has explicit opt-in exclusions | Highly modular codebase, but consumer dependency profiles/BOM/starter experience and external-consumer verification need dedicated audit |

## 2. Important architecture observations

### 2.1 The repository is already large and modular

The recursive tree contained 3,013 files and 315 Gradle build files (including root/buildSrc and nested example builds). There are 261 model-family Gradle projects, 17 engine projects, 10 backend projects, 7 SPI projects, 6 core projects and 9 projects under `plugins/ml`.

This scale makes navigation and dependency clarity essential. Avoid introducing a new module for every concept before checking whether an existing artifact can be made coherent and independently consumable.

The model catalogue should remain optional. A normal numerical/tensor consumer must not resolve hundreds of model-family projects or native provider dependencies.

### 2.2 Build inclusion currently encodes migration state

Root `settings.gradle.kts` recursively discovers projects and has explicit opt-in properties for legacy/experimental integrations, including:
- `withLegacyPluginCore`
- `withAutograd`
- `withBlackwell`
- `withLegacyCudaKernelPlugin`
- `withLegacyRocmKernel`
- `withHat` (requires a separate Babylon JDK path)

This is a useful transitional mechanism, but it means the distinction between **optional product capability**, **experimental provider**, and **temporarily excluded legacy module** must be visible to maintainers and consumers. These are different reasons for exclusion and should not be collapsed into a generic "optional" label.

The root build also configures Java 25 toolchains and incubator Vector API flags for selected projects. Public API compatibility and runtime requirements should be documented per artifact/profile, rather than assuming every consumer needs every compiler/runtime flag.

### 2.3 Tensor is a real foundation, but its public boundary deserves focused review

The current `Tensor` interface already offers shape, device, dtype, storage, arithmetic, reshape/slice, activation/reduction, device transfer, autograd and resource release. `Shape` defensively copies dimensions and checks element-count overflow. `TensorStorage` defines backend-neutral capacity and retain/release ownership. `DefaultTensor` validates storage capacity and owns a storage reference. This is a solid base to preserve.

Focused review items:
1. `Tensor.backend()` exposes `ComputeBackend` directly. Decide whether backend identity/diagnostics belong in a separate execution/device capability view so ordinary tensor consumers depend on semantic operations rather than backend implementation APIs.
2. `DType` currently combines ordinary numeric types and block-quantized storage types, and includes both `I8` and `INT8`. Review canonical naming, dtype promotion, arithmetic support, quantized layout/block metadata and whether storage encoding should be distinct from compute dtype.
3. Public scalar overloads use `float`; review double/long scalar ergonomics, typed scalar promotion and consistency with tensor dtype.
4. The API includes `release()` and `close()` semantics. Audit idempotence, ownership transfer, borrowed gradients, view lifetimes and use-after-close behavior consistently across all implementations.
5. Shape/layout/strides/offsets and broadcasting semantics should be explicit. Do not infer layout from shape or device.
6. Clarify whether operations are eager, lazy or provider-deferred and how evaluation/materialization failures are surfaced.

Do not change these APIs based on this inventory alone. First map every implementation and downstream use, then propose compatible, test-backed evolution.

### 2.4 Memory ownership is a differentiator worth standardizing

The existing `TensorStorage` contract is backend-neutral and `ManagedArena` provides reference-counted FFM lifetime management. This aligns strongly with the vision. The next audit should trace all storage implementations and adapters, including CPU, Metal, Safetensors and KV-cache paths.

Required contract matrix:
- Owned, borrowed and shared references.
- Retain/release balance and double-close behavior.
- Views/slices and parent lifetime independence.
- Host-accessible versus device-only storage.
- Arena scope/thread constraints.
- Async operation/transfer completion and resource retention.
- Mutation/aliasing and read-only views.
- Failure and cancellation cleanup.
- Copy/materialization/transfer observability.

Zero-copy claims should be attached to a specific boundary and conditions, not to a model or backend as a whole.

### 2.5 Model catalogue size is not model execution coverage

The 261 model projects and the SPI's rich capability/report types are a major extension foundation. However, declared capabilities, architecture adapters, format support, tokenizer availability, operator coverage, provider support and validated end-to-end execution are separate facts.

The developer-facing model support report should distinguish at least:
- Metadata/config recognition.
- Weight format loading.
- Tokenizer/preprocessor availability.
- Architecture construction.
- Required operator availability.
- Modality pipeline completeness.
- Backend/provider compatibility.
- Numerical correctness fixture status.
- End-to-end execution status.
- Production suitability (owned by deployment/runtime validation, not a model-family label alone).

Avoid inferring support from a plugin descriptor alone.

## 3. Vision-to-repository gap assessment

### P0 — Establish canonical SDK contracts

- Produce a complete public API inventory by artifact/package and identify duplicate concepts.
- Define the canonical Tensor, Shape, DType, Layout/strides, Device, Storage, ownership and execution contracts.
- Separate semantic tensor APIs from backend/provider inspection.
- Define lifecycle and ownership rules for all closeable resources.
- Define dtype promotion, scalar conversion, broadcasting, indexing and view/copy behavior.
- Establish compatibility policy before changing public signatures.

### P1 — Complete foundational scientific and ML capabilities

- Audit numerical and linear algebra APIs for precision, stability, error behavior and tensor interoperability.
- Identify the missing general statistics/probability/descriptive-statistics contracts.
- Define general dataset/source/transform/batch abstractions without pulling in Tafkir orchestration.
- Complete reusable autograd, loss, optimizer, scheduler and gradient utilities within Aljabr; keep training loops and experiments in Tafkir.
- Define a consistent error/diagnostic model.

### P2 — Model and algorithm developer experience

- Simplify the extension path for model family, algorithm, operator, format adapter and backend.
- Provide a model/algorithm development template and conformance test kit.
- Generate a machine-readable capability matrix from tested fixtures, not declarations alone.
- Separate model runner composition from AQLI serving.
- Make multimodal composition a shared architecture capability.
- Define dedicated diffusion primitives and a diffusion pipeline contract without coupling them to token generation.

### P3 — Developer tooling and workflows

- Consolidate profiler, graph inspector, tensor inspector, model inspector and numerical diagnostics.
- Add copy/transfer/memory visibility and actionable backend-selection explanations.
- Provide JBang and JShell smoke-tested examples.
- Document Jupyter integration through an established Java kernel (e.g. IJava), including resource cleanup and dependency setup.
- Ensure examples work from a published artifact repository, not only a composite build.

### P4 — Packaging and distribution

- Audit all `api`, `implementation`, `runtimeOnly`, project and external dependencies for each published module.
- Verify `java-library` usage and generated Gradle module metadata/POMs, especially for public API types.
- Create a BOM only after the artifact graph and version policy are stable.
- Define dependency recipes for numerics, scientific/statistics, tensor, ML, multimodal, training-primitives, CPU, accelerators, research and embedded/minimal.
- Verify each profile in clean external Gradle and Maven fixture projects.
- Treat jlink/JPMS, GraalVM Native Image and shrinkers as later packaging optimizations, not substitutes for modular dependencies.

## 4. Recommended first implementation sequence

Do these in small independent changes:

1. **Inventory only:** generate a machine-readable artifact/API inventory and dependency graph; identify duplicate public types and package overlaps.
2. **Tensor contract audit:** trace all implementations, tests and consumers of Tensor/Shape/DType/storage/backend APIs. Produce a compatibility-preserving proposal; do not redesign in place yet.
3. **Memory contract tests:** add missing lifecycle, aliasing, retain/release, asynchronous and failure-path tests where source inspection demonstrates gaps.
4. **Public module metadata:** correct/verify Java library publication and API dependency exposure; add external consumer fixtures.
5. **Developer quickstarts:** provide one canonical numerical, tensor, autograd, model extension and multimodal example that works from a published artifact.

The first code change should be selected from evidence gathered in steps 1–2, not from a guessed module structure.

## 5. Audit rules and status vocabulary

Use these labels consistently in docs and capability reports:

- **Present:** code or contract exists.
- **Implemented:** a concrete implementation exists.
- **Tested:** focused tests exercise the implementation.
- **Validated:** correctness and integration tests pass for the stated scope.
- **Supported:** documented user-facing workflow and compatibility commitment exist.
- **Experimental:** usable but not covered by the stable support commitment.
- **Planned:** accepted direction with no complete implementation.
- **Legacy:** retained for migration/compatibility, not a preferred path.

A file/class/module/test existing is not enough to mark a capability supported. A capability is supported only when its documented scope, dependencies, compatibility, tests and developer workflow are all clear.

## 6. Non-goals

- Do not rewrite the model catalogue or create a new module for every proposed concept.
- Do not force every Aljabr consumer to depend on tensors, neural networks, model plugins or accelerators.
- Do not make Aljabr depend on AQLI or Tafkir.
- Do not move serving, request scheduling or deployment lifecycle into Aljabr.
- Do not move Tafkir's higher-level training orchestration into Aljabr.
- Do not promise zero-copy, deterministic execution, cross-backend parity or tree-shaking without scoped evidence.
- Do not change public APIs until all implementations and consumers are mapped.
