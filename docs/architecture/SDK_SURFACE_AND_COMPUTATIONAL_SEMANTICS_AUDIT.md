# Aljabr SDK Surface and Computational Semantics Audit

**Audit date:** 2026-11-20  
**Audited ref:** `main` at tree `b95aa769cc4c9a6efce462c28cc37f15f0f23ea3`  
**Method:** Git tree inventory and static inspection of Gradle settings/build scripts, representative tensor/memory APIs, ML modules, provider SPI, and existing architecture/distribution inventories.  
**Validation:** No Gradle build, dependency resolution, test suite, Javadoc generation, or external-consumer fixture was executed.

> This is a source-level SDK audit, not a complete behavioral or binary-compatibility certification. Findings are marked as observed facts, risks to validate, or recommendations. It deliberately avoids proposing a replacement Tensor API before the existing semantics are mapped.

## 1. Executive summary

Aljabr already contains meaningful foundations for the intended Java-native scientific and AI framework:

- Separate numerical, linear-algebra, tensor, autograd, NN, optimizer, backend, model-family and format modules exist.
- The provider SPI has explicit descriptors, compatibility metadata, discovery and provider/session lifecycle concepts.
- Tensor storage has backend-neutral ownership/reference-counting contracts and allocator abstractions.
- Tensor, eager/lazy execution and reverse-mode differentiation have tests.
- Model families are independently packaged and registered through Java service metadata.
- There is an external provider fixture and existing documentation audits for capability scope and Gradle distribution.

The main problem is not an absence of capability. It is that the public developer experience and module boundaries are not yet consistently presented as one coherent scientific-computing SDK.

### Highest-priority findings

1. **The Tensor surface is broad and backend-coupled.** `Tensor` exposes many operations and directly returns `ComputeBackend`, mixing mathematical values, execution implementation access, autograd, training helpers and device transfer in one public contract.
2. **Memory ownership is more advanced than the public ergonomics.** `TensorStorage` and `TensorAllocator` define retain/release and ownership transfer, but safe scoped composition, view/alias contracts and copy/transfer observability need a systematic audit.
3. **Module publication/API boundaries need attention.** Several modules use Gradle's `java` plugin rather than `java-library`; public signatures and transitive API dependencies need verification. Root-wide build policy and publication risks are already documented in `GRADLE_DISTRIBUTION_AUDIT.md`.
4. **The framework is already modular but not yet organized around the whole scientific domain.** Numerics and linear algebra exist; statistics, probability/distributions, datasets and classical ML need dedicated capability inventories before adding modules.
5. **Model catalogue scale is significant.** The repository tree contains 261 model-family Gradle projects. This is useful breadth, but raises configuration, publication, documentation and support-status concerns. A model plugin's existence must not imply validated execution support.
6. **Developer entry points and tools are not yet a unified product experience.** JShell, JBang and Jupyter should be documented and tested as consumers of published artifacts; diagnostics, profiling and inspection need a common roadmap and discoverable API.
7. **No single validated consumer profile/BOM story was proven by this audit.** Existing module separation is a good start, but dependency graph, published POM/Gradle metadata, BOM behavior and minimal packaging require clean external-consumer tests.

## 2. Repository shape and capability evidence

The recursive Git tree contains **313 Gradle build files** and **261 directories matching `models/aljabr-model-*`**. These counts describe checked-in build files, not necessarily the number of projects included in every Gradle invocation: `settings.gradle.kts` has conditional exclusions and dynamic project discovery.

Observed major areas:

| Area | Observed foundation | Audit interpretation |
|---|---|---|
| `plugins/ml/aljabr-numerics` | Tolerance, precision, random source, numerical checks | Useful minimal foundation; no production dependencies declared in its module build file |
| `plugins/ml/aljabr-linear-algebra` | Matrix/vector and decompositions | Depends on numerics; current module is a focused starting point |
| `plugins/ml/aljabr-tensor` | Tensor, shape, dtype, eager/lazy ops, memory abstractions | Central AI/scientific compute surface; needs semantic/API review |
| `plugins/ml/aljabr-autograd` | Autograd context and variable | Present as a separate module; default settings exclude only legacy `plugins/ml/autograd`, not this module |
| `plugins/ml/aljabr-nn` | Layers, modules, parameters, attention, losses | Useful NN primitives; verify public dependency metadata |
| `plugins/ml/aljabr-optim` | SGD, AdamW, Adagrad, Adafactor and others | Reusable training primitives belong in Aljabr |
| `engine/clustering` | Clustering API and algorithms | Early scientific/classical-ML capability |
| `core/aljabr-core` | Memory/runtime and KV cache infrastructure | Avoid allowing inference-specific concerns to define the minimal scientific core |
| `spi/*` | Provider, compute, model, tokenizer, plugin and testkit contracts | Preserve existing SPIs; avoid duplicating provider abstractions |
| `backend/*` | CPU, Metal, CUDA, HAT and other integrations | Optional provider model is directionally correct; support must be capability-specific |
| `engine/gguf`, `engine/safetensor` | Format APIs, loaders, converters and bridges | Keep formats as adapters, not tensor API owners |
| `models/*` | 261 model-family projects | Catalogue breadth; validate implementation maturity independently |

The existing `COMPUTATIONAL_CAPABILITY_INVENTORY.md`, `GRADLE_DISTRIBUTION_AUDIT.md` and `ALJABR-ROADMAP-IMPLEMENTATION-STATUS.md` contain deeper related findings. This document consolidates the SDK implications rather than duplicating those inventories.

## 3. Tensor API and semantics findings

### 3.1 Observed public contract

The inspected `Tensor` interface currently includes:
- Shape, device, dtype, backend and optional storage access.
- Arithmetic, scalar operations, matmul, reshape, slicing, concatenation and reductions.
- Activations, normalization, convolution and other neural-network-oriented operations.
- Device transfer and casting.
- Reverse-mode differentiation, gradient access and mutation.
- Explicit resource release through `AutoCloseable`.

The API also documents important ownership details: `backward(Tensor)` borrows its gradient, `grad()` returns a borrowed reference, `setGrad` transfers ownership, and `close()` releases owned resources.

### 3.2 Risks and design questions

- **Backend leakage:** `Tensor.backend()` exposes `ComputeBackend` through the core mathematical object. Determine whether this is required for public consumers or should be an advanced execution-context/diagnostic capability.
- **Contract breadth:** generic tensor arithmetic and neural-network-specific operations coexist on one interface. Evaluate whether operation groups should be capability interfaces or extension functions while preserving a simple common Tensor entry point.
- **Device and placement:** clarify whether `device()` is logical placement, physical device identity or memory residency. Define transfer completion and asynchronous semantics.
- **Shape/layout:** inspect whether shape alone adequately expresses strides, offset, contiguity, memory order and view semantics. Do not introduce a second shape model without evidence.
- **DType and promotion:** document scalar promotion, mixed-dtype operation rules, accumulation precision, overflow and unsupported dtype behavior.
- **Mutation:** mathematical immutability of shape/dtype is not equivalent to immutable contents. Define mutation, in-place operation, parameter updates and alias visibility.
- **Error model:** standardize shape mismatch, dtype mismatch, unsupported operation, device mismatch, closed-resource and allocation failures.
- **Autograd:** define graph retention, repeated backward, gradient accumulation/zeroing, detach/no-grad behavior, graph release and thread safety.
- **Ownership and views:** the Tensor documentation says derived tensors retain their own storage references and may outlive parents. Verify this invariant for every view-producing operation and failure path.

### 3.3 Recommended direction

Do not replace `Tensor` wholesale. First produce a behavior matrix for every public operation and then evolve the API compatibly:

1. Keep a compact, discoverable Tensor core for shape/dtype/device, common arithmetic, indexing/views and reductions.
2. Group optional capabilities (autograd, differentiable ops, NN, quantization, sparse, distributed) behind focused APIs/modules where this improves dependency isolation.
3. Keep backend/provider selection outside ordinary tensor expressions; provide an explicit execution context or policy API for advanced users.
4. Add fluent conveniences as facades over the same operation contracts, not a parallel tensor engine.
5. Make shape, dtype, layout, device, ownership and execution failures diagnosable before dispatch and explain backend failures after dispatch.

These are target directions, not approved public signature changes.

## 4. Memory ownership and zero-copy findings

### Observed strengths

- `TensorStorage` is backend-neutral and defines capacity plus explicit retain/release.
- `TensorAllocator` defines allocation ownership transfer and an aligned allocation extension point.
- `DefaultTensor` documents ownership transfer on successful construction and retains caller ownership on validation failure.
- The tensor interface exposes optional storage rather than requiring every implementation to expose host memory.
- Storage can represent heap, native, mapped or device memory without assuming a `MemorySegment`.

### Gaps to validate

- A public ownership vocabulary: owned, borrowed, retained, shared, view, read-only and transferred.
- Whether storage reference counting is safe under concurrent retain/release and how use-after-release is diagnosed.
- Whether all view operations retain storage and carry offset/strides correctly.
- Whether allocator and arena close semantics can invalidate outstanding tensors safely or report misuse.
- Whether host access and device access are separate capabilities with explicit synchronization.
- Whether zero-copy adapters expose alignment, lifetime, mutability and aliasing constraints.
- Whether copy/materialization/host-device transfer events can be inspected and profiled.

### Recommendation

Create a memory contract/conformance suite before adding more allocator implementations. Require each storage provider to test retain/release balance, idempotent tensor close, failed construction, views outliving parents, alias behavior, concurrent ownership where supported, and access after release. Add explicit copy/transfer inspection later as a separate diagnostics capability.

## 5. Module and dependency boundary findings

Representative module build files show:

- Numerics declares no production dependency.
- Linear algebra uses numerics as an implementation dependency.
- Tensor declares no production dependency in its module build file; CPU backend is a test dependency.
- Autograd exports tensor using `java-library` and `api`.
- NN declares tensor as `implementation`.
- Optimizer exports NN and tensor with `api`.

This is not automatically wrong, but it needs public-signature verification. If a public class or method exposes a type from another artifact, consumers may need that dependency on their compile classpath. The Java plugin choice and dependency configuration should express the intended public API surface.

### Required audit

For every public artifact:
1. List public/protected signatures and referenced external/project types.
2. Verify `api` versus `implementation` configurations.
3. Generate and inspect published POM and Gradle module metadata.
4. Compile an external consumer against the published artifact alone.
5. Ensure test-only CPU/native providers do not become production dependencies.
6. Ensure optional model catalogues, format adapters, native runtimes and accelerators are not pulled into minimal consumers.

Do not change all `java` plugins to `java-library` mechanically; only modules with intentional exported dependency contracts should use `api`.

## 6. Scientific framework breadth

The repository already has numerical, linear algebra, clustering, tensor and 3D foundations. The existing capability inventory identifies gaps or incomplete foundations around statistics, probability/distributions, statistical inference, regression/classical ML, dataset pipelines, NLP vectorization and scientific simulation.

Recommended sequencing:

1. Stabilize numerical semantics and reference-test conventions.
2. Complete core linear algebra and solver inventory.
3. Define statistics/probability contracts and a small reference implementation set.
4. Define general dataset/source/transform/batch contracts, separate from Tafkir training orchestration.
5. Consolidate classical ML estimators, preprocessing and evaluation under clear reusable contracts.
6. Add scientific domains (time-series, signal processing, simulation, Bayesian, survival, causal) only when there is a concrete API boundary and test suite.

Avoid a module-per-algorithm pattern. Prefer cohesive domain modules with small, composable public interfaces and independently selectable dependencies.

## 7. Model, multimodal and generative ecosystem

The model catalogue is already very large. The next improvement is not simply adding more model-family modules; it is making extension and maturity discoverable.

For every model family, distinguish:
- Metadata/configuration support.
- Weight-format support.
- Tokenizer/preprocessor support.
- Architecture implementation completeness.
- Operation coverage.
- Backend/provider compatibility.
- Numerical conformance status.
- Integration and example status.

A module existing or registering a `ModelFamilyPlugin` is not proof that a checkpoint can execute end-to-end.

Diffusion should be represented as a model/generative capability family, with reusable noise schedules, conditioning, samplers/solvers and latent tensor workflows. Multimodal capability metadata should describe modality combinations and actual encoder/fusion/decoder support, not only a generic multimodal flag.

## 8. Developer experience gaps and opportunities

### One semantic core, multiple Java styles

Support traditional Java, fluent builders, functional composition and annotation-assisted/generated metadata as complementary entry points. Avoid separate execution semantics or hidden reflection-heavy behavior.

### JShell, JBang and Jupyter

Use the same published artifacts and APIs. Establish smoke-tested examples:
- JShell: dependency/classpath setup, imports, a small numerical/tensor task, and explicit resource cleanup.
- JBang: self-contained script with dependency declarations and an example model/algorithm extension.
- Jupyter: established Java kernel (e.g. IJava), kernel setup, repeatable dependency installation and notebook-to-project promotion.

Do not make a custom notebook kernel a prerequisite.

### Diagnostics and inspection

Prioritize shared primitives that work across scientific and AI capabilities:
- Shape/dtype/layout validation and readable mismatch reports.
- Numerical assertions, tolerances, NaN/Inf checks and gradient checking.
- Model/parameter/algorithm inspection.
- Provider/capability explainability.
- Memory ownership, allocation, copy and transfer tracing.
- Operation-level profiling and JFR integration.

### Extension author experience

Create a small extension template and test kit for a model family, algorithm, dataset adapter and compute provider. Document lifecycle, capability descriptors, registration, compatibility, errors, optional dependencies and examples.

## 9. Distribution and build findings

The existing `GRADLE_DISTRIBUTION_AUDIT.md` already records root-wide toolchain/preview/vector flags, Jandex task wiring, publication metadata concerns, broad project inclusion, global resolution rules and Maven Local behavior. This SDK audit adds the following acceptance requirements:

- A BOM aligns versions only; it must not pull every module.
- Feature artifacts remain the actual dependency-selection boundary.
- Consumer profiles are documented dependency recipes or curated starters, not a second runtime mode system.
- Test external consumers using published artifacts, not only composite-build substitution.
- Keep provider/model catalogues/native runtimes opt-in.
- Treat JPMS/jlink, GraalVM Native Image and shrinkers as later packaging optimizations with explicit ServiceLoader, reflection, FFM and resource tests.
- Measure resolved dependency graphs and packaged outputs; do not claim tree-shaking from a BOM or from source module count alone.

## 10. Prioritized implementation backlog

| Priority | Increment | Deliverable | Gate |
|---|---|---|---|
| P0 | Tensor semantics inventory | Operation-by-operation shape/dtype/layout/mutation/device/error matrix | Source and tests mapped; no API redesign yet |
| P0 | Memory contract conformance | Ownership/view/lifetime/reference-count test suite | CPU storage implementation passes |
| P0 | Public API dependency audit | API-signature to Gradle dependency map | External consumer compiles against published artifacts |
| P1 | Tensor API boundary proposal | Small core plus optional capability interfaces proposal | Compatibility analysis and representative usage examples |
| P1 | Numerical contract baseline | Promotion, tolerance, determinism and reference-test conventions | Cross-provider tests consume shared rules |
| P1 | Developer quickstarts | JShell, JBang, Jupyter and ordinary Java examples | Clean setup and smoke-tested examples |
| P1 | Extension author kit | Provider/model/algorithm/dataset adapter templates and test fixtures | Third-party fixture builds without internal project access |
| P2 | Statistics and dataset scope | Domain APIs and module dependency proposals | Clear independence from NN/AQLI/Tafkir |
| P2 | Diagnostics foundation | Shape errors, numerical checks, provider explanation, memory tracing | Stable diagnostic contracts |
| P2 | Profile/BOM consumer verification | Minimal/numerics/tensor/ML profile fixtures | Resolved dependency and package evidence |
| P3 | Generative/diffusion SDK | Model family and sampler contracts | Reusable operators and reference correctness |

## 11. Recommended next implementation increment

The next code change should be **P0: Tensor semantics and memory conformance**, not a large Tensor redesign.

Start with the existing Tensor and TensorStorage APIs:
1. Inventory all Tensor methods and their backend implementations/tests.
2. Record behavior for broadcasting, shape validation, indexing, views, reductions, dtype conversion, device transfer, mutation and errors.
3. Add focused tests for the already documented ownership contract: construction transfer, failure rollback, retain/release balance, derived views outliving parents, idempotent close and use-after-close.
4. Add no new public abstractions unless a test demonstrates an unexpressed contract.
5. Separate behavior fixes from any later source/binary compatibility API redesign.

This work is foundational for scientific computing, NN, multimodal and diffusion alike.

## 12. Scope and confidence

**High confidence (direct source evidence):** repository module counts from the recursive tree; Tensor interface shape; TensorStorage/TensorAllocator contracts; representative Gradle dependency declarations; dynamic model-family inclusion; existing scientific/ML modules and tests.

**Medium confidence (needs wider verification):** whether every view implementation retains correctly; public API dependency leakage; whether existing operation semantics are consistent across providers; completeness of diagnostics and extension registration.

**Not established by this audit:** resolved runtime dependency sizes, external published artifact behavior, complete test status, numerical parity across all backends, JShell/JBang/Jupyter smoke tests, or production readiness of the 261 model-family modules.

No Gradle build or tests were run for this documentation-only audit.
