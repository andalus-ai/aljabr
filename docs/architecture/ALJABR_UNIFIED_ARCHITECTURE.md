# Aljabr Unified Computational Architecture

**Status:** Architecture direction and repository consolidation  
**Baseline:** 2026-11-03  
**Repository snapshot:** `main`, tree `017583d2441f0a3b890a8de94bd503df06d5f565`  
**Java baseline:** JDK 25, subject to per-module compatibility policy

> This document is the architectural source of truth for Aljabr's mission, ownership boundaries, layering and growth constraints. It reconciles the more focused scope, capability-inventory, framework, distribution and Gradle-audit documents linked below. It does not claim that every target capability is implemented, validated, optimized or published.

## 1. Decision

Aljabr is the **shared, Java-native computational framework** for the Andalus ecosystem. It owns a coherent programming model for reusable numerical, mathematical, statistical, scientific and machine-learning computation.

Aljabr is intentionally broad in the *capabilities it can host*, but must remain narrow in the *dependencies and runtime capabilities required by any one consumer*.

The governing principle is:

> One Aljabr-owned public programming model; cohesive capability modules; interchangeable execution providers; independently selectable distributions.

Aljabr is not a universal application framework, an inference server, a data platform, an agent runtime or a workflow product. It must not become a monolithic dependency that every consumer has to load.

## 2. Ecosystem ownership

| Project | Owns | Explicit boundary |
|---|---|---|
| **Aljabr** | Numerical and mathematical semantics, tensor/array and memory contracts, reusable algorithms, computational APIs, provider SPI, conformance and diagnostics | No product workflows, serving control plane, tenant governance or agent orchestration |
| **AQLI** | Model loading/execution lifecycle at runtime, inference requests, scheduling, generation policy, streaming, serving APIs and deployment operations | Consumes Aljabr computation; does not own general-purpose tensor/math algorithms |
| **Tafkir** | Training and adaptation workflows, dataset pipelines, experiment orchestration and fine-tuning policy | Consumes Aljabr primitives; must remain optional to Aljabr and ordinary AQLI inference |
| **Farabi** | Research/analytics product, data access, semantic layer, query/research planning, governance, provenance, interpretation and presentation | Selects and invokes computational methods; does not fork canonical algorithms |
| **NumU** | Shared platform foundation and infrastructure | Does not replace Aljabr's computational semantics |
| **Andalus / Andalus Framework** | General-purpose agent runtime, tools, planning and workflow orchestration | Does not become the numerical execution authority |
| **Toba** | Data/lakehouse connectivity, source adapters and query integration | Does not become a statistical-computation engine |
| **Batik / Tenun** | UI, visualization and presentation | Does not own algorithm execution or numerical interpretation |

Dependency direction is inward: products and runtimes consume Aljabr; Aljabr does not depend on them. Farabi can choose a statistical method and govern the research workflow, while Aljabr supplies the reusable algorithm and its diagnostics. If runtime or JDK compatibility prevents safe in-process use, the integration boundary belongs in a separately designed worker/service contract—not inverting dependencies.

## 3. The AQLI relationship: engine, not serving product

Aljabr began as a computational engine for AQLI, and that remains a first-class use case. The relationship should be:

- Aljabr supplies tensors, numerical kernels, memory/resource contracts, model-computation primitives and execution-provider abstractions.
- AQLI owns inference-specific runtime behavior: model/session lifecycle, request batching and scheduling, token generation policy, streaming, cancellation, service APIs, deployment and operational controls.
- AQLI may define requirements that motivate reusable Aljabr capabilities, but those requirements do not automatically make AQLI-specific orchestration part of Aljabr.
- Aljabr must be usable without AQLI. A numerics-only or statistics-only consumer must not load inference, model-format, tokenizer, GPU or serving dependencies.

This keeps Aljabr useful to AQLI without turning it into AQLI's control plane.

## 4. Layered architecture

The logical layers are:

```text
Products and applications
  Farabi | AQLI | Tafkir | Andalus applications
                  |
          Aljabr public APIs
                  |
     Cohesive capability modules
  Statistics | Scientific | Classical ML | Neural | NLP
                  |
       Computational foundations
  Numerics | Linear algebra | Tensor | Probability | Optimization
                  |
       Execution contracts / SPI
                  |
       Execution providers
  Pure Java | Vector API | Native/FFM | GPU | Distributed
```

### 4.1 Foundation

The foundation owns stable primitives and semantics: numerical precision/tolerances, shapes and dtypes, array/tensor and matrix/vector abstractions, memory ownership/lifetime, deterministic random-source contracts, common errors and diagnostics, and execution context/device contracts where justified.

Keep this layer small. It must not depend on domain algorithms, model catalogues, file-format integrations, accelerator SDKs, serving runtimes or products.

### 4.2 Capability modules

Domain capabilities are independently consumable modules built on the foundation. Candidate areas include linear algebra, probability/distributions, statistics, optimization, scientific computing, structured/tabular data, classical ML, autograd, neural primitives and NLP foundations.

These are capability areas, **not instructions to create one Gradle project per topic or algorithm**. Reuse existing projects and packages. Create or split a module only when it has:
- a coherent public API and ownership boundary;
- a useful independent consumer/dependency boundary;
- focused tests and a clear validation strategy;
- manageable dependencies and a lifecycle that can be versioned;
- a demonstrated use case.

Do not create a new module merely to mirror a Python package or to increase the module count.

### 4.3 Execution providers

The public API and mathematical semantics must be provider-independent. Providers implement existing Aljabr contracts; they do not redefine the public programming model.

- **Pure Java reference path:** always available and suitable as a correctness baseline.
- **Java Vector API:** optional SIMD implementation, isolated to modules that actually use it.
- **Native/FFM:** optional integration for libraries such as BLAS/LAPACK or FFT implementations, with explicit ABI, memory and resource-lifecycle validation.
- **GPU and specialized hardware:** optional, capability- and platform-gated.
- **Distributed:** optional execution strategy with explicit partitioning, aggregation, consistency and failure semantics.

A provider declaration is not proof of availability or correctness. Capability status must distinguish defined, implemented, installed, available, validated, stable and optimized. Automatic dispatch must be explainable; unsupported operations must fail clearly rather than silently claiming acceleration or falling back to an incompatible semantic path.

## 5. Aljabr owns its public programming model

This is a non-negotiable architectural constraint.

Consumers program against Aljabr-owned types, contracts, lifecycle rules, errors, diagnostics and semantics. Third-party projects—including Apache Commons Math, Hipparchus, EJML, Smile, Tribuo, OpenBLAS and other native libraries—may be evaluated as implementation dependencies behind internal adapters. They must not become the public API or define Aljabr's behavior.

In particular:
- Do not expose third-party matrix/tensor/model types in Aljabr public signatures.
- Do not leak third-party exceptions, lifecycle or configuration objects as the stable contract.
- Translate implementation-specific errors into Aljabr-owned error/result semantics.
- Preserve provider-independent numerical meaning; document tolerances where operation ordering or hardware precision changes results.
- Keep adapters replaceable and optional. A third-party implementation can accelerate or fill a validated capability, but it does not become the architecture.
- Do not add a dependency solely because a library has a feature; assess licensing, maintenance, compatibility, performance, semantics and adapter cost.

The same rule applies to native libraries: Aljabr's public API remains Java-native and stable even when an internal provider delegates to BLAS/LAPACK or another native implementation.

## 6. Anti-monolith and dependency rules

1. **Small foundation:** no all-capabilities facade as the required entry point.
2. **Opt-in domains:** consumers select only the capability artifacts they need.
3. **Opt-in providers:** no mandatory GPU SDK, native library, model catalogue or hardware runtime.
4. **No product dependencies:** Aljabr does not depend on AQLI, Tafkir, Farabi, NumU or Andalus application/runtime modules.
5. **No accidental re-exports:** use Gradle `api` only when a dependency's types are genuinely part of the module's public contract; otherwise use `implementation`.
6. **No premature module explosion:** prefer cohesive modules and preserve existing coordinates unless a migration is justified.
7. **No false tree-shaking promise:** first keep unwanted artifacts out of the dependency graph. JPMS/jlink, GraalVM Native Image and shrinking are later deployment optimizations with explicit compatibility tests.
8. **No premature BOM:** a BOM aligns versions of dependencies the consumer already selected; it does not add modules, remove classes or guarantee provider availability. Publish it only after artifact coordinates and release policy are reliable.
9. **No hidden build coupling:** compiler flags, native access, Jandex, substitutions and external composite builds must be scoped to modules that require them.
10. **No unverified capability claims:** source presence, project inclusion or a provider class does not prove correctness, production readiness or performance.

## 7. Distribution and developer workflows

Distribution should support three complementary mechanisms:

- **Feature artifacts:** the primary way to select capabilities.
- **BOM:** version alignment only, after publication metadata and release compatibility are established.
- **Starter bundles:** optional curated convenience sets (for example numerics, scientific, statistics, ML or research); they must not be mandatory and must not pull in heavyweight providers by default.

Gradle/Maven dependency selection is the first and most reliable form of modularity. JPMS and jlink can be adopted where module boundaries and runtime requirements justify them. Native-image/shrinking must be evaluated against reflection, ServiceLoader, FFM, resources and provider discovery.

Research workflows should reuse the same APIs:
- JBang for lightweight Java experimentation and promotion to a regular project.
- Jupyter through an established Java kernel (for example IJava), with Aljabr examples and reproducibility guidance; do not build a custom notebook kernel as a prerequisite.
- Gradle/Maven for production projects.
- An optional CLI may generate project configuration, inspect capabilities and help with setup; it must not become a second algorithm runtime.

## 8. Current repository baseline and interpretation

The live recursive Git tree inspected for this consolidation contains **6,484 entries** and is not marked truncated. It includes existing `aljabr-numerics`, `aljabr-linear-algebra`, `aljabr-tensor`, `aljabr-nn`, model/SPI/format projects and CPU/accelerator backend projects. The exact Gradle project graph remains `./gradlew projects`, not directory naming.

Existing evidence includes:
- Numerics contracts for precision/tolerance, validation and deterministic random sources.
- Dense CPU reference linear algebra implementations including LU, QR, Cholesky, SVD and eigen/Schur-related routines.
- Tensor abstractions and operations, memory/buffer code and lazy execution components.
- Neural-network building blocks and CPU/accelerator backend implementations.
- Existing model, tokenizer, format and storage integrations.

This evidence must be interpreted narrowly. It does not prove broad statistics, classical ML, scientific-computing or backend-parity coverage. In particular, a tensor matrix-multiply operation is not a complete linear-algebra API; a model catalogue is not a catalogue of validated general-purpose algorithms; and source presence is not proof of a passing build or production readiness.

The existing `COMPUTATIONAL_CAPABILITY_INVENTORY.md` and `GRADLE_DISTRIBUTION_AUDIT.md` are explicitly dated source/build-script snapshots and state that no Gradle build or test task was executed for those audits. Their findings are useful evidence, not fresh runtime validation. Refresh those audits before implementation decisions that depend on current source or resolved dependencies.

## 9. Delivery sequence and gates

| Phase | Focus | Required evidence before advancing |
|---|---|---|
| 0. Source of truth | Reconcile architecture documents, inspect source/build graph, map implemented vs planned capabilities and ownership | Reviewed inventory, dependency map, known limitations and prioritized backlog |
| 1. Build/release integrity | Publication allowlist and metadata, scoped compiler/build flags and substitutions, clean consumer fixtures | Generated POM/module metadata and minimal external consumers resolve without local/composite artifacts |
| 2. Public semantics | Stabilize numerics, tensor/array, linear-algebra, memory, errors and execution contracts | API specification, compatibility policy and conformance tests |
| 3. Provider architecture | Provider SPI, discovery, capability states, selection/fallback and validation | Pure Java reference path and provider conformance suite |
| 4. Computational foundations | Extend validated numerical, linear-algebra, probability/statistics, optimization and scientific capabilities | Algorithm-specific reference tests, assumptions, diagnostics and reproducibility |
| 5. Performance | Optimize Java kernels; add Vector API and selective FFM/native providers | Correctness parity plus repeatable, workload-scoped benchmarks |
| 6. ML/AI foundations | Classical ML, autograd, neural primitives and NLP foundations | Independent capability tests and explicit training/inference ownership |
| 7. Distribution/DX | Feature artifacts, BOM, starters, JPMS, JBang, Jupyter guidance and optional CLI | Clean consumer profiles and tested onboarding workflows |
| 8. Distributed computation | Partitioning, aggregation, scheduling and fault/consistency contracts | Distributed conformance and failure-injection evidence |
| 9. Ecosystem integration | AQLI, Tafkir, Farabi, NumU and Toba integrations; deployment profiles | Cross-repository compatibility tests and documented runtime boundaries |

Phases may overlap when dependencies are clear, but later convenience work must not conceal unresolved public API, build or correctness problems.

## 10. Immediate engineering priorities

The next implementation slice should be evidence-led and should not add a broad new domain surface yet:

1. Refresh the source inventory against the current tree and inspect representative public APIs, tests and module build files.
2. Run the Gradle project/configuration and dependency reports in a clean JDK 25 environment; this review has not executed them.
3. Fix build/release integrity issues identified by the Gradle audit, with scoped changes and focused consumer fixtures.
4. Define the Aljabr-owned public contracts and provider conformance model before adding more third-party adapters or accelerator paths.
5. Add statistical/scientific capabilities only after selecting a cohesive API and validation plan; keep each independently consumable.
6. Keep AQLI integration focused on engine contracts and execution primitives, not serving orchestration.

## 11. Existing architecture documents

This document consolidates direction; the following documents remain the detailed evidence and implementation references:

- [Computational Framework Scope](COMPUTATIONAL_FRAMEWORK_SCOPE.md) — domain taxonomy and ecosystem ownership.
- [Computational Capability Inventory](COMPUTATIONAL_CAPABILITY_INVENTORY.md) — source-level capability evidence and gaps.
- [Framework Architecture](FRAMEWORK_ARCHITECTURE.md) — existing framework layers and execution design.
- [Modular Distribution Architecture](MODULAR_DISTRIBUTION.md) — artifact selection, BOM, providers and packaging.
- [Gradle Distribution Audit](GRADLE_DISTRIBUTION_AUDIT.md) — static build/publication configuration findings.
- [Research and Training Readiness](RESEARCH_TRAINING_READINESS.md) — autograd and training readiness limitations.
- [Tafkir–Aljabr Dependency Direction](TAFKIR_ALJABR_DEPENDENCY_DIRECTION.md) — training integration boundary.
- [Quantized Tensor Interoperability](QUANTIZED_TENSOR_INTEROPERABILITY.md) — quantized tensor interoperability.

The README remains the entry point for users; this document is the architectural decision and roadmap reference. Detailed audit documents should retain their snapshot dates and evidence limitations until explicitly refreshed.


## External Extension and Plugin Architecture

Aljabr owns the public programming model and extension contracts. Implementations may live in this repository or in independently versioned external repositories. A consumer opts into an extension by adding its artifact and configuring or discovering its provider; installing Aljabr alone must not pull every extension or provider into the runtime.

### Extension layers

1. **Core contracts** — minimal Java-native identifiers, semantic versions, capability descriptors, provider interfaces, compatibility metadata, diagnostics, and lifecycle contracts. This layer must not require inference-serving APIs, CDI, Mutiny, a logging implementation, native SDKs, or domain-specific model catalogues.
2. **Capability SPIs** — focused contracts owned by Aljabr for cohesive extension points such as linear algebra, statistics, optimization, tensor execution, random generation, or scientific computing. Each SPI depends only on the smallest Aljabr API surface it needs.
3. **Provider implementations** — optional artifacts that implement one or more Aljabr SPIs. These may live in Aljabr's repository or in third-party repositories and may depend internally on EJML, Commons Math, BLAS/LAPACK, CUDA, Metal, or other implementation libraries. Such dependencies and types must not leak through public Aljabr signatures.
4. **Discovery and runtime integration** — explicit provider registration/discovery, capability negotiation, selection policy, lifecycle management, and diagnostics. Discovery must be separable from execution so applications can use explicit wiring in restricted environments.

### Required contract properties

- Stable, namespaced provider IDs; semantic versions; declared Aljabr API compatibility range; and machine-readable capability descriptors.
- Explicit distinction between a capability being declared, implemented, installed, available on the current host, validated, stable, and optimized.
- Deterministic provider selection with explainable reasons. No silent fallback that changes numerical semantics or hides provider failures.
- Aljabr-owned value types, configuration contracts, errors, diagnostics, and lifecycle semantics. Translate provider-specific failures at the adapter boundary.
- Resource ownership and lifecycle rules for initialization, execution, cancellation, shutdown, native handles, threads, and memory.
- Thread-safety, concurrency, determinism, precision/tolerance, device constraints, and supported-shape guarantees documented per capability.
- Compatibility and conformance tests that external providers can run independently against a published Aljabr SPI test kit.
- No arbitrary class loading or automatic downloading of executable plugins. Applications explicitly choose dependencies and trust boundaries; isolated processes are required for untrusted extensions.

### Discovery model

Use Java `ServiceLoader` as the lightweight default discovery mechanism for classpath/module-path providers, with explicit programmatic registration as a first-class alternative. Do not require a heavyweight plugin framework or CDI in the core contract. If a deployment needs dynamic loading, hot reload, isolation, or remote providers, provide those as optional integration modules with their own security and lifecycle guarantees.

### Existing repository assessment

The repository already contains several extension patterns: `aljabr-spi-plugin` defines a general plugin lifecycle, `aljabr-spi-model` defines model-family plugins and a registry, and backend artifacts register `ComputeBackend` implementations through `META-INF/services`. This is useful prior art, but it is not yet one coherent general computational-provider architecture.

In particular, the current `AljabrPlugin` contract imports Mutiny and its module exposes inference/model SPIs plus CDI and logging dependencies. It is therefore inference-oriented and heavier than the minimal extension contract described above. The model-family registry is domain-specific and should remain so; it is not a universal provider registry. The CUDA plugin artifact's service descriptor for `ComputeBackend` demonstrates externalizable provider discovery, while its implementation dependencies remain provider-specific.

The next implementation step is not to replace all existing plugin systems at once. First define a small, stable computational provider contract and conformance kit; then adapt one existing provider (preferably the pure Java CPU/reference provider) and one optional accelerated provider. Migrate model-family and inference plugins only if a concrete benefit justifies it, preserving compatibility during any transition.
