# Aljabr Modular Distribution and Dependency Architecture

**Status:** Proposed target architecture, based on the 2026-11-01 source inventory  
**Java baseline:** JDK 25

> This document defines distribution principles. It does not assert that every candidate artifact is implemented, independently publishable or production-validated.

## 1. Decision

Keep one Aljabr repository and coordinated release train, but let consumers select cohesive capabilities as independent artifacts. A small numerical or tensor consumer should not transitively acquire model catalogues, training code, format adapters, native libraries or accelerator SDKs.

**Dependency selection is the primary modularity mechanism. JVM tree shaking is not.** Prevent unwanted dependencies from entering the graph; treat dead-code elimination as an optional deployment optimization.

## 2. Current repository baseline

Aljabr already has a multi-project Gradle build. Inspected examples include:

| Existing project | Observation |
|---|---|
| :core:aljabr-numerics | Numerical foundation project |
| :core:aljabr-linear-algebra | Depends on numerics |
| :core:aljabr-tensor | Tensor API and operations |
| :core:aljabr-core | java-library facade exporting tensor and model SPI |
| :core:aljabr-nn | Neural-network building blocks |
| :backend:cpu:aljabr-backend-cpu | CPU provider depending on tensor |
| CUDA / Metal projects | Separate backend projects |
| GGUF / Safetensors projects | Separate format and adapter projects |
| :core:autograd | Legacy/training area, excluded from default build unless enabled |

The settings file already uses opt-in switches for legacy autograd, Tafkir integration, Safetensors quantization and experimental HAT. This is a useful starting point for optional capabilities.

A Gradle project is not automatically a clean published artifact. Inspected modules vary in plugin usage (basic java versus java-library); published coordinates, API leakage, metadata and consumer resolution must be verified module by module. Legacy autograd is excluded by default and references snapshot artifacts; selected optimizer APIs are incomplete. Do not pull this area into a minimal consumer by default.

See [Computational Capability Inventory](COMPUTATIONAL_CAPABILITY_INVENTORY.md) for source evidence and maturity status.

## 3. Distribution layers and candidate artifacts

These are logical boundaries, not a requirement to create a project for every row.

| Layer | Candidate artifact(s) | Rule |
|---|---|---|
| Numerical foundation | aljabr-numerics, aljabr-linear-algebra | Small, CPU-friendly; no NN, serving or accelerator dependency |
| Tensor API | aljabr-tensor | Stable tensor/shape/dtype/storage contracts; no concrete provider dependency |
| Domain computation | aljabr-statistics, aljabr-ml-classical, aljabr-nn, aljabr-autograd, aljabr-nlp | Cohesive domain modules depending inward on the smallest foundations |
| Providers | aljabr-backend-cpu, aljabr-backend-cuda, aljabr-backend-metal | Explicitly selected implementations behind SPIs |
| Interoperability | GGUF, Safetensors and other format adapters | Optional; never required by numerical/tensor-only consumers |
| Alignment | aljabr-bom | Version alignment only |
| Convenience bundles | Curated bundles | Explicit opt-in; never the default dependency of core |

Candidate artifact names are illustrative, not a claim that matching projects or packages already exist. Do not split one artifact per algorithm. Create a module only when it has a coherent API, stable ownership, independent tests, a useful consumer boundary and a manageable dependency graph. Retain existing coordinates where possible; renames require a migration plan.

## 4. Dependency direction

    Consumer application
      -> selected domain modules (statistics / ML / NN / NLP)
          -> numerical / linear-algebra / tensor APIs
          -> optional provider SPI
      -> selected backend provider(s)

Foundations must not depend on domains, applications, serving runtimes or concrete providers.

- Statistics and classical ML must work without GPU or deep-learning modules.
- Tensor/operator contracts must not import CUDA, Metal or native provider implementations.
- A provider may depend on the SPI it implements; the SPI must not depend on that provider.
- Format adapters depend inward on stable tensor/model contracts.
- Numerical foundations must not depend on AQLI, Tafkir, Farabi or agent frameworks.
- Use Gradle api only for types exposed by a module's public compile-time contract; use implementation for internal dependencies.
- Keep aljabr-core intentionally small. Narrow consumers should depend directly on narrow artifacts rather than a facade that re-exports every capability.

## 5. BOM semantics

A Maven BOM aligns versions of dependencies that the consumer chooses. **It does not add dependencies, install modules, remove unused classes or guarantee runtime provider availability.**

A future BOM should be a Gradle java-platform project and list only artifacts actually published and supported together. Illustrative usage after publication:

    dependencies {
        implementation(platform("tech.kayys.aljabr:aljabr-bom:<release>"))
        implementation("tech.kayys.aljabr:aljabr-numerics")
        implementation("tech.kayys.aljabr:aljabr-linear-algebra")
        implementation("tech.kayys.aljabr:aljabr-statistics")
    }

Importing the BOM alone must not add any Aljabr module. A tensor-only consumer should not declare statistics, NN, formats or GPU providers. Prefer platform(...) for normal alignment; use enforcedPlatform(...) only when an application intentionally needs strict enforcement. Libraries should generally avoid forcing downstream versions.

Do not create a BOM before supported artifact coordinates and release policy are known.

## 6. Optional providers and capability discovery

Keep provider selection behind a stable API/SPI. The application declares the provider dependency; the provider registers through ServiceLoader or explicit registration; the runtime checks declared capabilities, hardware, native libraries, JDK and operator support before selection. Unsupported requests must return actionable typed errors, not silently claim GPU execution or change numerical semantics.

Capability discovery must distinguish:
- **Defined:** capability identifier is part of the API.
- **Implemented:** an implementation exists in source.
- **Installed:** provider artifact is on the runtime classpath/module path.
- **Available:** environment prerequisites are satisfied.
- **Validated:** correctness and compatibility suites passed for a named environment.

The existence of a CUDA project is not evidence that CUDA is installed or usable on a host. Keep the provider SPI small; rich algorithm metadata (assumptions, parameters, diagnostics and maturity) is a separate concern.

## 7. Packaging: choose dependencies before shrinking

| Mechanism | Role | Limitation |
|---|---|---|
| Gradle/Maven selection | Keeps unselected artifacts and transitive dependencies out of the graph | Does not remove unused classes inside a selected artifact |
| JVM class loading | Loads classes as needed | Does not remove dependency JARs from deployment |
| JPMS | Explicit module boundaries and readability | Does not automatically remove all unused code/dependencies |
| jlink | Builds a custom runtime from a modular application and selected JDK modules | Does not replace application artifact modularity |
| GraalVM Native Image | Reachability-based native executable construction | Reflection, resources and dynamic loading may need configuration |
| Shrinking tools | May remove unreachable bytecode | Can break reflection, service loading and native/resource access |

Recommended order: select only required artifacts; isolate optional integrations; minimize transitive dependencies; use JPMS/jlink where supported; consider native-image or shrinking only for a concrete deployment target with integration tests. Do not promise automatic tree shaking for ordinary JVM consumers.

## 8. Release and compatibility policy

- Publish supported artifacts from a coordinated release train; the BOM version matches that train.
- Publish sources/Javadoc where practical.
- Validate published POM/Gradle metadata and clean external-consumer resolution; a composite build alone is insufficient.
- Test that a minimal consumer does not resolve NN, training, formats, CUDA, Metal or model catalogue dependencies unless explicitly selected.
- Test provider absent, provider installed but hardware absent, provider available and unsupported-operation cases.
- Record JDK/platform requirements per artifact. Aljabr documents JDK 25; do not imply JDK 21 compatibility without bytecode and runtime validation.
- Treat public API, package and coordinate breaks as migrations.

## 9. Implementation sequence

| Phase | Work | Acceptance evidence |
|---|---|---|
| 1. Graph audit | Generate Gradle project/dependency reports; map coordinates, plugin types, API leakage and publication status | Reviewed artifact and dependency inventory |
| 2. Boundary hardening | Fix accidental API leaks and unwanted transitive dependencies | Minimal consumer graph tests |
| 3. Provider contract | Define provider SPI and capability maturity/availability states | Provider absent/present/hardware-missing tests |
| 4. BOM | Add java-platform project after artifact coordinates are confirmed | Published BOM aligns selected artifacts |
| 5. Consumer examples | Tensor/CPU, statistics-only and optional accelerator examples | Clean external Gradle consumer builds |
| 6. Packaging validation | Evaluate JPMS/jlink/native-image for real targets | Reproducible size/startup measurements and integration tests |

Do not combine this with a broad source-tree reorganization. Measure the current graph, fix demonstrated leaks, then implement the BOM and provider SPI against stable coordinates.

## 10. Consumer profiles

These are target profiles, not assertions that every artifact is currently published.

| Consumer | Select | Must not pull by default |
|---|---|---|
| Numerical utility | Numerics, linear algebra | NN, autograd, formats, GPU |
| Farabi statistics worker | Numerics, linear algebra, statistics, selected CPU provider | Model catalogue, NN, training, GPU/native runtimes unless selected |
| AQLI inference runtime | Tensor/model APIs, required format adapter, selected backend | Statistics, classical ML, Tafkir workflow |
| Tafkir training workflow | Tensor, NN, validated autograd/optimizer modules, selected backend | Serving product and unrelated format adapters |
| NLP preprocessing | Tokenizer and selected NLP/vectorization modules | GPU/model execution unless required |

Farabi's existing architecture has a JDK 21 baseline while Aljabr documents JDK 25. Until deliberately reconciled, use a separately validated worker/service boundary rather than assuming in-process compatibility.

## 11. Open decisions

1. Which current projects are supported published artifacts versus internal implementation projects?
2. Should aljabr-core remain a small facade, or should consumers prefer direct module dependencies?
3. Which provider SPI is canonical, and how are current providers registered?
4. Can any artifact support a lower JDK baseline without compromising the JDK 25 framework baseline?
5. Which real deployment targets justify JPMS/jlink or native-image?

Resolve these through the graph audit and small consumer prototypes. Do not pre-emptively add every candidate artifact or a BOM containing unverified coordinates.

## 12. Related documents

- [Computational Framework Scope](COMPUTATIONAL_FRAMEWORK_SCOPE.md)
- [Computational Capability Inventory](COMPUTATIONAL_CAPABILITY_INVENTORY.md)
- [Framework Architecture](FRAMEWORK_ARCHITECTURE.md)
- [Research and Training Readiness](RESEARCH_TRAINING_READINESS.md)
