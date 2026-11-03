# Phase 0–1: Repository audit and provider SPI stabilization

**Status:** source-level audit in progress; runtime/build validation outstanding  
**Audit date:** 2026-11-03  
**Scope:** Aljabr roadmap Phase 0 and Phase 1 only

This document is an evidence-based snapshot of the repository configuration and selected source contracts. It is not a claim that all modules compile, all declared capabilities work, or all build profiles have been exercised. The next audit pass must run the Gradle wrapper in a JDK 25 environment and inspect the complete source tree and dependency reports.

## 1. Scope and exit criteria

### Phase 0 — Source-of-truth audit

Required outputs:
- Gradle project inventory, including conditional and dynamically included projects.
- Public API versus implementation inventory.
- Dependency direction and coupling risks.
- Existing tensor, numerical, model, backend, plugin and SPI architecture.
- Test/build integrity evidence.
- Duplicated or misplaced responsibilities.
- Prioritized refactoring backlog with evidence and acceptance criteria.

### Phase 1 — Core provider SPI

Required outputs:
- Independently consumable minimal SPI artifact.
- Namespaced provider identity and metadata.
- Machine-readable capability declarations.
- Defined Aljabr API compatibility semantics.
- Documented initialization and shutdown lifecycle.
- Small immutable provider context.
- Aljabr-owned exceptions and diagnostics.
- No inference, CDI, reactive, logging, native SDK or product dependency in the core SPI.

Phase 2 registry implementation, capability conformance, provider selection and capability-specific correctness suites are explicitly out of scope for this audit.

## 2. Gradle project inventory (settings-level)

The following inventory is derived from the current `settings.gradle.kts`. It records declared inclusion rules, not a successful `./gradlew projects` execution.

### Always included explicitly

- `:core:aljabr-numerics`
- `:core:aljabr-linear-algebra`
- `:core:aljabr-tensor`
- `:core:aljabr-core`
- `:core:aljabr-error-code`
- `:core:aljabr-nn`
- `:core:aljabr-spi-core`
- `:core:aljabr-spi-testkit`
- `:core:aljabr-spi-model`
- `:core:aljabr-spi-inference`
- `:core:aljabr-spi-plugin`
- `:core:aljabr-spi-tokenizer`
- `:core:aljabr-3d`
- `:backend:cpu:aljabr-backend-cpu`
- `:backend:cuda:aljabr-backend-cuda`
- `:backend:cuda:aljabr-kernel-cuda`
- `:backend:metal:aljabr-backend-metal`
- `:core:aljabr-gguf-api`
- `:core:aljabr-gguf-llamacpp`
- `:core:aljabr-gguf-core`
- `:core:aljabr-gguf-java`
- `:core:aljabr-rocksdb`
- `:core:aljabr-helixdb`

### Included conditionally or discovered dynamically

- `:core:autograd` only when `-PwithLegacyAutograd=true`.
- `:backend:hat:aljabr-backend-hat` only when `-PwithHat=true` and a valid Babylon JDK 26 path is supplied.
- `:models:<model-project>` for each matching `models/aljabr-model-*` directory containing a Gradle build file.
- Optional SafeTensors / GGUF integration projects are included only when their candidate directories contain a Gradle build file:
  - `:core:aljabr-safetensor-api`
  - `:core:aljabr-safetensor-spi`
  - `:core:aljabr-safetensor-core`
  - `:core:aljabr-safetensor-loader`
  - `:core:aljabr-safetensor-bridge`
  - `:core:aljabr-safetensor-quantization`
  - `:core:aljabr-gguf-bridge`
  - `:core:aljabr-gguf-fast-bridge`
  - `:core:aljabr-gguf-converter`
  - `:core:aljabr-gguf-converter-java`
- `:core:aljabr-core` is passed through an optional-directory helper and also appears as a direct include later in settings; this duplication should be checked and simplified if confirmed to be redundant.

The settings file also contains commented-out modules and experimental backend references. They are not active Gradle projects and should not be counted as included modules.

**Outstanding verification:** run `./gradlew projects` and record the resolved project list. Confirm all optional paths and dynamically discovered model modules from a clean checkout.

## 3. Architecture inventory and ownership observations

| Area | Existing source-level evidence | Phase 0 assessment |
|---|---|---|
| Numerics | `aljabr-numerics` module; prior audit identified precision/tolerance policies, validation, non-finite handling and deterministic RNG | Inspect public surface, determinism guarantees and consumer dependencies |
| Linear algebra | `aljabr-linear-algebra` depends on numerics | Establish which matrix/vector algorithms are implemented and tested; do not infer complete coverage from module presence |
| Tensor | `aljabr-tensor`; backend CPU appears as a test runtime dependency | Map tensor/storage/operator contracts and identify why tests need the CPU backend at runtime |
| Neural primitives | `aljabr-nn` depends on tensor | Inspect API overlap with legacy autograd/model abstractions |
| Autograd/training | Legacy `:core:autograd` is excluded by default and conditionally included | Keep out of foundational claims until build visibility and behavior are validated |
| Model SPI | `aljabr-spi-model` depends on tensor/error-code and exposes Mutiny/Jackson/validation/logging implementation dependencies | Keep model-family concerns domain-specific; audit whether each dependency belongs on public or implementation surface |
| Plugin SPI | `aljabr-spi-plugin` exposes model, inference and error-code projects and uses Mutiny, CDI and logging | Inference/application plugin lifecycle, not the universal computational provider SPI |
| Backends | CPU, CUDA and Metal projects; CUDA depends on CPU backend and CUDA kernel module | Map fallback/delegation, backend contract, capability coverage and platform constraints |
| Formats/integrations | GGUF, SafeTensors, tokenizer, RocksDB and HelixDB projects | Treat as optional integration areas; verify that foundational numerical consumers do not pull them transitively |
| Build/publication | Root build applies Java and Maven Publish to subprojects and sets Java 25 toolchain | Audit artifact publication boundaries, generated metadata, and consumer dependency leakage |

The repository has useful prior art but several overlapping concepts: model-family plugins, inference plugins, backend interfaces and the new general provider SPI. Phase 0 should map and document these before any migration. Do not mass-replace existing plugin or backend contracts during Phase 1.

## 4. Build and dependency risks

### 4.1 Global build behavior is too broad for a minimal SPI by default

The root `build.gradle.kts` currently:
- Applies the Java and Maven Publish plugins to all subprojects.
- Sets a Java 25 toolchain for all subprojects.
- Adds `--enable-preview` to every JavaCompile task.
- Adds `--enable-preview` and `--enable-native-access=ALL-UNNAMED` to every Test and JavaExec task.
- Adds a shared JUnit BOM, JUnit Jupiter and AssertJ to every subproject test classpath.
- Contains dependency substitution and version-alignment logic for selected artifacts.

These choices may be valid for the current repository baseline, but they are not minimal defaults for every published library. In particular, the core provider SPI should not require preview features or native access. Phase 0 should identify which modules genuinely need each flag; Phase 1 should verify that the SPI can compile and test without them. Avoid changing all build conventions in the same PR as an SPI contract change.

### 4.2 Module-local build duplication

The new `aljabr-spi-core` build file repeats Java 25 toolchain and JUnit/test configuration already supplied by the root convention. This is redundant and creates two places to change the same policy. The module also declares test dependencies that are already injected by the root build. Consolidate only after verifying the root convention behavior and standalone publication requirements.

### 4.3 Dependency boundary checks still required

Generate dependency reports for at least:
- `:core:aljabr-spi-core:dependencies`
- `:core:aljabr-numerics:dependencies`
- `:core:aljabr-linear-algebra:dependencies`
- `:core:aljabr-tensor:dependencies`
- `:backend:cpu:aljabr-backend-cpu:dependencies`
- `:core:aljabr-spi-model:dependencies`
- `:core:aljabr-spi-plugin:dependencies`

Acceptance: a consumer of `aljabr-spi-core` must not resolve Mutiny, CDI, logging implementations, accelerator SDKs, model-format libraries or product/runtime dependencies.

## 5. Existing extension architecture

### 5.1 Existing patterns

- `aljabr-spi-model` is a model-family extension contract and registry. Keep it specialized.
- `aljabr-spi-plugin` defines a lifecycle for inference-kernel plugins. Its `AljabrPlugin` API imports `io.smallrye.mutiny.Uni`; the module also exposes model/inference/error-code projects and has CDI/logging dependencies.
- Existing backend implementations and service descriptors provide discovery prior art. Their exact service interface, descriptors and behavior must be inventoried before migration.
- `aljabr-spi-core` is the new provider-neutral computational extension contract.

### 5.2 Migration rule

Do not rename or replace every existing plugin system with `AljabrProvider`. The new provider SPI is for computational execution providers. Model-family and inference plugin contracts remain separate unless a concrete use case proves that a shared abstraction reduces coupling without weakening their semantics.

## 6. Phase 1 SPI review

Current core types present in `core/aljabr-spi-core`:

- `AljabrProvider`
- `ProviderDescriptor`
- `CapabilityDescriptor`
- `CapabilityStability`
- `ProviderContext`
- `ProviderException`

The core artifact has the correct intended dependency direction: provider-owned Java contracts, no dependency on inference/CDI/reactive/logging/native APIs in its source signatures. This is the right foundation, but the current shape should be treated as an initial proposal, not yet a stable external contract.

### 6.1 Gaps to close before declaring Phase 1 stable

1. **Compatibility is unstructured.** `ProviderDescriptor.apiCompatibility` is a free-form string (current test uses a caret expression), but no grammar, parser, comparison rule, inclusive/exclusive boundary, prerelease behavior or compatibility check is defined. Choose and document one minimal version-range format. Do not accept arbitrary strings as if they were validated.
2. **Provider and capability version semantics are underspecified.** Define whether these are semantic versions and what compatibility guarantees apply. Keep provider release version distinct from capability contract version and Aljabr SPI version.
3. **Capability IDs need a canonical rule.** Provider IDs are checked for namespace/name form; capability IDs currently accept any nonblank string. Define a stable namespaced ID syntax and case policy before external adoption.
4. **Duplicate capability declarations need a rule.** A Java `Set` can contain two descriptors with the same ID but different versions/stability. Validate uniqueness by capability ID, not record equality alone.
5. **Lifecycle semantics are too brief.** Specify whether initialization may be called once, what happens after partial failure, whether close is idempotent, whether a provider instance is reusable, and what thread-safety/concurrency guarantees apply.
6. **Context is generic but ungoverned.** The immutable string map is appropriately small, but configuration key namespacing, secret handling, mutability and provider-specific key ownership need documentation. Do not add device, metrics, diagnostics or runtime services to core without a concrete cross-provider requirement.
7. **Failure model is narrow.** `ProviderException` is Aljabr-owned, but descriptor construction/inspection currently can fail with unchecked exceptions. Define which failures belong to provider construction, descriptor access, initialization and shutdown. Preserve causes.
8. **ServiceLoader is documented but not proven end-to-end.** Phase 1 needs a small external-provider fixture or sample proving that the published SPI can be implemented and discovered without internal project dependencies. Discovery/catalog implementation remains Phase 2; do not add a registry in this phase.
9. **Public API documentation and publication need validation.** Check Javadoc, source/Javadoc artifacts, POM metadata and a clean external consumer build.

### 6.2 Recommended Phase 1 contract decisions

- Keep `AljabrProvider` as a small synchronous lifecycle contract; do not add reactive types.
- Keep Java `ServiceLoader` as a convention for provider discovery, but do not put a registry or selection policy in the core SPI.
- Define stable IDs and semantic version meanings before external providers ship.
- Use Aljabr-owned types and checked provider lifecycle failures.
- Keep provider context immutable and minimal.
- Keep capability-specific execution interfaces in later, focused SPIs.
- Make close safe after partial initialization and document idempotency.
- Publish the SPI independently and verify it from a separate minimal Java project.

## 7. Prioritized Phase 0–1 backlog

| Priority | Phase | Work | Acceptance evidence |
|---|---|---|---|
| P0 | 0 | Run `./gradlew projects` on clean JDK 25 checkout | Resolved project inventory captured; optional modules clearly separated |
| P0 | 0 | Run focused baseline compile/tests for core, numerics, linear algebra, tensor and CPU backend | Commands, environment, outcomes and failures recorded; no unverified green claims |
| P0 | 0 | Generate dependency reports for core/SPI and foundational modules | Public and runtime dependency boundaries documented |
| P0 | 0 | Inventory public APIs and service descriptors for backend, model, inference and plugin systems | Ownership map with source paths and migration/non-migration decisions |
| P1 | 1 | Specify provider/capability IDs and semantic version meanings | Written contract plus validation tests |
| P1 | 1 | Define and implement API compatibility range semantics | Valid/invalid syntax, boundary and compatibility tests |
| P1 | 1 | Define lifecycle and failure guarantees | Javadocs and tests for failed initialization, close-after-failure and repeated close |
| P1 | 1 | Validate duplicate capability IDs | Constructor rejects ambiguous declarations |
| P1 | 1 | Verify no unwanted dependencies and publishability | Dependency report, JAR/Javadoc/POM inspection and external consumer smoke test |
| P1 | 1 | Add a minimal external provider fixture | Separate project compiles against the published artifact and registers via ServiceLoader |
| P2 | 1 | Decide whether the testkit is Phase 1 or Phase 2 deliverable | Keep optional test helpers out of the runtime API; no registry/conformance claims yet |

## 8. Validation log

As of this audit:
- Source files and Gradle configuration were inspected through repository file retrieval.
- No local repository checkout was available in the execution environment.
- `./gradlew projects`, Gradle dependency reports, compilation, tests, Javadoc and publication smoke tests have **not** been run.
- Therefore build integrity and runtime behavior remain unverified.

This audit must be updated with actual command output before Phase 0 is marked complete.

## 9. Phase gates

### Phase 0 complete when
- The resolved Gradle project list and conditional modules are recorded.
- Public API and implementation ownership are mapped.
- Dependency direction and cycles are checked using Gradle reports.
- Focused baseline build/test results are recorded.
- Existing extension systems and backend service descriptors are inventoried.
- A prioritized, evidence-linked backlog is accepted.

### Phase 1 complete when
- Provider identity, capability IDs and version semantics are specified.
- API compatibility has a documented grammar and executable validator.
- Lifecycle and failure behavior are documented and tested.
- The SPI has no unwanted runtime dependencies.
- A separate external consumer can compile, load and close a provider.
- Artifact metadata/Javadoc/publication are verified.
- No Phase 2 registry, capability-selection or numerical conformance behavior is falsely implied.

## 10. Explicit non-goals

- No capability registry or provider selection engine.
- No general plugin framework or dynamic classloader.
- No automatic plugin download or marketplace.
- No GPU/native/distributed provider implementation.
- No migration of model-family or inference plugin systems.
- No numerical/statistical/ML capability expansion.
- No broad Gradle convention rewrite without dependency/build evidence.
