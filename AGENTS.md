# AGENTS.md — Aljabr contributor guidance

## Product vision and scope

Aljabr is the reusable, Java-native computational foundation for mathematical, numerical, statistical, scientific-computing, and machine-learning workloads. It should provide a coherent public programming model and computational semantics, with multiple interchangeable execution providers.

The architectural principle is:

> One Aljabr programming model. One set of computational semantics. Multiple interchangeable execution providers. Independently extensible capabilities. From laptop to distributed infrastructure.

Aljabr owns:
- Public Java APIs and the developer-facing programming model.
- Computational semantics: data types, shapes, broadcasting, precision, determinism, memory/resource behavior, concurrency, and errors.
- Capability identity, metadata, registry contracts, and provider compatibility/selection mechanisms. Aljabr evaluates capabilities and initializes providers; the consuming application/runtime owns product-specific selection policy.
- Provider and extension SPIs, compatibility rules, validation, and conformance contracts.
- Reusable mathematical, numerical, statistical, scientific, ML, and deep-learning primitives.

Keep these boundaries explicit:
- **AQLI** owns inference serving: model serving, request scheduling, runtime lifecycle, deployment, and serving orchestration. Aljabr supplies reusable computation and model-execution primitives; it is not the serving control plane.
- **Tafkir** owns training workflows, fine-tuning orchestration, dataset pipelines, and training lifecycle. Aljabr owns reusable algorithms, tensors, optimizers, and computational primitives. Do not introduce a Tafkir dependency into Aljabr.
- **Farabi** owns research and analytics workflows: data import, research questions, analysis planning, method selection, governance, interpretation, BI, and reporting. Farabi may consume Aljabr's mathematical/statistical/ML capabilities without moving research workflow responsibilities into Aljabr.
- Visualization and presentation belong in the appropriate presentation products (such as Tenun/Batik), not in Aljabr's numerical core.
- Aljabr must remain independently useful outside the Andalus ecosystem.

Do not turn Aljabr into a mandatory all-in-one runtime. Keep the foundational core small; domain modules independently consumable; execution providers optional; and convenience bundles opt-in. A statistics-only application must not require deep-learning, GPU, native, distributed, CDI, or serving infrastructure.

## Architecture and dependency principles

- Keep one stable public Aljabr API independent of implementation libraries and execution providers.
- Third-party libraries (for example, numerical, linear-algebra, or ML libraries) may be used behind internal adapters. Never leak their types, annotations, exceptions, or lifecycle assumptions into Aljabr's public API.
- Prefer pure Java reference implementations for portability and as correctness oracles. Native libraries, Java Vector API, FFM integrations, GPU runtimes, and distributed execution are optional providers or modules, not mandatory core dependencies.
- Keep incubating/preview JDK APIs isolated in optional modules. Do not make core consumers adopt preview flags or a newer JDK solely to use an optional accelerator.
- Treat FFM as an optional native interoperability mechanism. Do not make JNI or any native runtime a default requirement; retain compatibility paths only where justified and documented.
- Preserve modular Gradle consumption. A BOM aligns versions; it must not pull every module transitively. Curated bundles/starters are optional convenience layers.
- Avoid circular dependencies. Domain APIs depend on stable lower-level contracts; provider implementations depend on the relevant SPI and API, never the reverse.
- Introduce domain-specific SPIs only when the corresponding capability contract is sufficiently understood. Do not create speculative SPI modules for every future domain.
- Prefer Java `ServiceLoader` and explicit registration for provider discovery. Avoid mandatory heavyweight plugin frameworks, automatic executable-plugin downloads, and reflection-heavy discovery in minimal deployments.
- External providers must be publishable and usable from repositories outside Aljabr through documented, versioned Aljabr APIs/SPIs.
- Do not assume an external provider is trusted. Define clear ownership and cleanup for resources, classloaders, initialization failures, and shutdown. Use process isolation for untrusted or operationally isolated providers where appropriate; do not claim in-process plugin loading is a security sandbox.
- Keep provider selection explainable. Never silently violate requested precision, determinism, device, compatibility, or resource constraints during fallback.
- Distributed execution is an optional execution model. Aljabr owns computational and execution contracts, not the cluster scheduler or cloud control plane.

## Provider and capability contracts

Provider metadata and capability metadata are related but distinct. Keep the following states distinct; do not collapse them into one boolean or one provider-supplied stability label:
- **Declared:** a provider claims to implement a capability.
- **Installed:** the provider artifact is present.
- **Available:** required runtime resources are currently available.
- **Compatible:** provider/API/capability versions and constraints match.
- **Validated:** the implementation passed the applicable conformance checks.
- **Stable:** the capability has an Aljabr-defined maturity designation.
- **Optimized:** representative benchmarks demonstrate an optimization under documented conditions.

A provider's declaration or self-reported maturity is not proof of compatibility, correctness, or performance. Capability descriptors should express relevant constraints such as operation, dtype, shape, device, precision, determinism, memory, and distribution support where applicable.

Provider lifecycle must fail clearly and clean up partially initialized resources. Explicit provider selection should be possible for reproducibility and diagnostics. Automatic selection must expose enough diagnostics to explain the selected provider and rejected alternatives.

## Ultimate implementation roadmap

Treat this roadmap as the shared architectural direction. It is a sequence of dependency-aware phases, not a requirement to finish every phase before any later work can start.

### Phase 0 — Source-of-truth repository audit

- Inventory actual Gradle projects/modules, source sets, public APIs, implementations, tests, dependencies, and build conventions.
- Map existing tensors, arrays, operators, numerical kernels, model abstractions, execution backends, provider/plugin mechanisms, and native integrations.
- Identify duplicate abstractions, circular dependencies, API leaks, compatibility constraints, incomplete implementations, and missing tests.
- Produce an evidence-based module map, dependency graph, gap analysis, and prioritized migration backlog.
- Inspect the repository before adopting proposed module names or creating new module boundaries. Architecture sketches are targets, not evidence that a module already exists or is needed.

### Phase 1 — Minimal core provider SPI

- Establish provider identity, descriptor metadata, API compatibility declarations, capability declarations, provider context/configuration, lifecycle, and well-defined failure behavior.
- Keep the base SPI small and independent of domain-specific implementation libraries and optional runtime frameworks.
- Support both Java `ServiceLoader` discovery and explicit registration.
- Review and evolve existing provider-SPI work rather than creating competing contracts.
- Do not put inference-serving, training orchestration, CDI, Mutiny, logging implementations, or accelerator SDKs in the base SPI.

### Phase 2 — Capability registry and conformance

- Define stable capability identifiers, metadata/versioning, compatibility evaluation, registration/discovery, and diagnostic reporting.
- Implement registry lifecycle and capability state transitions using the Phase 1 contracts.
- Build reusable provider SPI tests and capability-specific conformance suites.
- Establish independent numerical references and correctness tests; a provider's self-test alone is insufficient evidence.
- Separate correctness conformance from maturity and performance benchmarking.
- Ensure an external repository can implement, register, inspect, and validate a provider using published contracts.

### Phase 3 — Public API and computational semantics

- Define the public Java programming model for core numerical values, arrays/tensors, vectors/matrices, operations, and execution contexts.
- Specify dtype promotion, shape rules, broadcasting, indexing, precision, overflow, NaN/infinity behavior, determinism, random-number semantics, memory ownership, concurrency, cancellation, and error behavior.
- Keep semantics independent of the selected provider. Provider-specific limitations must be explicit and inspectable.
- Provide stable API contracts and tests before broad domain API expansion.
- Coordinate capability identifiers and constraints with Phase 2; do not let registry metadata invent or redefine computational semantics.

### Phase 4 — Reference and baseline providers

- Implement a portable pure Java reference provider as a correctness baseline.
- Build baseline CPU numerical and linear-algebra functionality on the agreed public API and SPI.
- Reuse existing implementations where they meet the contracts; avoid unnecessary rewrites.
- Demonstrate end-to-end discovery/registration, capability inspection, execution, diagnostics, and conformance.
- Establish deterministic fixtures and representative correctness tests before optimization.

### Phase 5 — Accelerated execution providers

- Add optional CPU parallelism and Java Vector API implementations where supported by the project's JDK policy.
- Add optional FFM/native and BLAS/LAPACK adapters behind Aljabr contracts.
- Evaluate GPU providers (including CUDA or other accelerators) as independent optional integrations.
- Implement explainable provider selection, explicit overrides, compatibility checks, safe fallback, and resource management.
- Benchmark only after correctness conformance. Document exact JDK, device, shape, dtype, precision, warmup, and measurement conditions.

### Phase 6 — Mathematical, statistical, and scientific expansion

- Expand reusable math, linear algebra, sparse operations, probability, distributions, descriptive and inferential statistics, regression, Bayesian methods, optimization, integration, interpolation, ODEs, simulation, signal processing, and time series.
- Build domain algorithms on shared foundational semantics rather than creating incompatible parallel data models.
- Keep modules independently consumable; do not make statistical consumers depend on ML/DL or accelerator stacks.

### Phase 7 — ML, deep learning, and NLP foundations

- Add classical ML estimators, preprocessing, feature engineering, pipelines, evaluation, classification, regression, clustering, and dimensionality reduction.
- Add autograd, computational graphs, neural-network primitives, losses, optimizers, and model-execution primitives as their foundational tensor contracts mature.
- Add reusable NLP foundations such as tokenization, embeddings, and text/vector operations where they fit Aljabr's computational scope.
- Keep model serving in AQLI and training workflow/orchestration in Tafkir.

### Phase 8 — Distributed execution

- Define partitioning, data locality, serialization, remote execution, aggregation/reduction, retries, fault handling, cancellation, resource limits, and deterministic distributed random/reduction semantics.
- Express whether each capability is distributable and under which constraints; do not imply every algorithm can run on a cluster.
- Keep cluster scheduling, deployment, and cloud control planes outside Aljabr.
- Preserve the same public computational model where feasible, while making distributed behavior and limitations explicit.

### Phase 9 — Developer ecosystem and product integration

- Provide a version-aligned BOM, optional curated bundles/starters, CLI diagnostics, JBang workflows, and notebook/Jupyter integration where practical.
- Keep the same Aljabr API across application, CLI, and notebook usage; do not create a separate notebook-only programming model.
- Publish extension-author documentation, compatibility policy, conformance kits, examples, and provider diagnostics.
- Integrate with AQLI, Tafkir, and Farabi through stable APIs without transferring their product responsibilities into Aljabr.

## Parallel execution strategy and dependency gates

Parallelize work where contracts allow it. Do not enforce a strictly serial phase-by-phase process, and do not let parallel work independently invent conflicting interfaces.

Recommended work waves:

1. **Foundation:** run the Phase 0 repository audit and Phase 1 SPI work concurrently.
2. **Contract design:** develop Phase 2 registry/conformance design and Phase 3 computational-semantics/public-API design concurrently. Review capability metadata against the evolving provider SPI and semantics.
3. **Implementation preparation:** scaffold Phase 4 provider work and Phase 2 conformance fixtures in parallel. Final provider implementation must use stable SPI and public API contracts.
4. **Execution and domain planning:** research/prototype Phase 5 accelerators and design Phase 6 domain capabilities while Phase 4 establishes a working reference implementation. Do not make accelerated providers the correctness baseline.
5. **Expansion:** develop Phase 6 and Phase 7 domain modules in parallel only where their shared foundations and module contracts are stable. Phase 8 distributed design can begin earlier, but production implementation depends on stable execution and data semantics.
6. **Ecosystem:** documentation, BOM design, CLI planning, extension examples, and downstream integration planning can proceed incrementally; finalize them against published APIs and actual module boundaries.

Use these dependency gates:

- **SPI gate:** provider identity, lifecycle, compatibility, and capability-declaration contracts are reviewed and stable before dependent implementations are treated as final.
- **Semantics gate:** public types, numerical behavior, precision, shape, memory, and execution contracts are documented and tested before broad provider/domain implementations depend on them.
- **Conformance gate:** shared provider tests and capability-specific correctness criteria exist before a provider is described as validated.
- **Reference-provider gate:** at least one end-to-end pure Java provider demonstrates registration/discovery, execution, diagnostics, and conformance before accelerated providers become the primary implementation path.
- **Interoperability gate:** an independently developed provider can implement the published SPI without changing consumer-facing APIs or depending on Aljabr internals.
- **Distributed gate:** distribution constraints, data semantics, failure behavior, and deterministic reduction/RNG expectations are explicit before claiming a capability is distributed.

Design, repository audits, compatibility research, test-fixture preparation, documentation drafts, and isolated prototypes may begin before a gate is complete. Do not merge dependent production implementations that encode provisional contracts without clearly marking and isolating the experiment.

For parallel PRs:
- Keep each PR focused on one contract, module, provider, or roadmap deliverable.
- State upstream dependencies and the gate each PR satisfies.
- Avoid simultaneous incompatible edits to shared API/SPI files; nominate one owner/reviewer for cross-cutting contracts.
- Prefer additive, reviewable changes and explicit migration notes over broad rewrites.
- Rebase or reconcile against current repository state without discarding other contributors' work.

## Engineering principles

- Inspect the owning module, its callers, tests, and existing conventions before changing code.
- Prefer small, cohesive changes with explicit contracts, deterministic behavior, and useful failure messages.
- Preserve binary compatibility and validate native/FFM signatures together. Keep Java layouts, native declarations, exported symbols, and argument order synchronized.
- Do not silently turn unsupported shapes, formats, or backends into success. Validate dimensions, byte sizes, alignment, batch bounds, and native return codes.
- For Metal code, keep dispatch policy separate from kernel correctness. Never widen an automatic dispatch range based only on a benchmark or parity between two optimized kernels.
- Use independent references where possible: compare batched rows with repeated single-row production calls, and use representative format-valid fixtures rather than arbitrary byte patterns alone.
- Keep performance claims scoped to the exact device, shape, data format, warmup, and measurement method. Isolated kernel timings are not end-to-end model throughput.
- Do not claim local hardware validation unless the corresponding test was actually run on that hardware.
- Do not claim a capability is implemented, validated, stable, accelerated, or distributable merely because its API, descriptor, or provider declaration exists.

## Current Metal Q4_K dispatch policy

- Batch size 1 uses the single-row Q4_K path.
- Batch sizes 2–15 use B4 (the current production correctness-validated range).
- Batch sizes above 15 must use MR4; preserve the B4 threshold at 15 until broader correctness and performance evidence justifies changing it.
- Explicit forced B4/MR4 selection is for opt-in tests/benchmarks, not a production routing API.
- A benchmark comparing B4 against MR4 does not establish correctness against an independent reference.
- Any change to this policy must include production-entry-point tests comparing every batched output row against repeated single-row calls, including boundary sizes 15, 16, and 17 and larger batches.

## Tests and validation

- Run the narrowest relevant unit test first, then the owning module's test task, followed by broader checks when practical.
- Metal GPU tests require macOS with a working Metal device and a freshly built native library. CI on non-Apple hosts cannot substitute for those hardware checks.
- Before interpreting a Metal test result, verify which dylib was loaded; prefer the module-local build output over an installed copy to avoid stale-library false results.
- For Q4_K production-dispatch changes, exercise the public production batch binding, compare each row to a repeated batch-size-1 invocation, assert finite outputs, and use a documented tolerance.
- Keep synthetic deterministic fixtures for reproducibility, but add format-valid representative quantization data and multiple shapes before making broad claims.
- Benchmark only after correctness checks pass. Include warmups, repeated samples, median/percentile reporting, and CSV output where the project already follows that convention.
- Document validation status, known limitations, and exact hardware/test scope alongside performance changes.
- If a build or test cannot be run in the available environment, state that explicitly; never imply it passed.

## Repository workflow

- Work on the task branch requested by the maintainer. Do not force-push, reset, or overwrite divergent work.
- Preserve uncommitted work and backups. If local and remote histories diverge, inspect and reconcile explicitly rather than discarding either side.
- Keep commits focused and messages descriptive. Do not merge or change the default branch unless explicitly requested.
- Ask the maintainer to run local tests only when the result depends on their Apple Silicon/Metal hardware or other unavailable local environment; complete code inspection, implementation, documentation, and CI-oriented work autonomously first.
- Never report a test, build, benchmark, or hardware validation as passing unless its output was observed.

## Developer experience and extension ecosystem

Aljabr is a Java-native computational framework for research, development, education and production use. Its intended breadth is comparable to foundational parts of NumPy, SciPy, PyTorch and the surrounding scientific Python ecosystem, without promising Python API compatibility.

- Keep the same public Java APIs usable from ordinary Java applications, Gradle/Maven projects, JBang scripts and JShell. Add Jupyter integration where it can reuse the same API; do not create a notebook-only programming model.
- Provide small, executable examples for numerical computing, tensors, linear algebra, statistics, classical ML, neural networks and model execution. Examples must state which artifacts/providers they require.
- Treat JDK 25 as the currently documented project baseline unless a reviewed decision changes it. Verify each downstream artifact's actual bytecode/runtime requirements; do not promise compatibility with older JDKs without testing.
- Use the Java Vector API for optional CPU SIMD implementations. It is an implementation detail behind stable Aljabr contracts, not a required dependency of core consumers. Since it is incubating in the documented JDK 25 baseline, isolate its module/compiler/runtime flags to modules that use it.
- Use FFM for optional native memory and ABI integration where appropriate. Keep layouts, arenas/scopes, handles, alignment, native access permissions and cleanup behind well-defined adapters. FFM use must not force unrelated numerical or statistics consumers to load native libraries. Avoid introducing new JNI dependencies unless an ADR explains why.
- Keep a portable scalar/reference implementation wherever practical. Optimized Vector API, parallel, FFM/native and accelerator paths must pass the same applicable correctness/conformance tests before performance claims are made.
- Separate numerical dtype from storage encoding and quantization format. Define promotion, accumulation precision, casting, overflow and tolerance behavior independently of backend implementation.
- A DataFrame/Series and tabular-data API is a valuable independent data-science extension, not part of the minimal tensor core. It may integrate with Aljabr tensors through explicit conversions/adapters. Keep CSV/JSON/Parquet/Arrow/SQL connectors and large data engines optional; do not make tensor or statistics consumers depend on them.
- DataFrame operations (selection, filtering, joins, grouping, aggregation, missing values, reshape and window operations) belong to the tabular-data layer. General probability, inference tests and distributions belong to statistical computing. ML feature transforms/pipelines belong to the ML/data-preparation layer. Avoid duplicating algorithms across these boundaries.
- Do not create a repository or Gradle project for every named Python analogue. Prefer cohesive, independently consumable artifacts and split only when API ownership, dependency isolation, maintainers and tests justify it.
- External libraries may be integrated behind Aljabr-owned adapters after license, JDK, numerical behavior, dependency and native-runtime review. Do not recreate mature libraries solely to imitate Python package names.

## Capability and roadmap status language

Use explicit status for roadmap and documentation claims:
- **Target:** intended architecture; not implementation evidence.
- **Planned:** accepted work with no implementation yet.
- **In progress:** implementation exists but acceptance criteria are incomplete.
- **Implemented:** code exists and focused tests pass.
- **Validated:** relevant conformance and compatibility tests pass in the stated environment.
- **Stable:** public compatibility and support policy are established.
- **Optimized:** benchmark evidence shows an improvement under documented conditions.

Never equate source presence, Gradle inclusion, provider declaration, or a passing compilation with a validated capability. Keep capability inventories dated and distinguish source inspection from executed tests.

## Autograd and optimizer boundaries

- Tensor autograd is a reusable numerical capability, not a training workflow. The target module is `:plugins:ml:aljabr-autograd` at `plugins/ml/aljabr-autograd`; keep the established Java package `tech.kayys.aljabr.autograd` unless a reviewed compatibility decision requires otherwise.
- Do not treat the legacy `plugins/ml/autograd` project as a working tensor autograd engine. Its current graph engine operates on external IR graph types, has incomplete gradient rules, and is excluded from the default build due to unpublished IR/trainer snapshot dependencies. Audit actual consumers before migrating or removing legacy code.
- The tensor module must not depend on the autograd implementation. Design a narrow, acyclic integration point for operation recording and backward execution. Keep backend implementation details, native memory types and accelerator-specific buffers out of autograd's public contracts.
- Define graph ownership, saved-tensor lifetime, gradient accumulation, scalar/non-scalar backward seeding, no-grad scopes, broadcasting reduction, unsupported-operation behavior and repeated-backward semantics before broad implementation.
- Validate every supported gradient rule with analytical tests and finite-difference checks. A forward operator existing does not imply its derivative is implemented. Missing rules must fail explicitly; never silently skip gradient propagation.
- Optimizer algorithms are independent of graph construction and belong in a separately consumable `:plugins:ml:aljabr-optim` module after parameter update/resource ownership semantics are correct. Optimizers should consume gradients without requiring a dependency on the autograd engine.
- Tafkir owns training workflow orchestration, datasets/pipelines, experiment lifecycle and fine-tuning. Do not put trainers or training loops in the autograd module.
- Do not bulk-move the legacy autograd project. Audit trainers, data loaders, metrics, augmentation, model hub/export and IR graph differentiation independently; move each only to an established owning module with real consumers and tests.
- See [ADR-0004: Tensor Autograd and Optimization Module Boundaries](docs/architecture/ADR-0004-AUTOGRAD-AND-OPTIMIZATION-BOUNDARIES.md) for the migration sequence and acceptance criteria.


## Triad ecosystem alignment: Aljabr, Tafkir, and AQLI

The product-level direction for the Java AI/ML ecosystem is described in the [Triad Ecosystem Vision](https://github.com/andalus-ai/aqli/blob/main/docs/architecture/TRIAD_ECOSYSTEM_VISION.md). Aljabr is one independently usable product in that triad; do not interpret the ecosystem goal as permission to absorb the responsibilities of Tafkir or AQLI.

- **Aljabr owns the reusable framework and compute foundation:** tensor and numerical semantics, operators/kernels, model abstractions and reusable model execution, formats, tokenizers, autograd/optimizers and ML primitives where implemented, backend/provider SPIs, and optional native/accelerated providers.
- **Tafkir owns training workflows:** training loops, fine-tuning/adaptation orchestration, datasets/batching pipelines, evaluation workflows, and training lifecycle built on Aljabr. Aljabr may provide reusable algorithms and primitives but must not own Tafkir's workflow policy.
- **AQLI owns inference and production serving:** model serving, request scheduling, inference runtime lifecycle, deployment, streaming, service APIs, and production operations. Aljabr provides reusable computation/model-execution primitives, not the serving control plane.
- Aljabr must remain independently consumable without AQLI or Tafkir. Never introduce a dependency on either project into foundational Aljabr modules.
- The triad should support embedded Java libraries, Java SDKs, CLI tools, JBang/scripts, JShell/Jupyter workflows, and production AQLI runtime deployments. Aljabr APIs should be usable directly from Java applications and scripts without launching an AQLI process.
- Keep adapters and entry points thin: CLI, notebooks, REST, and gRPC must not become independent owners of computation or model semantics.
- Aim for practical capability coverage of workflows commonly served by PyTorch, Hugging Face Transformers, and scikit-learn, but do not claim blanket or drop-in parity. Track each capability as implemented, partial, experimental, planned, or unsupported, with tests, examples, provider/format constraints, and known limitations.
- Framework-generic numerical/statistical/ML algorithms belong in Aljabr only when they are reusable framework capabilities. Training workflow orchestration belongs in Tafkir; inference-serving and production policy belong in AQLI.
- Prefer portable, documented public contracts and artifacts so work developed with Aljabr/Tafkir can be consumed by AQLI without duplicate compute stacks or format-specific ownership confusion.

## 16. Unified Java-native scientific and AI development experience

Aljabr is not only an ML/AI engine. It is a general-purpose Java computational framework spanning mathematics, numerical computing, statistics, scientific computing, data abstractions and machine learning. AI/ML tensors are a first-class capability and must receive deep investment, but they must not force unrelated scientific consumers to depend on neural-network, model, accelerator or serving modules.

The target is one coherent Aljabr-owned programming model across these capabilities, with independently selectable artifacts and optional providers. Do not reproduce Python package fragmentation as a set of unrelated Java APIs. Integrate useful existing Java/scientific libraries through replaceable adapters while retaining Aljabr-owned public types, semantics, diagnostics and lifecycle rules.

### Programming styles

Support several idiomatic Java entry styles over the same underlying contracts; these are complementary facades, not separate frameworks:

- **Traditional/object-oriented:** explicit types, constructors, interfaces and ordinary Java control flow for maximum transparency and debuggability.
- **Fluent/compositional:** builders and method chaining for readable pipelines, tensor expressions, model composition and configuration.
- **Declarative/annotation-assisted:** annotations and generated metadata for model discovery, parameter/configuration metadata, validation and documentation. Annotation processing or generated code is preferred over reflection-heavy hot paths.
- **Functional/composable:** typed functions and transformations for datasets, preprocessing, operators and scientific workflows.

Never require annotations, builders or a DSL to use the framework. Every convenience style must preserve an escape hatch to the explicit API and must not create hidden execution semantics.

### Memory, ownership and zero-copy

Treat resource safety as part of the public programming model, not an implementation detail. Document and test ownership, borrowing, aliasing, mutation, close/lifetime, thread access, cancellation and cleanup for heap, off-heap, mapped, native and device storage. Make view-versus-copy behavior observable. Provide safe defaults plus explicit advanced control through scoped arenas/allocators, memory pools, mapped storage, pinned memory and asynchronous transfers where supported. Zero-copy is a capability with precise preconditions, not a blanket promise; expose when conversion, materialization or device transfer occurs.

### Model, algorithm and runner extensibility

Make it straightforward for external projects to add a model family, scientific algorithm, operator, format adapter, dataset source or execution provider using small versioned Aljabr-owned contracts. Provide extension templates/examples, capability descriptors, registration/discovery, lifecycle rules, diagnostics and conformance test kits. Keep model architecture/weight-format compatibility separate from execution-provider support.

A reusable Aljabr model runner is a library-level composition/execution facility, not a serving control plane. It may provide model construction, input/output contracts, preprocessing/postprocessing composition, execution options, inspection and resource lifecycle. AQLI continues to own production inference serving, request scheduling, streaming and deployment operations. Tafkir owns higher-level training loops/workflows, dataset orchestration and adaptation policy; Aljabr owns reusable tensor, autograd, loss, optimizer and training-computation primitives.

### Multimodal and generative capability

Multimodal is a foundational model-composition concern, not a text-only special case. Architecture capabilities should cover text, image, audio, video, spatial/3D and combinations, including modality encoders, projections, fusion, cross-attention, temporal/spatial representations and modality-aware preprocessing.

Diffusion and other iterative generative families are first-class model/algorithm capabilities alongside autoregressive models, embedding models, classifiers and discriminative models. Provide reusable noise schedules, samplers/solvers, timestep/conditioning abstractions, latent/tensor workflows and scheduler-independent model contracts where justified. Do not put diffusion-specific semantics into the generic Tensor API or assume every generative model belongs to AQLI's token-generation loop. Keep model-family implementations optional and independently consumable.

### Developer workflows and ecosystem compatibility

The same published Java APIs must work in ordinary applications, Gradle/Maven projects, JShell, JBang and Jupyter through established Java kernels (for example IJava). These are different entry points to one API, not separate runtimes. Provide tested examples, dependency snippets, expected output, lifecycle guidance and promotion paths from an experiment to a production project. Do not make a custom notebook kernel a prerequisite.

Interop with Java libraries and data formats is encouraged through explicit adapters and conversion boundaries. Document dtype, null/missing-value, shape, layout, ownership, aliasing and copy semantics. Third-party types, exceptions and lifecycle assumptions must not leak into stable Aljabr public APIs.

### Packaging, profiles and tree-shaking

Prefer dependency-graph minimization over claims of automatic tree-shaking. Publish cohesive feature artifacts; a BOM aligns versions only; optional starter bundles provide curated combinations. Define documented consumer profiles (for example numerics, scientific, statistics/data, tensor, ML, multimodal, training-primitives, CPU, accelerator, research and embedded/minimal) as dependency recipes or tested distribution bundles, not mandatory runtime modes. Keep providers, model catalogues, dataset connectors, native runtimes and training-only features opt-in.

JPMS/jlink, GraalVM Native Image and bytecode/resource shrinking may improve packaging after module metadata, reflection, ServiceLoader, FFM, resources and provider discovery are tested. Do not promise that a BOM removes dependencies or that class-level tree-shaking can compensate for a monolithic artifact. Test clean external Gradle and Maven consumers for each supported profile, including dependency reports, published metadata, startup/runtime behavior and optional-provider absence.

### Developer tooling and diagnostics

Treat tooling as a product capability: model/algorithm inspectors, graph and shape inspection, parameter/state summaries, numerical assertions and gradient checks, NaN/Inf diagnostics, backend/provider explainability, memory/copy/transfer visibility, per-operation profiling and JFR integration. Diagnostics must identify the operation, input shapes/dtypes/layouts, violated constraint, selected provider and actionable remediation without leaking provider-specific exceptions as the public contract.

### Scientific capability breadth

The framework scope includes numerical foundations, math, linear algebra, probability, statistics, optimization, scientific algorithms, structured/tabular data, dataset access/transforms, classical ML, tensors, autograd, neural networks, multimodal and generative models. Keep these capabilities cohesive and independently selectable. Dataset abstractions in Aljabr should cover reusable in-memory/streaming source and transformation contracts needed by computational workflows; Tafkir owns higher-level training dataset pipelines and experiment orchestration. DataFrame semantics remain an optional tabular extension, not part of the minimal tensor core.

### Acceptance gates

A new public capability or convenience layer must demonstrate: (1) Aljabr-owned semantics and API; (2) a useful independent dependency boundary; (3) explicit memory/lifecycle and error behavior; (4) reference correctness and conformance tests; (5) tested examples in the relevant Java workflows; (6) compatibility with supported profiles and published artifacts; and (7) no inversion of AQLI/Tafkir/product ownership. Mark target, planned, implemented, validated, stable and optimized status separately.

## 17. SDK audit execution gate

Before adding new public SDK abstractions or reorganizing modules, consult `docs/architecture/SDK_CAPABILITY_AUDIT.md`. Treat its findings as an initial repository inventory, not a certification of runtime support. Verify current source, implementations, dependency metadata, tests and external-consumer behavior before changing contracts. Use the audit's status vocabulary (present, implemented, tested, validated, supported, experimental, planned, legacy) consistently. Prefer small compatibility-preserving increments and keep the Tensor/memory API, model-extension path and publication graph evidence-driven.

## 18. Developer tooling implementation gate

Use `docs/architecture/DEVELOPER_TOOLING.md` as the source of truth for developer-facing diagnostics, inspection, profiling, memory tracing and numerical validation. Keep diagnostics contracts dependency-light and optional. Measurements must distinguish measured from estimated, host from device, and asynchronous launch from synchronized execution. Never expose tensor contents or sensitive inputs by default. Prefer JFR/JMH and optional ecosystem adapters over custom replacements. Do not introduce a monolithic tooling artifact or couple Aljabr core to AQLI, Tafkir, notebooks, dashboards or exporters.


## 19. Seven engineering principles — mandatory design and review rules

Apply these principles proportionally. They are decision rules for reviewing real changes, not a requirement to introduce an interface, module, abstraction or design pattern for every noun.

### 19.1 Separation of concerns and ownership

Every component must have one clear primary responsibility and an identifiable owning layer:

| Concern | Aljabr ownership | Boundary |
|---|---|---|
| Public SDK/API | Stable Java types, computational semantics and library contracts | Entry points adapt to the API; they do not fork its semantics |
| CLI and developer tools | Thin command parsing, validation, presentation and diagnostics | No kernels, model execution engines, provider implementations or duplicated business rules in command handlers |
| Model construction/loading | Reusable model configuration, architecture composition, format adapters, tokenizer/preprocessor contracts and weight decoding/loading primitives | Keep file-format parsing separate from architecture construction and execution; CLI delegates. AQLI owns serving-time model/session lifecycle and loading orchestration |
| Execution | Reusable operation/model execution plans, execution contexts and execution primitives | Keep execution mechanics separate from product workflow and deployment lifecycle |
| Provider/backend integration | Provider SPI, capability declarations, compatibility evaluation, conformance, resource adapters and backend implementations | Aljabr exposes explainable selection/initialization mechanisms; the consuming application/runtime owns product-specific policy. Implementations depend on stable contracts; public APIs do not depend on a concrete backend |
| Scheduling | Local execution ordering/resource coordination only where it is part of a reusable computation contract | AQLI owns inference-request scheduling, admission, batching policy and serving lifecycle; Aljabr does not own cluster/cloud scheduling |
| Streaming | Reusable data/operation stream contracts where they are computational primitives | AQLI owns token/response streaming, client backpressure and serving cancellation policy |
| Configuration | Each setting has one authoritative owner, schema, default and validation point | Aljabr owns provider technical configuration and capability constraints; AQLI owns serving/runtime profiles and provider-selection policy; adapters translate rather than shadow settings |

Do not infer ownership from a package name or current file location. Verify callers, dependencies, lifecycle and consumers before moving code. A CLI command may orchestrate calls, but must not become the owner of the underlying computation. A model loader must not become a model runner, and a model runner must not become a serving runtime.

### 19.2 SOLID, applied pragmatically

- **Single Responsibility:** split a class when it has independent reasons to change, not merely because it has multiple methods.
- **Open/Closed:** use existing SPIs and composition for genuine extension points; do not make every implementation pluggable by default.
- **Liskov Substitution:** implementations of a public contract must preserve its documented semantics, errors, lifecycle and resource guarantees.
- **Interface Segregation:** keep consumer-facing contracts focused; do not force simple consumers to implement unrelated lifecycle, diagnostics or optional capability methods.
- **Dependency Inversion:** domain/public contracts must not depend on CLI, concrete providers, native runtimes or product-specific orchestration.

Prefer a direct cohesive class over a speculative interface hierarchy. Introduce an abstraction when there are multiple real implementations, a stable extension boundary, a test seam, or a demonstrated reduction in coupling.

### 19.3 Single source of truth

- One canonical public contract for each concept: tensor/storage ownership, model identity/configuration, format metadata, capability IDs, provider lifecycle, execution options, errors and streaming semantics.
- One authoritative owner for each configuration value, default, precedence rule and validation rule. Document overrides explicitly; do not silently keep parallel copies.
- Reuse canonical registries, descriptors, parsers, validators and conversion paths. Thin adapters may translate external representations but must delegate to the canonical implementation.
- Do not duplicate model-family logic in format loaders, CLI commands, backend implementations or downstream products.
- When two existing abstractions overlap, map their callers and compatibility constraints, choose the canonical direction, and migrate incrementally. Do not add a third abstraction to hide the conflict.
- Generated metadata and documentation should derive from authoritative declarations where practical; avoid hand-maintained competing capability inventories.

### 19.4 Correct placement

Place code according to responsibility and dependency direction, not convenience or historical accident:

- Public contracts belong in the smallest stable API/SPI artifact that owns them.
- Provider-specific kernels, native bindings and device resources belong in the provider implementation.
- Format parsing and format-specific weight decoding belong in format adapters; architecture/model composition belongs in model modules.
- CLI parsing/presentation belongs in CLI modules; reusable logic belongs in library modules.
- Tests belong beside the owning module and should distinguish unit, contract/conformance, integration and hardware-specific coverage.
- Test fixtures belong in the narrowest reusable test-fixture source set; do not make production artifacts depend on test utilities.
- Documentation belongs with its audience: root guidance for repository-wide rules, module README/AGENTS for local contracts, architecture/ADR documents for cross-module decisions, and user guides for supported workflows.
- Resources, service descriptors, native binaries and generated sources belong to the module that owns their lifecycle and packaging.

Before moving a type, inspect package/API compatibility, Gradle dependency edges, resource lookup, ServiceLoader metadata, test ownership, published artifact consumers and documentation links.

### 19.5 Reusable, cohesive code

- Prefer composition and small cohesive components over god classes, catch-all managers and utility dumping grounds.
- Keep pure transformations separate from I/O, mutable state, provider lifecycle and orchestration where doing so improves testability or reuse.
- Avoid framework-specific copies of generic numerical, tensor, model or format logic. Put shared behavior behind the canonical Aljabr contract and keep framework adapters thin.
- Do not extract one-use helpers into public APIs without a real reuse or extension case.
- Avoid both extremes: duplicated implementations and over-generalized frameworks. Reuse should reduce semantic drift, not create indirection without value.

### 19.6 Developer experience

- Make the supported module/API entry point discoverable from README, module docs and runnable examples.
- Use consistent names for the same concept across packages, CLI options, configuration, diagnostics and documentation.
- Document prerequisites, dependency coordinates, exact commands, expected output, cleanup and known limitations.
- Keep public APIs idiomatic and composable; preserve explicit escape hatches behind convenience builders, annotations or fluent APIs.
- Provide actionable errors with the failed operation, relevant input/capability constraint, selected provider where applicable, and a practical remediation.
- Make provider selection, copies/transfers, resource use and execution behavior inspectable without requiring provider-specific knowledge.
- Keep examples runnable against published artifacts; label source-tree-only examples and planned APIs clearly.

### 19.7 User experience and operational predictability

- CLI commands must be predictable, discoverable and consistent. Validate arguments early, provide useful help, stable exit behavior and actionable errors. CLI commands are adapters, not alternate owners of SDK behavior.
- Configuration must have documented names, types, defaults, precedence and ownership. Reject conflicting or unsupported combinations instead of silently ignoring them.
- Execution should expose meaningful progress and diagnostics without leaking implementation exceptions or overwhelming normal users.
- Streaming must define ordering, completion, cancellation, backpressure, partial failure and resource cleanup. Never report successful completion before the stream and owned resources reach their defined terminal state.
- Provider fallback must be explicit and explainable; never silently violate requested device, precision, determinism, memory or compatibility constraints.
- Resource ownership and close behavior must be predictable on success, failure and cancellation.
- Distinguish unsupported, unavailable, incompatible and failed capabilities in user-facing errors.

### 19.8 Review checklist

For a structural, API, execution, provider, CLI or documentation change, reviewers should be able to answer:

1. What responsibility does this change own, and what explicitly remains elsewhere?
2. Is there already a canonical API, implementation, configuration or descriptor to reuse?
3. Does the dependency direction remain acyclic and provider/framework neutral?
4. Is each class, package, module, resource, test and document in its proper home?
5. Does the design introduce an abstraction only where it earns its cost?
6. Can a consumer discover, configure, execute, diagnose and clean up the capability?
7. Are error, cancellation, streaming and fallback behaviors explicit and tested?
8. Are claims about implementation, validation and support backed by evidence?

If a question is not applicable, say why in the PR rather than adding unnecessary machinery. These principles supplement the existing test, compatibility, provider, memory and hardware-specific rules; they do not override them.


## 20. Diagnostics, observability, and auditability — mandatory framework capability

Treat diagnostics, observability, and auditability as a first-class Aljabr capability across numerical, tensor, model, format, memory, execution, and provider layers. This is not permission to add logging calls indiscriminately or to build a serving control plane.

### Ownership and boundaries

- Aljabr owns framework-level diagnostic semantics, stable diagnostic codes, computational execution events, provider decision explanations, resource/memory lifecycle observations, numerical validation findings, and reproducible execution evidence.
- AQLI owns request/serving telemetry, admission and scheduling decisions, deployment/operational audit, service-level objectives, and production request correlation. Aljabr may accept an explicit correlation context and emit framework events, but must not own AQLI policy or depend on AQLI.
- Tafkir owns experiment/training workflow history and dataset/training lifecycle audit. Aljabr emits reusable computation and optimizer events only where those are genuine framework primitives.
- Farabi and presentation products own research governance, analytics interpretation, dashboards and visualization. Aljabr supplies structured evidence, not a dashboard or governance workflow.
- Keep the diagnostics API JDK-only or dependent on the smallest existing stable API. Logging, metrics, tracing, JFR, OpenTelemetry, Micrometer, JSON/CSV exporters and UI integrations are optional adapters.
- Never require a global singleton, background thread, network exporter, file output, dashboard, agent, or instrumentation dependency for ordinary tensor/model consumers. The disabled/no-op path must be cheap and allocation-conscious.

### Required observability signals

Every relevant capability should expose appropriate, bounded, opt-in signals for:
- **Execution:** operation/model scope start, completion, failure, duration, execution identity, and parent/correlation IDs.
- **Provider decisions:** requested constraints, candidate providers, compatibility/rejection reasons, selected provider, fallback path and initialization outcome. Never silently fall back across an explicit user constraint.
- **Numerics:** shape/dtype/layout validation, non-finite/overflow findings, tolerance/parity results and gradient-check outcomes. Value scans that can synchronize or copy device data must be explicit.
- **Memory and ownership:** allocation/release, retain/borrow/view, copy/materialization, pool reuse, host/device transfer, failure and lifecycle violations, with bytes and storage identity only when known.
- **Performance:** operation/provider/kernel timing, counters and distributions with clock source, warmup policy, sample count, synchronization policy, and host/device attribution.
- **Model and format loading:** format/version metadata validation, tensor mapping/shape mismatches, missing/unexpected weights, conversion decisions and unsupported architecture/operation reasons. Do not expose weight contents.
- **Capability/provider lifecycle:** discovery, declared/installed/available/compatible/validated state transitions, initialization, shutdown and cleanup failures.

Do not instrument every hot-path operation unconditionally. Define explicit scopes, sampling, filters and bounded buffers; ensure event ordering and context propagation are documented for concurrent/asynchronous work.

### Structured diagnostic and audit records

- Diagnostics are immutable structured records with stable code, schema version, severity, category, operation/component, concise message, explanation, remediation, timestamp/clock semantics, correlation/parent IDs and typed context.
- Include relevant input metadata (shape, dtype, layout, device, operation and provider) only when available. Mark missing values as unavailable; never guess.
- Preserve the original cause for programmatic debugging while keeping public exception and diagnostic contracts provider-neutral.
- Distinguish diagnostic findings from audit events and measurements: a warning is not an audit record; a timing sample is not proof of correctness; a provider declaration is not validation evidence.
- Audit events are append-only facts about framework-relevant decisions and lifecycle transitions. Each event identifies actor/source (framework, caller, provider), action, target/resource, outcome, timestamp, correlation ID and schema version. Include before/after values only for safe configuration metadata and only when policy permits.
- Audit history must be bounded/retention-controlled by the consuming application or exporter. Do not silently persist records, create a hidden database, or claim tamper-proof/non-repudiation guarantees. Cryptographic integrity, retention, access control and compliance policy belong to an explicitly selected audit sink/product.
- Never capture tensor values, prompts, user data, credentials, secrets, raw model weights, or arbitrary configuration payloads by default. Redaction and allowlisting are mandatory at export boundaries. Sensitive capture requires explicit opt-in and a documented scope.
- Keep event/diagnostic schemas versioned and deterministic. JSON/CSV/JFR exporters must preserve units, provenance and measured/estimated/unavailable status.

### Evidence and provenance rules

- Every measurement must say measured, estimated, sampled, derived or unavailable; identify host/device, clock, unit, scope (inclusive/exclusive), and synchronization semantics.
- Distinguish wall time, CPU time, asynchronous launch latency and synchronized device execution time. Never compare them as equivalent.
- Memory reports must distinguish logical tensor bytes, physical allocation bytes, retained/peak bytes and estimates. Never label an estimate as measured usage.
- Provider/model compatibility and validation reports must include the tested API/provider/model/format versions and relevant environment constraints.
- Numerical audit reports should record reference implementation, seed/determinism conditions, tolerance, comparison count, mismatch count and max absolute/relative error where applicable.
- Performance reports should record build/runtime/JDK, provider/device, shape/dtype, warmup, sample count, statistic and benchmark method. JMH remains the preferred JVM microbenchmark harness.
- Do not call an execution reproducible unless inputs/configuration, seeds, provider, versions and relevant nondeterminism constraints are captured sufficiently to reproduce it.

### Review and acceptance gate

For any new or changed public operation, model, format, memory path or provider:
1. Identify the relevant diagnostic/error codes and failure context.
2. Explain provider selection, fallback and unsupported-capability outcomes.
3. Cover resource allocation/ownership/cleanup and failure/cancellation paths.
4. Add opt-in observability at meaningful boundaries without hidden hot-path cost.
5. Add audit events for consequential framework decisions and lifecycle transitions where evidence is needed.
6. Test disabled/no-op behavior, event ordering/correlation, boundedness, redaction, exception paths and concurrency as applicable.
7. Document what is measured versus estimated and which exporter/adapters are optional.
8. State exactly which tests were run; source presence or compilation is not validation.

A feature is not observability-ready merely because it logs an exception or exposes a metrics counter. It must be diagnosable, explainable, privacy-safe, bounded, testable and useful without coupling core computation to a product runtime.


### 20.1 Existing diagnostics foundation — reuse before extending

The canonical starting point is `core:aljabr-diagnostics`. Extend its existing `Diagnostic`, `DiagnosticSeverity`, `DiagnosticSink` and `DiagnosticFormatter` contracts; do not create a competing diagnostics artifact or duplicate the existing API.

New integrations should prefer `DiagnosticContext` and `Diagnostic.withContext(...)` for bounded correlation/parent IDs and explicit operation/component/provider metadata. Keep existing map-based constructors source-compatible. Use `DiagnosticSink.noop()` (the shared `DiagnosticSink.NOOP`) for disabled diagnostics. Do not introduce implicit ThreadLocal context propagation or capture unreviewed payloads as context attributes.


### 20.2 Explicit execution lifecycle scopes

The diagnostics foundation now includes explicit execution lifecycle contracts in `core:aljabr-diagnostics`:

- Use `ExecutionScope.start(context, observer)` around meaningful computation boundaries. The context must explicitly provide a correlation ID and operation; pass parent IDs when creating child scopes.
- Emit STARTED followed by exactly one terminal event: SUCCEEDED, FAILED or ABANDONED. Closing an unfinished scope records ABANDONED; closing a completed scope is harmless.
- Use monotonic elapsed nanoseconds for duration. Wall-clock timestamps identify event occurrence only; do not derive elapsed time from wall-clock subtraction.
- Failure events expose exception class identity only. Do not export exception messages or stack traces through lifecycle context by default.
- Observer callbacks are synchronous and should be fast/non-blocking. Runtime exceptions are isolated and retained on the scope through `observerFailure()`; fatal errors are not swallowed.
- Context propagation is explicit across nested and asynchronous work. Do not add ThreadLocal propagation, background dispatch, implicit buffers or persistence.
- Lifecycle events are observations, not diagnostics, audit facts or metric samples. Add separate contracts only when their semantics and delivery guarantees are defined.
- Test start/terminal ordering, abandoned scopes, observer failure isolation, privacy, duplicate completion and context requirements. Keep observer-disabled paths cheap.
