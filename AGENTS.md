# AGENTS.md — Aljabr contributor guidance

## Product vision and scope

Aljabr is the reusable, Java-native computational foundation for mathematical, numerical, statistical, scientific-computing, and machine-learning workloads. It should provide a coherent public programming model and computational semantics, with multiple interchangeable execution providers.

The architectural principle is:

> One Aljabr programming model. One set of computational semantics. Multiple interchangeable execution providers. Independently extensible capabilities. From laptop to distributed infrastructure.

Aljabr owns:
- Public Java APIs and the developer-facing programming model.
- Computational semantics: data types, shapes, broadcasting, precision, determinism, memory/resource behavior, concurrency, and errors.
- Capability identity, metadata, registry contracts, and provider selection policy.
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
