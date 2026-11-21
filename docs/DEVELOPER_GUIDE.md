# Aljabr Developer Guide

**Status:** Contributor and extension-author guidance  
**Java baseline:** JDK 25 (verify each artifact's declared requirements)

This guide complements the repository-root AGENTS.md, which remains the authoritative instruction set for coding agents and contributors. The architecture documents describe target boundaries; this guide explains how to make changes safely.

## 1. Understand the project

Aljabr is a Java-native computational foundation for numerical, mathematical, statistical, scientific-computing and machine-learning workloads. It is not an inference-serving product, training workflow orchestrator, BI application or mandatory all-in-one runtime.

- AQLI owns inference serving and deployment/runtime policy.
- Tafkir owns training workflow and fine-tuning orchestration.
- Farabi owns research/analytics workflows, governance, interpretation and reporting.
- Independent extensions may provide tabular data, statistics, classical ML, NLP, visualization adapters and domain-specific scientific algorithms.

Aljabr owns reusable computation, public contracts, numerical semantics, capability metadata and provider/conformance contracts.

## 2. Before changing code

1. Read root AGENTS.md and the owning module's local AGENTS.md, if present.
2. Inspect the actual Gradle project, source packages, module descriptor, build file, tests and callers.
3. Search for existing contracts and implementations before adding a new abstraction.
4. Check dependency direction and public API leakage.
5. Identify compatibility impact and whether an ADR is needed.
6. Record what is implemented versus merely proposed. Do not infer support from class names or project inclusion.

Do not create a new Gradle project, repository, SPI or parallel tensor/model abstraction until the existing code and consumers have been mapped.

## 3. Public API and module design

- Prefer narrow, stable Java interfaces and immutable configuration/metadata.
- Keep implementation types, native handles and third-party library types out of public APIs unless explicitly approved.
- Use api dependencies only when dependency types appear in a module's public signatures; otherwise use implementation.
- Keep foundations independent of AQLI, Tafkir, Farabi, CDI, serving frameworks and accelerator SDKs.
- Keep domain extensions independently consumable. Statistics must not require neural networks or GPUs; tabular-data consumers must not require model serving.
- Use ServiceLoader and explicit registration where suitable. Keep discovery explainable and testable.
- Define typed errors for unsupported operations, dtype/shape/device incompatibility and provider initialization failures.

## 4. Numerical and tensor correctness

For every public operation, document and test:
- Input/output shape and layout.
- Supported dtypes and promotion/casting rules.
- Accumulation precision and numerical tolerances.
- NaN, infinity, overflow and degenerate-input behavior.
- Determinism and random-number behavior where relevant.
- Aliasing, allocation, mutation and device-copy behavior.
- Thread safety, cancellation and resource ownership.

Prefer an independent scalar/reference implementation as a correctness oracle. Compare optimized implementations against it across representative shapes, dtypes and edge cases.

## 5. Performance, Vector API and FFM

- Keep the portable scalar CPU path available.
- Use the Java Vector API only in optional CPU acceleration code and only where benchmarks justify it. Under JDK 25, isolate incubator module/compiler/runtime flags to modules that use the API.
- Use FFM behind narrow native integration contracts. Specify arena/scope lifetime, ownership, alignment, thread access, native error handling and cleanup.
- Do not require native libraries for unrelated CPU-only modules.
- Avoid new JNI integrations unless an ADR records the need and alternatives.
- Never trade correctness or defined numerical semantics for a benchmark result.
- Report hardware, JDK, OS, dtype, shapes, warmup, repetitions and measurement method.

## 6. Extensions and domain libraries

A new extension should state:
- The domain problem and intended consumers.
- Its public API and computational semantics.
- The smallest Aljabr artifacts it needs.
- Optional external libraries and their licenses/runtime requirements.
- Provider/capability needs, if any.
- Conformance, numerical reference and lifecycle tests.
- Why it deserves a separate artifact or repository rather than an existing module.

Examples of possible independent extensions include a DataFrame/Series library, statistics, scientific solvers, classical ML, NLP, time-series and visualization adapters. These are ecosystem directions, not permission to create empty or speculative modules.

A DataFrame extension owns tabular operations such as filtering, joins, grouping, aggregation, missing values and reshaping. Aljabr tensors own N-dimensional numerical computation. Convert between them explicitly and document copies, dtypes, layouts and null handling. Arrow, Parquet, SQL and large data engines remain optional integrations.

## 7. Developer experience and examples

- Keep ordinary Java applications first-class.
- Provide JBang examples with real, released dependency coordinates.
- Keep JShell usage compatible with the same published APIs.
- Add Jupyter examples only when they use the same library APIs.
- Examples must include prerequisites, exact commands, expected output and limitations.
- Test examples in CI where practical. Do not use placeholder API calls in documentation presented as runnable code.

## 8. Validation and reporting

Run the narrowest relevant tests, then owning-module tests and broader checks as practical. Hardware-specific validation must be performed on the relevant hardware.

Every PR should state:
- What changed and why.
- Public API/ABI and dependency impact.
- Tests actually executed and their results.
- Hardware/platform scope.
- Known gaps and migration notes.

Use the status terms from AGENTS.md: Target, Planned, In progress, Implemented, Validated, Stable and Optimized. Never claim validation or performance without observed evidence.

## 9. Related documents

- [Contributor instructions](../AGENTS.md)
- [Unified computational architecture and roadmap](architecture/ALJABR_UNIFIED_ARCHITECTURE.md)\n- [Framework architecture](architecture/FRAMEWORK_ARCHITECTURE.md)
- [Computational framework scope](architecture/COMPUTATIONAL_FRAMEWORK_SCOPE.md)
- [Modular distribution](architecture/MODULAR_DISTRIBUTION.md)
- [Computational capability inventory](architecture/COMPUTATIONAL_CAPABILITY_INVENTORY.md)
- [Research and training readiness](architecture/RESEARCH_TRAINING_READINESS.md)

## 10. Java API styles and development entry points

Aljabr APIs should be idiomatic Java and support different developer preferences over one semantic core:

- Traditional Java: explicit types, interfaces, constructors and control flow.
- Fluent Java: builders and method chaining for configuration, tensor/model composition and pipelines.
- Declarative/annotation-assisted Java: annotations and generated metadata for discovery, configuration, validation and documentation.
- Functional Java: typed transformations and composable operators.

These styles must interoperate and must not create separate execution semantics. Annotations are optional; generated metadata is preferred to runtime reflection in hot paths; explicit APIs remain available for debugging and advanced control.

The same published API should be usable from:
- Gradle/Maven applications and IDEs.
- JShell for short experiments.
- JBang for executable scripts and examples.
- Jupyter through an established Java kernel such as IJava.

Provide tested snippets with real dependency coordinates, setup instructions, expected output, cleanup behavior and known limitations. Show how to promote an experiment into a normal Java project. Do not require a custom notebook kernel or a separate notebook API.

## 11. Building model, algorithm and runner extensions

Extension authors should be able to implement a model family, algorithm, operator, format adapter, dataset source or backend without forking Aljabr or depending on AQLI/Tafkir. Start from the smallest relevant Aljabr API/SPI, declare stable capability IDs and compatibility, register explicitly or through ServiceLoader, and implement lifecycle, diagnostics and conformance tests.

A library-level model runner may compose a model, input/output contracts, preprocessing/postprocessing, execution options, inspection and resource lifecycle. It is not an HTTP server or serving runtime. AQLI owns request handling, scheduling, streaming, cancellation policy and production deployment. Tafkir owns training workflow orchestration, dataset pipelines and adaptation policy; Aljabr supplies reusable tensor/autograd/loss/optimizer computation primitives.

## 12. Memory, zero-copy and diagnostics

For every tensor/data/model API, explain ownership, borrowing, aliasing, mutation, view-versus-copy, lifetime, thread access and cleanup. Make copies, materialization and host/device transfers visible where practical. Zero-copy must list its preconditions and fallback behavior. Prefer safe defaults and offer explicit scoped allocation/pooling/mapped/native/device facilities for advanced users.

Use actionable Aljabr-owned diagnostics: identify the operation, input shape/dtype/layout, failed constraint, selected provider, resource context and remediation. Include numerical assertions, gradient checks, NaN/Inf detection, backend parity tests, memory/copy inspection and profiling examples where relevant.

## 13. Scientific, data and generative examples

Examples should cover numerical/math foundations, linear algebra, statistics, scientific computation, dataset transformations, classical ML, tensors/autograd, neural models, multimodal composition and diffusion/generative workflows as those capabilities become implemented and validated. Clearly label planned versus executable examples. Never present a target API as runnable until its dependency coordinates and example are tested.

## 14. Applying the seven engineering principles

The repository-root [AGENTS.md](../AGENTS.md#19-seven-engineering-principles--mandatory-design-and-review-rules) defines the normative rules. Use this section as the practical workflow when designing or reviewing a change.

### Start with ownership, not a proposed class or module

Trace the feature from its public entry point through configuration, model/format handling, execution, provider integration, resource lifecycle and tests. Identify which layer owns each decision. Keep CLI and other adapters thin; keep reusable computation in library modules; keep AQLI serving policy and Tafkir training workflow outside Aljabr.

A useful ownership test is: *if this interface or workflow changed, which product or layer should be responsible for the change?* If multiple layers independently own the same decision, clarify the contract before adding code.

### Reuse the canonical path

Before implementing behavior, search for existing public types, implementations, registries, descriptors, validators, error types and tests. Prefer one canonical implementation plus narrow adapters over parallel implementations. If a legacy and preferred abstraction overlap, document the migration path and preserve compatibility deliberately; do not silently fork semantics.

Configuration should follow the same rule. Every setting needs one owner, one documented default and validation/precedence behavior. A downstream product may expose a user-friendly profile, but it should translate to the owning library/runtime contract rather than maintain a second competing configuration model.

### Place artifacts by responsibility

Use the smallest owning module that can express the contract without depending on an implementation. Keep implementation, provider/native integration, CLI, test fixture and documentation concerns in their appropriate source sets and artifacts. Check Gradle api versus implementation, runtime dependencies, ServiceLoader resources, native packaging and published metadata before changing module boundaries.

Do not create a new module simply to make the directory tree look cleaner. Split only when an independently consumable capability, dependency boundary, lifecycle, release/support policy or meaningful ownership boundary justifies it.

### Apply SOLID without ceremony

Split classes when responsibilities have independent change drivers. Add interfaces for real substitution, external extension, stable module boundaries or useful test seams—not just to wrap a single implementation. Keep contracts narrow and preserve substitutability for semantics, errors and resource lifecycle. Prefer direct composition when it is clearer than a hierarchy.

### Verify the complete developer and user path

A capability is not complete merely because its implementation compiles. For the scope being changed, verify the relevant path:

1. **Discover:** Can a developer find the API, module, prerequisites and supported status?
2. **Configure:** Are names, defaults, precedence, validation and ownership consistent?
3. **Execute:** Does the public API/CLI delegate to the canonical implementation?
4. **Observe:** Can users understand selected providers, progress, copies/transfers and failures?
5. **Recover and clean up:** Are unsupported inputs, initialization failure, cancellation, stream termination and resource release well-defined?
6. **Extend:** Can another module/provider consume the contract without importing implementation internals?
7. **Validate:** Are focused tests, conformance tests, examples and external-consumer checks appropriate to the claim?

For streaming APIs, test normal completion, empty input, partial output, downstream cancellation, upstream failure, backpressure (where supported), and cleanup. For provider selection, test explicit selection, automatic selection, unavailable/incompatible providers and fallback diagnostics. For CLI changes, test help, invalid arguments, configuration conflicts, exit status and error remediation.

### Document evidence, not aspiration

Architecture documents describe intended boundaries and decisions. Capability inventories describe observed repository state. User guides describe workflows that can actually be followed. Keep these roles distinct, link them rather than copying large competing descriptions, and state whether a claim is target, planned, in progress, implemented, validated, stable or optimized. Record the exact test/platform scope; do not turn an API declaration or successful compilation into a support claim.
