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
- [Framework architecture](architecture/FRAMEWORK_ARCHITECTURE.md)
- [Computational framework scope](architecture/COMPUTATIONAL_FRAMEWORK_SCOPE.md)
- [Modular distribution](architecture/MODULAR_DISTRIBUTION.md)
- [Computational capability inventory](architecture/COMPUTATIONAL_CAPABILITY_INVENTORY.md)
- [Research and training readiness](architecture/RESEARCH_TRAINING_READINESS.md)
