# Aljabr Computational Framework Scope

**Status:** Proposed target architecture and capability taxonomy  
**Baseline:** 2026-11-01  
**Java baseline:** JDK 25

> This document expands Aljabr's target scope beyond machine learning into reusable mathematical, numerical, statistical, scientific-computing, ML and NLP foundations. It is an architecture proposal, not a claim that every capability or algorithm is currently implemented, tested, or published.

## 1. Decision and mission

Aljabr is the canonical reusable computational foundation for the Andalus ecosystem. It should provide composable, well-tested Java APIs for mathematical and numerical computation, statistics, machine learning, deep learning and foundational NLP.

The guiding rule is:

**Put reusable algorithms and computational primitives in Aljabr; put product workflows, policy, orchestration and user experience in the consuming product.**

Aljabr should be useful independently of Andalus agents, Farabi, AQLI or Tafkir. A basic statistics operation must not require a GPU, a model-serving runtime, a workflow engine or an agent framework.

## 2. Ownership boundaries across the ecosystem

| Project | Owns | Must not become |
|---|---|---|
| **Aljabr** | Reusable numerical, mathematical, statistical, ML, deep-learning and NLP computation; tensor/operator APIs; algorithm contracts; execution backends | BI product, agent runtime, HTTP serving product, tenant/governance authority or training workflow product |
| **[Farabi](https://github.com/andalus-ai/farabi)** | BI and research product; dataset catalog and lifecycle; semantic layer; query planning; permissions/governance; statistical research planning; provenance and interpretation; reports and dashboards | A competing implementation of canonical general-purpose algorithms |
| **[AQLI](https://github.com/andalus-ai/aqli)** | Model inference runtime; request lifecycle, scheduling, generation policy, streaming, serving APIs and deployment operations | Owner of general-purpose tensor, statistics or ML algorithms |
| **[Tafkir](https://github.com/andalus-ai/tafkir)** | Training/adaptation workflows; dataset pipelines; experiment orchestration; fine-tuning policy and lifecycle | A mandatory dependency for Aljabr core or ordinary inference |
| **[Andalus](https://github.com/andalus-ai/andalus)** and [Andalus Framework](https://github.com/andalus-ai/andalus-framework) | Agent runtime/framework; tools, planning and workflow orchestration | Trusted authority for numerical results, tenant permissions or product data policy |
| **[Toba](https://github.com/andalus-ai/toba)** | Data/lakehouse connectivity, source adapters and query integration | A competing statistical-computation engine |
| **[Batik](https://github.com/andalus-ai/batik)** and Tenun | UI/presentation adaptation and chart rendering | Algorithm execution or statistical interpretation authority |

Farabi may expose an algorithm catalog and choose methods for a research question. The actual reusable computation should call a versioned Aljabr API (in-process when runtime compatibility permits, otherwise through an isolated worker/service contract). Farabi remains responsible for authorization, data access, research-plan validation, result provenance and user-facing interpretation.

## 3. Computational capability families

The following families define the intended capability map. Names are conceptual boundaries, not promises that matching Gradle modules already exist.

### 3.1 Mathematical and numerical foundation

- Scalar, vector, matrix and N-dimensional tensor abstractions.
- Linear algebra: decompositions, factorizations, solvers, eigen/singular-value operations and sparse representations.
- Numerical methods: root finding, interpolation, integration, differentiation and approximation.
- Probability and mathematical functions, special functions and stable transforms.
- Numerical precision, error bounds, conditioning, convergence and tolerance policies.
- Random-number generation, deterministic seeds and sampling primitives.
- Units and dimensional-analysis support where appropriate.

### 3.2 Statistical computing

- Descriptive statistics, robust summaries, quantiles and distribution fitting.
- Probability distributions, random variates and resampling.
- Estimation, confidence/credible intervals and hypothesis tests.
- Parametric and non-parametric tests; contingency tables and categorical analysis.
- Correlation, association, effect sizes, power and sample-size calculations.
- Linear, generalized linear and mixed-effects model primitives.
- Bayesian computation and posterior summaries where implemented and validated.
- Bootstrap, permutation tests and multiple-comparison procedures.
- Time-series decomposition, forecasting primitives and backtesting metrics.
- Survival/reliability, survey, psychometric and spatial statistics as separately validated extensions.

Statistical procedures must report assumptions, degrees of freedom, sample sizes, uncertainty and diagnostics as applicable. A p-value must not be presented as effect size or practical importance.

### 3.3 Classical machine learning

- Supervised learning: regression, classification and calibrated prediction.
- Unsupervised learning: clustering, density estimation and anomaly detection.
- Feature transforms, encoding, scaling, selection and pipelines.
- Dimensionality reduction: PCA and other validated methods.
- Model selection, cross-validation, metrics, calibration and threshold analysis.
- Interpretability and feature-attribution methods, with limitations made explicit.
- Data splitting and leakage-aware evaluation utilities.

### 3.4 Deep learning

- Differentiable tensor operations and autograd.
- Neural-network layers, parameter abstractions and reusable model components.
- Loss functions, optimizers, schedules and initialization.
- Training primitives, gradient checking, clipping and checkpoint contracts.
- CPU reference execution and optional accelerator backends.
- Mixed precision and quantization only where numerical and backend behavior is tested.

Aljabr owns reusable low-level training computation. Tafkir owns the higher-level training loop/workflow, datasets, experiment tracking and adaptation policy. Existing autograd and optimizer readiness gaps remain governed by [Research and Training Readiness](RESEARCH_TRAINING_READINESS.md); this scope document does not imply those gaps are closed.

### 3.5 NLP foundations

- Tokenization and tokenizer model contracts.
- Text normalization and vocabulary/encoding utilities.
- N-gram and sparse text representations, TF-IDF and vectorization.
- Embedding and sequence/tensor operations.
- Reusable neural language-model components and model execution primitives.
- Evaluation primitives for text and language-model tasks where definitions are clear.

End-user NLP workflows such as document extraction, summarization, classification pipelines, RAG, agent-driven research and business process automation belong in Andalus applications or consuming products. Aljabr provides reusable computational foundations, not the complete workflow.

### 3.6 Optimization, simulation and scientific computing

- Unconstrained and constrained optimization.
- Numerical optimization algorithms and convergence diagnostics.
- Graph and combinatorial algorithms where they are computationally reusable.
- Monte Carlo simulation and stochastic processes.
- Differential-equation solvers and scientific simulation primitives.
- Signal-processing and domain-specific numerical extensions through optional modules.

These capabilities should be admitted based on a clear reusable API, testable semantics and a real consumer need—not simply because a topic is mathematically related.

## 4. Layering and dependency direction

A target layering model is:

```text
Applications and products
  Farabi | Tafkir | AQLI | Andalus applications
                 |
       Stable Aljabr public APIs
                 |
  Domain computation APIs
  Statistics | Classical ML | Deep Learning | NLP | Scientific
                 |
  Numerical foundation
  Tensor | Linear algebra | Probability | Optimization | RNG
                 |
  Compute SPI and execution
  CPU reference | Vectorization | Optional GPU/native backends
```

Rules:

1. Numerical foundation must not depend on domain algorithms, applications or serving runtimes.
2. Statistics and classical ML should work on the CPU reference path without deep-learning or accelerator dependencies.
3. Optional GPU, native, file-format and external-library integrations must remain isolated behind explicit modules/SPIs.
4. AQLI and Tafkir consume Aljabr APIs; Aljabr never depends on either.
5. Product-specific data permissions, tenant context and governance must not leak into general mathematical APIs.
6. Do not make a monolithic dependency that pulls every domain, native runtime or accelerator into a small consumer.

The exact Gradle module mapping must be designed after inspecting existing module boundaries and dependency graphs. Do not create one module per algorithm mechanically, and do not rename/move public packages without a compatibility and migration plan.

## 5. Canonical algorithms and external libraries

Aljabr should own stable public contracts, capability metadata, numerical semantics and validation for the algorithms it exposes. It does not need to reimplement every mature algorithm if a high-quality library can be integrated safely.

Use an external implementation only behind a narrow adapter when:

- Its license and redistribution terms are acceptable.
- Its supported Java/runtime baseline is compatible.
- Its numerical behavior and edge cases can be tested against reference fixtures.
- Its transitive dependencies and native requirements are understood.
- The public Aljabr API does not expose vendor-specific types unnecessarily.

Candidate libraries are investigation inputs, not approved dependencies: Apache Commons Math or Hipparchus for mathematical/statistical primitives; Smile or Tribuo for selected ML capabilities; and specialized libraries for Bayesian, survival or signal-processing methods. Benchmark and compare before choosing. Avoid duplicate engines and dependency sprawl.

For each exposed algorithm, record:

- Stable algorithm ID and semantic version.
- Purpose, supported input types and data/design constraints.
- Assumptions, parameters, defaults and validation rules.
- Numerical method and implementation/provider version.
- Output schema, uncertainty and diagnostics.
- Determinism/seed behavior and resource characteristics.
- Known limitations, unsupported cases and reference tests.

## 6. Capability maturity and quality gates

Every capability must have an explicit maturity state:

- **Planned:** target only; no implementation claim.
- **Experimental:** implementation exists but API/semantics or validation are incomplete.
- **Supported:** documented public API, focused tests, reference fixtures and defined limitations.
- **Production-validated:** supported plus compatibility, performance, resource-safety and release evidence for named environments.

A class, module, parser or algorithm name is not proof of end-to-end support.

Minimum quality gates:

- Numerical reference tests with documented absolute/relative tolerances.
- Edge cases: empty/small samples, missing/non-finite values, singular matrices, overflow/underflow and degenerate inputs as applicable.
- Deterministic tests with controlled seeds where randomness is involved.
- Gradient checks for differentiable operators.
- Statistical validation against published/reference fixtures and independent implementations where practical.
- Clear unsupported-capability errors rather than silent fallback or fabricated output.
- Resource ownership, cancellation and bounded-memory behavior for large workloads.
- Compatibility tests for the documented JDK and published consumer artifacts.

## 7. Reproducibility and execution safety

A computation result should be reproducible from, or traceable to:

- Input dataset identity/hash and schema/version (owned by the consuming product).
- Algorithm ID/version and provider/library version.
- Parameters, preprocessing and feature transformation definitions.
- Random seed and numerical precision/backend.
- Runtime/JDK and relevant environment metadata.
- Execution timestamp, status, diagnostics and output references.

Aljabr should accept typed data and computation requests, not arbitrary agent-generated source code. Workload isolation, tenant authorization, data retention, query limits and publication policy remain responsibilities of the calling product/runtime. Expensive or untrusted workloads should execute in bounded workers with explicit time, memory and concurrency limits.

## 8. Farabi integration contract

The intended division for Farabi Statistical Intelligence is:

1. Farabi imports and profiles datasets, enforces tenant/workspace and row/column permissions, and captures data lineage.
2. Farabi's research planner translates the user's question into a typed, declarative analysis plan.
3. Farabi validates method suitability, assumptions, sample size, leakage, missingness, temporal ordering and multiplicity before execution.
4. A Farabi-owned adapter invokes the selected Aljabr algorithm through a versioned API or isolated worker.
5. Aljabr returns typed estimates, diagnostics, uncertainty and execution metadata—not a product-specific report.
6. Farabi checks provenance and policy, interprets the findings, and stores research artifacts.
7. Farabi emits renderer-neutral chart/result models for Batik, Tenun and other clients.

No agent or Aljabr method may bypass Farabi's authorization and governance checks. No Aljabr computation may directly write to Farabi's database.

### Runtime compatibility

The current documented Aljabr baseline is JDK 25, while Farabi's existing architecture baseline has been JDK 21. Until Farabi's runtime baseline is deliberately upgraded and validated, do not assume direct in-process dependency compatibility. Prefer an isolated worker or versioned service boundary as an interim integration. Revisit this decision with compatibility tests before publishing a direct dependency.

## 9. Delivery sequence

| Stage | Outcome | Exit evidence |
|---|---|---|
| 0. Inventory | Map current modules, public APIs, dependencies, licenses and maturity | Reviewed module/capability matrix; no assumed module names |
| 1. Numerical foundation | Stabilize tensor, linear algebra, RNG and numerical semantics | CPU reference tests, API docs and resource-lifecycle tests |
| 2. Statistical core | Descriptive stats, distributions, inference, tests, regression and resampling | Reference fixtures, assumptions/diagnostics and reproducible examples |
| 3. Classical ML | Core supervised/unsupervised algorithms and evaluation | Leakage-aware examples, metrics and deterministic fixtures |
| 4. Deep-learning readiness | Repair and validate autograd, NN and optimizer contracts | Tiny deterministic training example converges in CI |
| 5. NLP foundations | Tokenization, vectorization, embeddings and sequence primitives | Compatibility fixtures and focused API tests |
| 6. Advanced scientific domains | Bayesian, time-series, survival, simulation and specialized methods | Separate acceptance criteria per domain; optional dependencies |
| 7. Ecosystem integration | Stable consumer artifacts and Farabi/AQLI/Tafkir adapters | Clean published-artifact resolution and cross-repository contract tests |

Stages may overlap where dependencies permit, but deep-learning training claims remain blocked by the readiness audit until its acceptance criteria pass.

## 10. Decisions required before implementation

- Confirm the supported JDK policy for downstream consumers and whether Farabi will move from JDK 21 to 25.
- Complete a source-level inventory of existing tensor, linear algebra, statistics and ML modules before proposing Gradle moves.
- Select library adapters only after license, API, performance and numerical validation.
- Decide the public algorithm registry/capability-discovery contract.
- Define the stable worker/service protocol if cross-JDK integration is required.
- Add ADRs for material API, dependency and module-boundary decisions.

## 11. Related documents

- [Framework Architecture](FRAMEWORK_ARCHITECTURE.md)
- [Research and Training Readiness](RESEARCH_TRAINING_READINESS.md)
- [Java Tensor Research Quickstart](../examples/TENSOR_RESEARCH_QUICKSTART.md)
- [Farabi Statistical Intelligence](https://github.com/andalus-ai/farabi/blob/main/docs/STATISTICAL_INTELLIGENCE.md)
- [AQLI System Architecture](https://github.com/andalus-ai/aqli/blob/main/docs/architecture/SYSTEM_ARCHITECTURE.md)
- [Tafkir](https://github.com/andalus-ai/tafkir)
- [Andalus Framework](https://github.com/andalus-ai/andalus-framework)
