# Aljabr SDK Capability and Developer-Experience Gap Audit

**Status:** Source-tree audit and prioritized follow-up plan  
**Baseline:** `main` after PR #710  
**Scope:** Public API, module boundaries, tensor/memory contracts, diagnostics, model/diffusion extensibility, developer workflows and distribution.

> This audit distinguishes source-tree evidence from claims of stable or published support. Module presence alone does not establish completeness or maturity.

## 1. Executive summary

Aljabr already has substantial foundations: a tensor API with eager/lazy execution, explicit storage ownership and immutable `TensorLayout`; FFM-backed/reference-counted storage and views; a structured diagnostics module; numerics and linear algebra modules; many model-family modules; provider registration; and a dedicated diffusion subsystem. The problem is therefore primarily cohesion, contract consistency and developer/distribution experience—not a lack of components.

## 2. Capability inventory

| Capability | Evidence | Assessment |
|---|---|---|
| Tensor | `plugins/ml/aljabr-tensor`, `Tensor`, `DefaultTensor`, `LazyTensor` | Implemented foundation; broad facade still exposes backend |
| Layout | `TensorLayout`, explicit-layout constructors | Present; cross-backend stride semantics need conformance |
| Memory | `TensorStorage`, FFM arena storage, storage views, ownership docs | Strong partial foundation; duplicate class ownership below |
| Diagnostics | `core/aljabr-diagnostics`: diagnostics, sinks, scopes, execution/provider observers | Foundational module exists; end-to-end usability needs review |
| Numerics | `plugins/ml/aljabr-numerics` | Exists, but grouped under ML namespace |
| Linear algebra | `plugins/ml/aljabr-linear-algebra` | Exists, but should be independently scientific |
| Statistics | No dedicated module identified in scanned build inventory | Gap or undiscovered implementation |
| Datasets | Clustering-specific dataset and legacy autograd dataset paths; no canonical cross-domain API identified | Fragmented / gap |
| Autograd/optimizers/NN | Dedicated modules under `plugins/ml` | Foundation exists; breadth and integration need audit |
| Model extensions | Many `aljabr-model-*` modules and service registrations | Large surface; external author journey/conformance needs work |
| Diffusion | Dedicated API/core/scheduler/execution modules and legacy model-common classes | First-class direction exists; consolidation remains |
| JShell/JBang/Jupyter | Examples/docs exist; no obvious maintained integration workflow identified | Developer workflow gap |
| BOM/profiles | No obvious BOM/profile artifact identified in build inventory | Distribution gap |
| Backends | ComputeBackend SPI and multiple provider modules | Exists; backend details leak through Tensor |

## 3. Priority findings

### P0 — Duplicate ManagedArena class across artifacts

Two source roots declare the same fully qualified public type:

- `plugins/ml/aljabr-tensor/src/main/java/tech/kayys/aljabr/core/memory/ManagedArena.java`
- `core/aljabr-core/src/main/java/tech/kayys/aljabr/core/memory/ManagedArena.java`

Both declare `tech.kayys.aljabr.core.memory.ManagedArena`. The implementations differ: tensor offers shared/confined factories and `arena()`; core offers `create()/wrap()` and checks wrapped arena liveness in `isOpen()`. Both already use CAS-based overflow and over-release protection, so older audit claims that over-release is unguarded are stale.

Further, `:core:aljabr-core` has an API dependency on `:plugins:ml:aljabr-tensor`. Publishing both artifacts can expose duplicate class definitions and classpath-order ambiguity.

**Next:** map references and published consumers, choose one canonical owner, retain compatibility through adapters/deprecation if needed, remove the duplicate FQCN, and add a clean consumer fixture proving one definition is present.

### P0 — Core/tensor dependency direction

`:core:aljabr-core` currently declares `api(project(":plugins:ml:aljabr-tensor"))`. This makes nominal core transitively expose the ML tensor artifact and conflicts with the goal that statistics/scientific-only consumers avoid tensor and NN dependencies.

**Next:** inventory public types and downstream consumers. Move only truly foundational contracts downward when justified; do not mechanically relocate packages or create unnecessary modules. Test the published dependency graph.

### P1 — Scientific capability taxonomy

Numerics and linear algebra live under `plugins/ml`, obscuring their use beyond ML.

**Next:** define stable artifact taxonomy and dependency graph first. Relocate only with compatibility mapping and migration strategy.

### P1 — Tensor contract conformance

Tensor already documents borrowed storage, ownership transfer, lazy materialization and layout metadata. Remaining cross-backend concerns include aliasing, eager/lazy parity, dtype semantics, materialization and capability discovery.

**Next:** reusable contract suites for ownership/borrowing/transfer, close and failure cleanup, view aliasing and parent lifetime, contiguous/strided/overlapping/empty layouts, eager/lazy parity, device transfer, dtype promotion and quantized storage. Do not add abstractions before these semantics are tested.

### P1 — Diffusion consolidation

Dedicated modules exist (`aljabr-diffusion-api/core`, scheduler API, DDIM/Euler and execution API/Euler), while `models/aljabr-model-common` still contains `StableDiffusionPipeline`, `DDIMScheduler` and related classes.

**Next:** map callers, tests and artifacts; migrate family-neutral behavior to the dedicated subsystem; retain family-specific topology in family modules; deprecate legacy duplicates only after numerical parity tests. Expand by algorithm/model compatibility, including discrete diffusion vs flow matching and prediction semantics.

### P1 — External model/algorithm extension experience

Many model modules and service registrations exist, but the inventory does not prove a complete external extension workflow.

**Next:** provide a clean external Maven/Gradle provider example, conformance test kit, capability descriptors, lifecycle rules, registration examples and actionable diagnostics using existing SPI.

### P2 — Statistics and dataset foundations

A generic dataset/source/transform API and dedicated statistics module were not identified.

**Next:** perform a focused algorithm/API inventory. Define general source/schema/transform/streaming/batching/lifecycle contracts separately from Tafkir's training orchestration. Keep statistics usable without tensor/NN dependencies where appropriate.

### P2 — Developer workflows

No maintained JShell/JBang/Jupyter workflow was obvious in the repository inventory.

**Next:** tested examples using published coordinates: JShell snippets, JBang scripts, Jupyter via an established Java kernel such as IJava, and Gradle/Maven promotion paths. All use the same API and lifecycle rules.

### P2 — BOM and consumer profiles

No obvious Aljabr BOM was identified.

**Next:** external consumer fixtures for numerics/scientific, tensor/CPU, ML, multimodal/diffusion and accelerator combinations. A BOM aligns selected versions; it does not remove dependencies. Add starter bundles only where they materially reduce setup friction. Verify dependency metadata before jlink/Native Image/shrinking work.

## 4. Recommended implementation order

1. Resolve duplicate `ManagedArena` ownership and core-to-tensor API dependency.
2. Build tensor ownership/layout/eager-lazy/provider conformance tests.
3. Improve public API cohesion and reduce backend leakage only where evidence supports it.
4. Define scientific capability and dependency taxonomy.
5. Consolidate diffusion with numerical parity.
6. Deliver external provider/model extension example and conformance kit.
7. Deliver JShell, JBang and Jupyter workflows.
8. Deliver BOM, consumer profiles and minimal packaging fixtures.

Each increment should be a small PR. Avoid combining package relocation, public API redesign and performance changes.

## 5. Guardrails

- Preserve JDK 25 baseline unless deliberately changed.
- Keep experimental HAT and optional accelerator dependencies opt-in.
- Do not introduce mandatory Python/native inference dependencies.
- Keep AQLI serving policy and Tafkir training workflow orchestration out of Aljabr.
- Keep tensors first-class without making them mandatory for every scientific consumer.
- Prefer generated metadata over reflection-heavy hot paths.
- Make copies, transfers, aliasing and resource lifetimes observable.
- Test published artifacts from external consumer projects, not only composite builds.
- Distinguish planned, implemented, tested, validated, stable and optimized capabilities.

## 6. Audit limitations

This is a repository-tree and targeted source inspection, not a semantic review of all 328 Gradle build files, every public class, downstream consumer or published artifact. "Not identified" is not proof of absence. Begin implementation only after verifying the P0 findings with targeted tests and a clean consumer fixture.
