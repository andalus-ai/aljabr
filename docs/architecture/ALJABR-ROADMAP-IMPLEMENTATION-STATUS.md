# Aljabr roadmap implementation status and next-step audit

**Audit date:** 2026-11-05  
**Scope:** Evidence-based checkpoint against the ten-phase Aljabr roadmap. This is a focused audit of the provider/compute architecture and the recently merged AQLI boundary work, not yet an exhaustive inventory of every module, public API, dependency, test, or downstream consumer.

## Executive summary

Aljabr has already moved beyond the original minimal provider-SPI starting point. The foundational provider contract, a compute-provider extension, provider sessions, a reusable lifecycle test kit, a pure-Java CPU compute provider, and a Metal compute-provider adapter are present in the current source tree. Recent work also clarifies that Aljabr owns reusable computation/model capabilities while AQLI owns inference serving and runtime orchestration.

The next step is **not** to add another provider or inference-engine abstraction. It is to complete the general capability-management and computational-semantics architecture around the existing contracts, while auditing current numerical/tensor APIs before proposing replacements.

### Verified current state

| Roadmap area | Evidence in current repository | Assessment |
|---|---|---|
| Phase 0 — repository audit | Root Gradle settings enumerate core, SPI, tensor, CPU, CUDA, Metal, GGUF, Safetensors, and model projects; recent architecture docs record selected inference ownership and consumer findings | **In progress.** Not an exhaustive module/API/dependency/test inventory |
| Phase 1 — base provider SPI | `core:aljabr-spi-core` contains `AljabrProvider`, descriptors, context, and provider exception; lifecycle semantics were strengthened in PR #295 | **Base contract delivered.** Needs a formal compatibility/versioning policy and external-provider compatibility example |
| Phase 2 — registry and conformance | `core:aljabr-spi-testkit` provides basic lifecycle conformance; `ProviderSession.operationSupport` provides conservative operation readiness reporting; CPU tests exercise known support | **Partially delivered.** A general capability registry, registration lifecycle, compatibility evaluation, and richer conformance/reporting model are not evidenced by the audited paths |
| Phase 3 — computational semantics | Existing tensor and compute APIs are present and are dependencies of compute SPI/backend modules | **Audit required.** Do not assume the current public semantics are complete or replace them before mapping actual contracts |
| Phase 4 — reference providers | `backend:cpu:aljabr-backend-cpu` implements `ComputeProvider`; CPU sessions expose the existing compute backend. Metal has a separate provider adapter | **Partially delivered.** CPU compute provider is not, by itself, proof of a complete pure-Java numerical reference provider across all target capabilities |
| Phase 5 — accelerated providers | Metal adapter is present; CUDA backend/kernel projects are included in Gradle settings; optional HAT is guarded behind an explicit JDK property | **In progress.** Per-capability conformance and benchmark evidence must gate readiness claims |
| AQLI boundary | ADR-0002 and the AQLI inference SPI consolidation inventory establish ownership and migration cautions | **Documented for the inspected inference surface.** The AQLI consumer inventory explicitly remains incomplete |

## Current architecture observations

### 1. Provider contracts are layered already

The repository has a useful separation:

- `core:aljabr-spi-core`: provider identity, metadata, compatibility declaration, initialization context, lifecycle, and provider failures.
- `core:aljabr-spi-compute`: optional compute-facing extension of `AljabrProvider`, exposing device inventory and device-scoped `ProviderSession` instances.
- `core:aljabr-spi-testkit`: reusable base provider lifecycle checks.
- `backend:cpu:aljabr-backend-cpu`: pure-Java CPU implementation of the compute-provider contract.
- `backend:metal:aljabr-backend-metal`: optional Metal compute-provider adapter.
- Tensor/backend APIs remain existing Aljabr contracts consumed by the compute SPI and providers.

Keep this architecture. Do not introduce a second generic `EngineProvider`, `ComputeRuntimeProvider`, or inference-engine SPI to duplicate these responsibilities.

### 2. The existing capability metadata is not yet a registry

Providers currently declare capability descriptors, for example `tensor.compute`, with a provider-supplied stability value. The code and documentation correctly distinguish a declaration from proof of correctness or availability.

The missing general capability-management concerns to design and verify are:

- Stable capability identity and capability contract versioning.
- Registration and deregistration semantics, including duplicate provider/capability handling.
- Provider/API compatibility evaluation.
- Runtime availability and device/resource readiness.
- Validation status derived from conformance results, not provider claims.
- Capability lookup by operation and constraints.
- Diagnostics explaining why a provider/capability is unavailable, incompatible, unsupported, or unvalidated.
- Registry lifecycle, thread safety, and deterministic behavior.
- External provider registration without requiring Aljabr implementation internals.

Before adding a registry module, inspect all existing discovery/registration code in core, compute, model, tokenizer, plugin, GGUF, and Safetensors areas. The registry must compose existing extension points rather than replace unrelated format-specific registries.

### 3. Operation support is a useful low-level signal, not full capability conformance

The current `ProviderSession.operationSupport(operationId, constraints)` defaults conservatively to `UNKNOWN`. The CPU implementation reports `NATIVE` only for its currently verified unconstrained `tensor.add` and `tensor.matmul` operations, reports `UNSUPPORTED` for `tensor.rope`, and remains conservative for constrained requests.

Preserve this behavior. Do not infer support from the existence of a Java method or from a provider's broad `tensor.compute` declaration. Over time, define the relationship between:
- provider-level declared capabilities;
- operation-level support/readiness;
- capability-specific conformance results;
- Aljabr maturity/stability;
- benchmarked optimization status.

These are related dimensions and must not be collapsed into one enum or boolean.

### 4. Compute-provider lifecycle has meaningful guarantees

The current provider/session contracts define single-use initialization, terminal/idempotent close, cleanup after partial initialization, provider shutdown of outstanding sessions, and failure when a closed session is used. CPU tests exercise important lifecycle and discovery cases.

Keep these lifecycle rules consistent across all providers. In particular, verify the Metal session ownership limitation documented in its adapter work: native binding resources are process-scoped and the current binding does not provide per-session native release. Do not claim per-session native cleanup where the implementation cannot provide it.

### 5. AQLI boundary work must remain separate from the general capability registry

The merged ADR and inventory clarify:
- Aljabr owns reusable tensors, model/format APIs, reusable model execution, compute providers, and low-level execution capabilities.
- AQLI owns runtime provider-selection policy, request admission/routing/scheduling, serving sessions, streaming delivery, deployment, and operational policy.
- GGUF and Safetensors remain separate format adapters.
- Existing AQLI public/runtime contracts must not be deleted mechanically; consumer and compatibility audits are required.

The general Aljabr provider/capability registry must not absorb AQLI serving policy. AQLI may consume Aljabr capability information and make runtime decisions in its own layer.

## Recommended next work

### Workstream A — finish the Phase 0 source-of-truth inventory

Create a complete module and dependency map from the actual Gradle build:
1. Enumerate every included and conditionally included Gradle project.
2. Classify modules as public API, SPI, implementation, provider/backend, adapter/format, model family, tooling, or test infrastructure.
3. Record module-to-module dependencies and identify cycles, accidental API leaks, and optional modules pulled into foundational builds.
4. Inventory public packages/types and identify overlapping tensor, compute, model, inference, tokenizer, plugin, and format contracts.
5. Map tests to modules and identify untested public SPI behavior.
6. Record JDK/toolchain requirements and optional preview/incubator/native build conditions.
7. Verify which modules are publishable and whether their public dependency metadata is sufficient for external consumers.

Deliverables: module catalog, dependency graph, API ownership map, test coverage map, and prioritized cleanup backlog.

### Workstream B — Phase 2 capability registry design

Design against the existing `AljabrProvider` and `ComputeProvider` contracts. The design must answer the registration lifecycle, identity/version model, compatibility checks, state model, query semantics, diagnostics, concurrency, and external extension use case before implementation.

Do not assume every Aljabr capability is a compute device capability. The registry should be general enough for numerical, statistical, model, tokenizer, and other extension domains without forcing those domains to depend on the tensor/compute SPI.

Potential capability states should remain distinct: declared, installed, available, compatible, validated, stable, and optimized. Define which states are runtime observations, which are validation records, and which are Aljabr governance/maturity decisions.

### Workstream C — Phase 3 computational-semantics audit

Audit the existing public tensor/numerical APIs before designing new ones. Record actual behavior and gaps for:
- dtype inventory and promotion;
- shape, layout, broadcasting, indexing, and slicing;
- numeric overflow, NaN/infinity, and precision guarantees;
- deterministic execution and random-number generation;
- memory ownership, views, copying, and device transfer;
- concurrency, cancellation, and resource lifetimes;
- operation error behavior and unsupported-operation reporting;
- provider-independent semantics and provider-specific constraints.

Deliverable: a semantics matrix mapping each behavior to existing API, implementation, tests, unresolved decision, and compatibility impact. Prefer documenting and tightening existing APIs over introducing parallel array/tensor models.

### Workstream D — provider conformance expansion

Keep the existing base lifecycle test kit. Add capability-specific conformance suites only when a capability contract is clear. For compute, test descriptor/device consistency, session lifecycle, operation-support conservatism, backend map immutability, and numerical correctness against independent references. Keep correctness conformance separate from performance benchmarks.

### Workstream E — reference and accelerator readiness

Treat the CPU provider as the first portable compute provider, then assess whether it can serve as the reference implementation for the relevant tensor operations. Do not call the entire CPU backend a reference provider until coverage and independent correctness tests justify that designation.

Keep Metal/CUDA/other accelerators optional. A provider must pass relevant correctness conformance before benchmark results or automatic selection claims are accepted.

## Parallel execution plan from this checkpoint

| Workstream | Can start now? | Dependency / merge gate |
|---|---|---|
| A. Complete Phase 0 inventory | Yes | None; inspect current main and actual Gradle projects |
| B. Capability registry design | Yes | Must reuse existing provider/compute contracts; no implementation until contract review |
| C. Computational-semantics audit | Yes | Must inspect existing APIs and tests; no competing tensor model |
| D. Conformance expansion design | Yes | Reuse current test kit; capability tests depend on explicit capability semantics |
| E. CPU/reference assessment | Yes | Use existing CPU backend; reference designation depends on independent correctness evidence |
| Registry implementation | After design review | Stable identity, lifecycle, state, compatibility, and query contracts |
| New numerical public APIs | Only for demonstrated gaps | Semantics matrix and compatibility review first |

## Immediate acceptance criteria

The next checkpoint is complete when:

1. The Gradle module catalog and dependency graph are evidence-based and exhaustive for the current build configuration.
2. Existing provider, compute, and format-specific discovery/registration mechanisms are documented and their ownership is clear.
3. The capability registry proposal reuses the existing provider contracts and does not duplicate AQLI runtime selection.
4. The computational-semantics matrix maps current public APIs and tests rather than proposing a replacement abstraction prematurely.
5. The CPU provider's verified operation support and conformance scope are explicit.
6. The next implementation PR is narrowly scoped, has upstream dependencies stated, and adds focused tests.

## Validation and scope disclaimer

This checkpoint was assembled from the current root Gradle settings, the provider SPI and compute SPI source files, the provider test kit, the CPU compute provider, and the merged provider lifecycle/Metal/operation-support and AQLI ownership changes. It is a focused architectural checkpoint, not a claim that every source file, Gradle dependency, public API, test, or external consumer has been exhaustively inspected.

No build or tests were run as part of this documentation audit.
