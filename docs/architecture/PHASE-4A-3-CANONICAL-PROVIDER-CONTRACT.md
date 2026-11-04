# Phase 4A.3 — Canonical Provider, Device, and Session Contract

**Status:** Proposed contract baseline for implementation review  
**Date:** 2026-11-04  
**Depends on:** ADR-0001, Phase 4A.1 audit, Phase 4A.2 contract comparison

## 1. Decision summary

This document resolves the Phase 4A.2 open questions and defines the smallest compatible direction for the next implementation steps.

1. **Keep `AljabrProvider` as the provider lifecycle SPI.** Do not create a second provider registry or another general-purpose compute-provider interface.
2. **Keep `core.backend.ComputeBackend` as the canonical tensor-operation API.** It remains an execution facet, not a provider identity, device registry, or runtime lifecycle.
3. **A provider may expose multiple device sessions and multiple execution facets.** A session is scoped to one selected device (or an explicitly device-less host runtime) and owns the resources used by its facets.
4. **Introduce provider-owned device discovery and session composition additively.** The proposed API belongs in the neutral provider SPI module and may refer to the tensor API only at the composition boundary. Vendor SDK types must not cross the SPI.
5. **Treat `KernelPlugin` as a legacy/specialized runner extension point, not as a universal provider SPI.** No new provider implementation should depend on it. Preserve existing code until concrete callers, modules, and tests are migrated; a later deprecation/removal decision requires a separate audit.
6. **Represent operation support explicitly.** Distinguish native, emulated/delegated, unsupported, and runtime-failed outcomes. A backend's existence does not imply that every operation is native or supported.
7. **Do not impose a GPU stream model on all providers.** The initial common tensor contract remains synchronous. Asynchronous queues/streams are optional, capability-specific extensions.
8. **No provider-level CPU fallback by default.** The consumer explicitly authorizes fallback. Every delegated or fallback execution must be diagnosable.

This is a design baseline, not authorization to change existing backend behavior or public signatures in one migration.

## 2. Provider and runtime lifecycle

The lifecycle is provider-scoped, not process-global:

```text
Consumer runtime (AQLI, Tafkir, research application, etc.)
  | chooses provider and device according to its own policy
  v
Aljabr provider discovery (ServiceLoader / explicit registration)
  | creates an isolated provider instance
  v
AljabrProvider
  | initialize(provider-scoped context)
  | enumerate devices and report provider facts
  | open a selected device
  v
ProviderDeviceSession (provider-owned, closeable)
  | exposes one or more execution facets
  +--> core.backend.ComputeBackend (tensor operations)
  +--> optional specialized capabilities (attention, transfer, collectives, ...)
```

The diagram describes the target composition. The current repository does not yet implement this complete path.

### 2.1 Discovery and registration

- The canonical service type remains `tech.kayys.aljabr.spi.provider.AljabrProvider`.
- A provider artifact registers its implementation using `META-INF/services/tech.kayys.aljabr.spi.provider.AljabrProvider`.
- Discovery must be scoped to an explicit `ClassLoader` or caller-provided provider source. Avoid hidden thread-context-classloader assumptions as the only option.
- Discovery creates provider instances but must not initialize native runtimes or select devices.
- Discovery failures are reported per provider; one broken optional provider must not prevent CPU/core discovery.
- Duplicate provider IDs are rejected or reported as an explicit discovery conflict; iteration order must not silently choose a winner.
- The initial implementation should provide an explicit registry/discovery object owned by an Aljabr runtime. It must not introduce a mutable static singleton.

### 2.2 Provider lifecycle

Retain the existing `AljabrProvider` identity, descriptor, initialize, and close semantics. Evolve additively toward:

- Provider descriptor is readable before initialization and contains stable provider ID, provider version, compatible Aljabr SPI/API range, and declared capability IDs.
- Initialization accepts immutable provider-scoped technical configuration and returns or makes available an initialized provider runtime handle.
- Initialization is idempotent only if the implementation explicitly documents that behavior; otherwise a second initialize call returns a typed lifecycle error.
- Closing a provider after partial initialization is safe. Closing an already-closed provider is safe.
- Provider instances are not shared implicitly across independent Aljabr runtime instances.
- Initialization, discovery, device-open, and close failures use Aljabr-owned typed exceptions with provider/device context and preserved causes.

The current `ProviderContext` string map is adequate for the first compatibility step. Do not replace it with a large configuration framework. Typed configuration can be added only where a concrete provider requires it.

## 3. Device model

The neutral device descriptor should contain only portable facts:

| Field | Required | Semantics |
|---|---:|---|
| `providerId` | Yes | Stable ID of the provider that owns this device |
| `deviceId` | Yes | Stable within that provider/runtime; opaque to consumers |
| `name` | Yes | Human-readable display name |
| `kind` | Yes | `CPU`, `GPU`, `ACCELERATOR`, or `OTHER` |
| `memory` | No | Total/free/available bytes when the provider can report them; unknown is not zero |
| `driverVersion` | No | Host driver version, if meaningful |
| `runtimeVersion` | No | Vendor/runtime version, if meaningful |
| `attributes` | No | Immutable provider-neutral string metadata for additional facts |

Rules:

- Device IDs are opaque; consumers must not parse vendor-specific formats.
- Device enumeration is not initialization. Detection does not prove that the device can be opened or execute a particular operation.
- CPU is a valid device kind. A provider may expose a host execution device without a physical accelerator.
- Vendor-specific properties remain available through namespaced attributes or provider diagnostics, not vendor classes in the SPI.
- Memory values must represent unknown explicitly; never report unknown capacity as zero.
- Device descriptors are snapshots. Dynamic memory/readiness queries belong to a session or diagnostics view.

## 4. Readiness and operation capability semantics

Do not compress provider state into a single `isAvailable()` boolean. Report independent facts/stages:

1. **DECLARED** — provider/capability is described by metadata.
2. **DISCOVERED** — provider implementation was found by the registry.
3. **INSTALLED** — provider artifact is present in the deployment (inferred from discovery, not hardware readiness).
4. **DEVICE_DETECTED** — provider enumerated the device.
5. **DEVICE_OPENABLE** — provider reports prerequisites satisfied sufficiently to attempt opening.
6. **INITIALIZED** — provider runtime/session opened successfully.
7. **VALIDATED** — a named conformance suite passed for a specific provider/runtime/device/software combination.

These are not one linear global enum: for example, one device may be initialized while another is only detected. Preserve provider/device/capability scope and validation provenance.

For each operation (or operation family), report one of:

- **NATIVE** — executed by the selected provider/device implementation.
- **EMULATED** — executed through a declared alternate implementation that is not native to the selected device.
- **DELEGATED** — intentionally executed by another backend/device (including CPU), with the actual executor identified.
- **UNSUPPORTED** — no valid implementation is available for the requested constraints.
- **UNKNOWN** — the provider cannot make a reliable declaration before execution.
- **FAILED** — execution was attempted and failed; include a typed reason. This is an execution outcome, not a static capability.

A capability declaration is not proof of correctness. Validation records must identify suite/version, provider version, device/runtime facts, and test result. Do not claim a provider is generally validated based on one smoke test.

## 5. Session and execution-facet composition

A provider runtime can enumerate devices and open a device session. The session:

- Is explicitly owned and closeable.
- Is bound to one device or a documented host context.
- Owns provider resources, command queues, allocators, and native handles needed by its execution facets.
- Exposes the canonical tensor `ComputeBackend` when tensor execution is supported.
- May expose optional, typed capability facets for specialized attention, collectives, memory transfer, graph execution, or other non-tensor services.
- Reports the actual device and implementation used by each facet.
- Does not select a different provider or silently substitute CPU after an execution failure.

Multiple sessions may coexist when the provider supports it. A provider may expose more than one tensor backend facet (for example, distinct execution modes), but each facet must state its device and operation support. Consumers own cross-provider scheduling and model-level selection.

Do not put provider ID, device enumeration, initialization, close, or ServiceLoader behavior onto `core.backend.ComputeBackend`. Keep the tensor operation interface focused on operations.

## 6. KernelPlugin disposition

The current `KernelPlugin` API has generic operation execution and string-based platform/operation metadata, while concrete classes carry additional lifecycle behavior. Its source/module placement and root-build participation are inconsistent.

Decision:

- It is **not** the provider registry, provider lifecycle, or canonical tensor backend.
- It may remain as a specialized runner/kernel extension point where existing code still uses it.
- New provider/device/session work must not depend on `KernelPlugin`.
- Do not move it into a neutral SPI module or promise public compatibility until its real callers, build inclusion, lifecycle methods, and operation types have been audited.
- Any eventual migration must map a demonstrated specialized use case to an optional session capability facet; it must not create a second device discovery or provider lifecycle system.
- Preserve existing classes and fallback behavior until a separate migration PR proves parity.

## 7. Asynchronous execution and streams

The first provider/session contract must not require streams, events, command buffers, or GPU queue types.

- Existing `ComputeBackend` methods retain their current synchronous completion expectations during this phase.
- Providers may internally use queues/streams and synchronize before returning when required by the existing contract.
- Future asynchronous execution should be an optional capability/facet with explicit completion, cancellation, error propagation, and resource-lifetime semantics.
- A CPU-only provider must not implement meaningless GPU abstractions to satisfy the common SPI.
- AQLI may build scheduling/streaming policy above these Aljabr mechanisms, but Aljabr should expose reusable low-level primitives where multiple consumers need them.

## 8. Fallback and diagnostics

Fallback policy belongs to the consumer, not to a provider's hidden default.

- Provider initialization failure is surfaced; the consumer may explicitly choose another provider/device.
- Unsupported operation is surfaced or resolved through a declared delegated facet according to consumer policy.
- Runtime execution failure is surfaced with the failed provider, device, operation, and cause.
- If fallback is explicitly enabled, the result/diagnostic must identify the actual executor and reason. Never report CPU-delegated work as native GPU execution.
- A provider must not silently mutate a selected session into a CPU session.
- Existing CUDA/Metal per-operation CPU delegation is preserved during the contract-design phase. It must be classified and made observable before changing defaults or numerical execution.

## 9. Packaging and dependency boundary

- Core and provider SPI modules have no mandatory vendor SDK/native dependencies.
- CPU, Metal, CUDA, ROCm, Vulkan, OpenCL, DirectML, and future providers are independently selectable artifacts.
- A provider's service descriptor and implementation are packaged only with that provider artifact.
- Native libraries are loaded only by the selected provider, not by core discovery.
- CPU-only dependency graphs must not include Metal, CUDA, Objective-C, or vendor native runtime artifacts.
- Apple Silicon, NVIDIA, and CPU-only packaging profiles are separate from runtime device selection.
- Experimental/unincluded modules are not advertised as supported providers merely because source files exist.

## 10. API design boundary (conceptual, not final Java signatures)

The implementation should add only these concepts to the existing provider SPI:

- A provider runtime handle created by provider initialization.
- A device descriptor and provider-scoped device enumeration result.
- A device session opened for an explicit device selection.
- A session capability/facet lookup or typed accessor for the tensor backend and optional services.
- Structured provider/device/session errors and diagnostics.

Do not add a second provider descriptor, registry, global backend manager, general plugin manager, or duplicate tensor operation interface. Exact Java signatures should be reviewed against the existing source/build dependency graph before implementation. Keep `aljabr-spi-core` independent of `aljabr-tensor` if possible; if the composition API needs the tensor type, put the narrow adapter contract in a separate module that depends inward on both SPIs rather than making the foundational provider SPI depend on tensor implementation details.

## 11. Conformance and validation plan

The existing `aljabr-spi-testkit` currently validates provider descriptor and basic initialize/close lifecycle only. It does not validate provider discovery, devices, operation support, or numerical behavior. Expand it only as the corresponding contracts are implemented.

Required staged gates:

1. **SPI unit tests:** immutable descriptors, duplicate IDs, lifecycle transitions, typed failures, partial-init close, idempotent close semantics.
2. **Discovery tests:** valid service descriptor, missing provider, provider constructor failure, duplicate IDs, isolated classloaders, one broken optional provider not blocking others.
3. **Fake-device tests:** zero devices, multiple devices, unknown memory, device open failure, independent sessions, concurrent provider instances.
4. **Operation capability tests:** native/emulated/delegated/unsupported/unknown reporting; execution failure diagnostics; explicit fallback only.
5. **Backend conformance:** shape, dtype, numerical tolerance, mutation guarantees, device ownership, resource lifetime, and unsupported operation behavior.
6. **Dependency graph tests:** core/CPU consumer has no accelerator SDK/native artifacts; each accelerator profile includes only its selected provider dependencies.
7. **Hardware validation:** Metal on Apple Silicon; CUDA/ROCm on their supported hosts. Hardware tests are separate from fake-provider contract tests.

## 12. Implementation sequence

| Step | Scope | Exit condition |
|---|---|---|
| 4A.3a | Reconcile stale `spi.tensor.ComputeBackend` references and service descriptors in excluded CUDA/ROCm plugin modules | Exact type/signature audit and targeted compile probe; no accidental API substitution |
| 4A.3b | Add provider discovery registry around existing `AljabrProvider` | Isolated classloader, duplicate ID, and provider-failure tests pass |
| 4A.3c | Add device descriptor/enumeration and provider runtime/session contracts | Fake provider/device lifecycle tests pass |
| 4A.3d | Add tensor-backend composition adapter in a module that preserves SPI dependency direction | CPU provider exposes `core.backend.ComputeBackend` without global state |
| 4A.3e | Add explicit operation support and execution diagnostics | Delegation/fallback is observable and test-covered |
| 4A.3f | Adapt Metal provider and validate on Apple Silicon | Existing Gemma/GGUF Metal behavior and benchmarks remain correct |
| 4A.3g | Adapt CUDA/ROCm only after optional modules compile and provider runtime paths are real | Provider-specific CI and hardware validation pass |

Each step is a separate, reviewable PR. No broad backend rewrite, fallback-default change, Metal kernel change, or AQLI policy implementation is part of this proposal.

## 13. Non-goals

- No production source/API changes in this proposal.
- No immediate removal or rewrite of `KernelPlugin`.
- No change to existing CUDA/Metal fallback behavior.
- No assumption that excluded provider modules are currently supported.
- No new global backend-selection singleton.
- No AQLI dependency in Aljabr.
- No hardware/performance claim.
