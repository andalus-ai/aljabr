# Phase 4A.2 — Backend Contract Comparison and Consolidation Proposal

**Status:** Proposal for review; no API migration authorized  
**Date:** 2026-11-04  
**Decision context:** ADR-0001 — Backend Ownership  
**Audit baseline:** Phase 4A.1 — Backend SPI Audit

## 1. Executive recommendation

Keep the three concerns separate, but define one explicit composition path:
- Provider SPI owns provider identity, compatibility metadata, provider-scoped initialization and disposal, and eventually provider/device discovery.
- Tensor ComputeBackend owns the tensor operation surface. It is the operation engine exposed to tensor/model code, not a provider registry.
- KernelPlugin is currently a separate, incomplete plugin execution abstraction. Its intended role must be decided before it can be adapted or retired.

A concrete provider should expose one or more tensor compute backends through an explicit provider-owned runtime/session object. AQLI (or another consumer) chooses a provider and device; Aljabr owns provider mechanics and backend execution. This must not rely on process-global mutable selection.

Do not add a fourth compute SPI. The existing core.backend.ComputeBackend is the only repository-local tensor operation contract found in the default source tree. First reconcile references to the absent spi.tensor.ComputeBackend, then add only the smallest provider-to-backend composition contract that the audit demonstrates is missing.

## 2. Contract-by-contract comparison

| Dimension | AljabrProvider SPI | core.backend.ComputeBackend | KernelPlugin |
|---|---|---|---|
| Main role | Provider lifecycle and identity | Tensor mathematical operations | Generic kernel-operation execution |
| Package/module | spi.provider, aljabr-spi-core | core.backend, aljabr-tensor | plugin.kernel, currently under DirectML plugin module |
| Identity | ProviderDescriptor id/version/API compatibility | No stable id or descriptor | id/name/version/description/platform |
| Capabilities | Declared CapabilityDescriptor values | No capability query | Supported architecture/version/operation string sets |
| Device discovery | None in interface | None | No structured device enumeration; concrete plugins perform vendor detection |
| Initialization | initialize(ProviderContext), default no-op | None | Interface does not declare initialization |
| Execution | None | Strongly typed Tensor operation methods | Generic execute(KernelOperation, KernelContext) |
| Resource lifecycle | close(), default no-op | None | Interface does not declare close/shutdown |
| Readiness/health | None | None | Interface has isAvailable(); concrete classes add validation/health methods not declared on interface |
| Configuration | Immutable string map in ProviderContext | None | Concrete plugin-specific KernelConfig via KernelContext |
| Discovery | Javadoc says ServiceLoader; no provider descriptor found | No generic registration contract | Plugin manager/registries exist; not a common provider registry |
| Build maturity | Included foundational SPI | Included tensor API | Interface is in excluded DirectML module; CUDA plugin module is excluded in root settings |

### 2.1 Provider SPI — good lifecycle foundation, insufficient compute contract

Current public shape: AljabrProvider.descriptor(), initialize(ProviderContext), close(); ProviderDescriptor(id, version, apiCompatibility, capabilities); ProviderContext(aljabrVersion, Map<String,String>); CapabilityDescriptor(id, version, stability).

Strengths: small and accelerator-SDK-independent; explicit provider-scoped initialization and close; immutable descriptors and configuration; appropriate dependency direction for independently published providers.

Gaps: no typed provider runtime/session return value; no device inventory or device identity model; no distinction between detected, available, initialized, and validated; no structured per-operation capability/readiness information; string-only configuration is weakly typed; documented ServiceLoader discovery is not demonstrated by repository service descriptors or concrete provider implementations.

**Recommendation:** retain the generic SPI and evolve it additively only after defining a provider-owned runtime/session boundary. Do not put every tensor operation into AljabrProvider.

### 2.2 Tensor ComputeBackend — correct operation layer, overly broad contract

The current core.backend.ComputeBackend is a large typed interface covering elementwise/scalar operations, matrix multiplication and embeddings, shape transforms, reductions and activations, normalization, convolution, pooling, dropout, losses, attention, device transfer, casts, and allocation helpers.

Strengths: strongly typed against Aljabr Tensor/DType/device abstractions; fits tensor/model code and is independent of provider discovery; implemented by included CPU and Metal backend source.

Gaps and risks: one interface spans core tensor math, training operations, and inference-specific operations such as attention without capability negotiation. Its Javadoc says implementations must provide hardware-specific kernels for all methods, while accelerator implementations visibly delegate many operations to CPU. Unsupported operations are represented by exceptions/default methods and implementation fallback without a common support query. It has no explicit backend identity, device binding, lifecycle, or concurrency contract. CUDA/ROCm plugin sources reference spi.tensor.ComputeBackend and spi.tensor.CpuBackend, absent from the audited main tree; these references must not be silently mapped to the core tensor API without checking method signatures and callers.

**Recommendation:** retain core.backend.ComputeBackend as the canonical tensor operation surface for now. In a later compatibility-reviewed change, correct its behavioral contract and introduce operation support/capability descriptors outside the interface or via a companion capability view. Do not add provider discovery methods to this operation interface.

### 2.3 KernelPlugin — generic operation wrapper with unclear ownership

The inspected KernelPlugin interface declares plugin identity/descriptive metadata, isAvailable/platform, supported architectures/versions/operations, generic execute(KernelOperation, KernelContext), and a metadata map.

Important mismatch: concrete CUDA and Metal plugin classes also define validation, initialization, health, and shutdown-style behavior. Those lifecycle methods are not declared by the inspected KernelPlugin interface itself, despite concrete classes using @Override on some of them. The interface lives under the DirectML plugin module, which is excluded from the root build; CUDA plugin and its dependent modules are also excluded. Metal plugin source is not part of the explicit Metal main-source include list.

Strengths: generic kernel-operation dispatch may be useful for runner-level or fused-kernel operations; supported-operation metadata can inform a dispatcher; existing plugin classes wrap concrete runner implementations and contain useful device diagnostics.

Risks: generic KernelOperation/result contract overlaps with, but is not interchangeable with, typed Tensor operations. Lifecycle behavior is not consistently represented in the interface. Package/module ownership under DirectML makes a cross-backend SPI dependency awkward. Current build exclusions mean source presence is not proof of a supported public extension point.

**Recommendation:** do not promote KernelPlugin as the canonical compute provider or tensor backend SPI. Phase 4A.3 must decide whether it is a runner/plugin API for fused or specialized operations, legacy code to deprecate after migration, or a private adapter implementation detail. If retained, move its neutral contract to a neutral SPI module only after dependency and API compatibility review, and make lifecycle explicit.

## 3. Proposed ownership/composition model

```text
Consumer runtime (AQLI or another Aljabr client)
  | selects provider + device + fallback policy
  v
Aljabr provider registry / discovery
  | ServiceLoader: AljabrProvider
  v
Concrete provider
  | initialize(provider-scoped context)
  | discover devices / report capabilities
  | open selected device session
  v
Provider runtime/session
  | exposes one or more tensor operation backends
  v
core.backend.ComputeBackend
  v
Tensor / model execution
```

Optional specialized-kernel path: provider runtime/session → specialized kernel dispatcher/adapter → KernelPlugin (only if retained and repaired). This is conceptual; it does not imply these components already exist.

## 4. Responsibilities

| Responsibility | Owner |
|---|---|
| Discover installed provider implementations | Aljabr provider registry |
| Detect physical devices and driver/runtime facts | Provider implementation |
| Describe normalized capabilities and operation support | Provider implementation using Aljabr-neutral types |
| Choose provider/device for a serving process or model | Consumer runtime (AQLI policy) |
| Initialize and release provider resources | Aljabr provider runtime/session |
| Execute tensor operations | core.backend.ComputeBackend implementation |
| Decide whether fallback is allowed | Consumer runtime |
| Report which operations used CPU vs accelerator | Backend execution diagnostics; surfaced by consumer |
| Request scheduling, queues, TTFT and throughput | AQLI |
| Package optional provider modules/native libraries | Build/distribution configuration |

## 5. Minimum API direction (not final signatures)

The eventual design should support these concepts without one oversized SPI:
1. Provider descriptor with stable id, version, API compatibility, and declared capability identifiers.
2. Provider discovery/registry service using a defined classloader and reporting discovery errors.
3. Provider instance initialized for one Aljabr runtime, with explicit close semantics.
4. Device descriptor with stable provider-local identity, display name, device kind, memory facts, and driver/runtime versions where available.
5. Provider-owned device/session handle for a selected device.
6. Session exposes one or more core.backend.ComputeBackend instances.
7. Structured readiness states: declared, implementation present, packaged/installed, device detected, device available, provider initialized, operation validated.
8. Operation support distinguishes native, emulated/delegated, unsupported, and runtime-failed implementations.
9. Explicit close/release for sessions, backend-owned buffers, and provider resources.
10. Stable errors for discovery, initialization, device selection, unsupported operation, and execution failure.

Keep these as small interfaces/value types. The provider SPI must not depend on AQLI classes, inference request types, logging APIs, or vendor SDK classes.

## 6. Fallback semantics

Current CUDA and Metal source shows three distinct behaviors:
- Operation delegation: backend deliberately uses CPU for operations it does not accelerate.
- Initialization fallback: accelerator initialization fails and implementation starts in CPU mode.
- Execution-failure fallback: native accelerator operation fails and the operation is retried on CPU.

Proposed policy: make all three observable; consumer owns whether provider-level fallback is allowed; operation delegation may be declared but must not be reported as native accelerator execution; runtime failure fallback is opt-in and surfaced as a structured outcome/diagnostic. Preserve current behavior during migration. Add tests and observability before altering defaults.

## 7. Compatibility and migration sequence

| Step | Change | Compatibility posture |
|---|---|---|
| 1 | Repair stale CUDA/ROCm references after signature/call-site comparison | Internal module fix unless public contract change is required |
| 2 | Add conformance tests for current core.backend.ComputeBackend implementations | Additive |
| 3 | Define provider-to-device/session composition using existing AljabrProvider | Additive SPI evolution; separate review |
| 4 | Implement CPU provider adapter first | Additive; retain direct CPU construction temporarily |
| 5 | Implement Metal provider adapter and validate on user Mac | Additive; retain current Metal path until parity |
| 6 | Adapt CUDA/ROCm only when modules compile and runtime paths are real | Optional modules; no premature support claim |
| 7 | Decide KernelPlugin disposition and migrate proven use cases | Gradual deprecation; no blanket deletion |
| 8 | Add explicit consumer selection and fallback policy | Consumer-side integration; no global singleton |
| 9 | Deprecate old constructors/selection after compatibility window and tests | Planned release only |

## 8. Required validation gates

- Compile default CPU, CUDA, and Metal project sets independently.
- Verify each ServiceLoader descriptor names an existing interface and concrete no-arg provider class.
- Test provider discovery, duplicate ids, API incompatibility, initialization failure, partial-init close, and isolated classloaders.
- Test device enumeration and readiness transitions with fake providers (no accelerator hardware required).
- Add backend conformance tests for shape, dtype, numerical tolerance, unsupported operations, and resource ownership.
- Test fallback policy and prove fallback is observable.
- Confirm Aljabr has no dependency on AQLI.
- Run Metal integration tests on Apple Silicon only after CPU/provider adapter is stable.

## 9. Open decisions for Phase 4A.3

1. Is KernelPlugin supported public API or internal legacy abstraction?
2. Should one provider expose one default backend or multiple backends (e.g. compute and specialized attention engines)?
3. What is the minimum stable device descriptor across CPU, Metal, CUDA, ROCm, and future accelerators?
4. Should operation support be queried through a separate capability interface or backend descriptor?
5. How should asynchronous execution/streams be represented without forcing every backend into a GPU-specific model?
6. Which fallback modes are allowed by default? Recommendation: no provider-level fallback without explicit consumer policy.
7. Which modules are publishable artifacts versus experimental source-only modules?

## 10. Non-goals

- No new public SPI is introduced in this proposal.
- No backend source is deleted or rewritten.
- No fallback defaults are changed.
- No ServiceLoader registration is activated for currently excluded modules.
- No performance claims or hardware benchmark results are asserted.
- No Mac/Metal hardware testing is required for this documentation-only comparison.