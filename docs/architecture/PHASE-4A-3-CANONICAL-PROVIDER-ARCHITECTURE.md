# Phase 4A.3 — Canonical Backend Provider Architecture

**Status:** Architecture decision proposal; implementation not authorized by this document
**Date:** 2026-11-04
**Depends on:** ADR-0001, Phase 4A.1 provider/SPI audit, Phase 4A.2 contract comparison

## 1. Decision summary

This document resolves the architectural questions required before provider implementation. It defines a single composition model while preserving the existing tensor operation API and existing backend construction paths during migration.

1. `AljabrProvider` remains the provider identity, discovery, and lifecycle SPI.
2. A provider may expose multiple devices and multiple backend instances. A selected device is opened as a provider-owned session; a session exposes one or more `core.backend.ComputeBackend` instances.
3. `core.backend.ComputeBackend` remains the canonical tensor-operation contract. It does not become a registry, device descriptor, lifecycle manager, or policy engine.
4. Device identity, readiness, and operation support are represented by backend-neutral value types and companion query contracts, not vendor SDK types and not additions to the large tensor interface.
5. Execution remains backend-neutral. Optional execution-context/queue capabilities may be exposed by a companion contract; synchronous CPU implementations remain valid and GPU-specific stream types do not leak into the common SPI.
6. `KernelPlugin` is not promoted to a provider or tensor SPI. Treat it as legacy/internal specialized-kernel infrastructure while call sites and module ownership are audited. No deletion or public deprecation is implied yet.
7. Aljabr reports facts and performs provider/session mechanics. The consuming application (AQLI, Tafkir, or another client) selects providers/devices and owns fallback policy.
8. Provider-level fallback is never implicit in the new contract. Operation delegation may exist but must be declared and observable.

## 2. Canonical composition model

```text
Consumer application / runtime
  - chooses provider, device, workload policy, and fallback permission
  - owns serving/training/application configuration
             |
             v
Aljabr ProviderRegistry (classloader-scoped)
  - discovers AljabrProvider implementations
  - validates descriptors and reports discovery failures
             |
             v
AljabrProvider
  - descriptor()
  - initialize(provider-scoped ProviderContext)
  - enumerate devices / report provider readiness
  - open(selected DeviceDescriptor, session options)
             |
             v
ProviderSession (provider + selected device + owned resources)
  - exposes one or more ComputeBackend instances
  - reports session/backend capabilities and diagnostics
  - closes session-owned resources deterministically
             |
             v
core.backend.ComputeBackend
  - typed tensor operations
             |
             v
Tensor / model execution
```

A provider is an installable implementation. A device is a provider-reported execution target. A session is the lifetime boundary for using a selected device. A compute backend is an operation implementation exposed by that session. These concepts must not be collapsed into one object.

## 3. Contract responsibilities

| Contract | Owns | Must not own |
|---|---|---|
| `AljabrProvider` | Stable provider identity, compatibility metadata, provider initialization, device enumeration, session creation, provider close | Tensor operation methods, AQLI policy, model/request types |
| `ProviderRegistry` | Classloader-scoped discovery, duplicate-id handling, descriptor validation, discovery diagnostics | Device selection policy, process-global mutable default |
| `DeviceDescriptor` | Stable provider-local device id, display name, device kind, optional memory and driver/runtime facts | Vendor SDK handles, mutable health state, selection policy |
| `ProviderSession` | Selected device binding, session resources, one-or-more backend instances, session diagnostics and close | Cross-provider fallback policy, serving scheduling |
| `ComputeBackend` | Typed tensor operations and their documented mathematical/resource semantics | Provider discovery, device inventory, lifecycle, policy |
| Capability/support views | Structured readiness and per-operation support facts | Claiming unvalidated support or silently changing execution target |
| Consumer runtime | Provider/device selection, runtime configuration, fallback permission, workload scheduling | Reimplementing provider/device mechanics |

## 4. Provider cardinality and session ownership

- One provider may enumerate zero, one, or many devices. A CPU provider may expose one or more logical CPU execution targets according to its implementation; an accelerator provider may expose multiple physical devices.
- One provider may open multiple independent sessions, including sessions for different devices, subject to provider capabilities and resource limits.
- One session may expose multiple compute backends when there is a real implementation need (for example, a general tensor backend plus a specialized attention backend). A session must identify each backend's role; callers must not infer that every backend supports the full tensor operation set.
- The common minimum is one default `ComputeBackend` per usable session. Additional backends are optional and additive.
- A provider instance may be shared by multiple sessions only if its descriptor/capabilities explicitly allow concurrent sessions. Otherwise the provider must reject conflicting opens with a typed error.
- Session close releases session-owned queues, buffers, command resources, and backend resources. Provider close releases provider-wide resources and is safe after partial initialization. Close operations must be idempotent or document a deterministic repeated-close result.
- A failed session open must clean up partially acquired resources before returning failure. Closing one session must not invalidate unrelated sessions unless the provider explicitly documents a shared-resource failure.

## 5. Device descriptor minimum

The common descriptor should be a small immutable value object with:

- `providerId`: the exact stable id from `ProviderDescriptor`.
- `deviceId`: stable within that provider implementation and host; not assumed globally unique across providers or machines.
- `name`: human-readable display name.
- `kind`: normalized category such as `CPU`, `GPU`, `ACCELERATOR`, or `OTHER`; allow future extension without vendor enum dependencies.
- Optional total/free memory facts with explicit units and unknown values represented as absent, not zero.
- Optional vendor, architecture, driver version, and runtime version strings.
- Optional ordinal/index for display only; never use ordinal as the persistent identity.

Device descriptors are snapshots. Dynamic free memory, temperature, utilization, and health belong in a separate observation/diagnostic view, not in descriptor equality or identity.

## 6. Readiness and support semantics

Do not represent all readiness with one boolean. The implementation must distinguish:

| State/fact | Meaning |
|---|---|
| `DECLARED` | A provider or capability contract is known to Aljabr |
| `DISCOVERED` | A provider implementation was found by the configured classloader |
| `INSTALLED` | Required provider artifacts are present in the distribution/runtime |
| `DEVICE_DETECTED` | The provider can enumerate the target device |
| `AVAILABLE` | Host, driver, native runtime, permissions, and prerequisites permit initialization |
| `INITIALIZED` | Provider initialization completed successfully |
| `SESSION_OPEN` | A session for the selected device was created successfully |
| `VALIDATED` | The relevant implementation passed declared correctness validation for the recorded environment/version |

These are lifecycle/evidence facts, not necessarily a single linear enum. For example, one device can be available while another device from the same provider is not. `VALIDATED` must include scope/version/environment evidence; it must not be inferred from successful initialization.

Operation support is queried separately from provider readiness. Each operation/dtype/layout/shape support result should distinguish at least:

- `NATIVE`: executed by the selected backend implementation.
- `DELEGATED`: deliberately executed by another implementation/backend (for example CPU), with the effective backend identified.
- `EMULATED`: implemented through a semantically equivalent decomposition rather than a native kernel.
- `UNSUPPORTED`: known not to be supported for the queried constraints.
- `UNKNOWN`: no reliable support claim is available.

A runtime failure is an execution outcome, not a support category. It must carry a typed failure and may update diagnostics/health without rewriting the static capability declaration.

## 7. Execution, queues, and asynchronous work

The common provider SPI must not require GPU streams, command buffers, CUDA events, Metal command queues, or any vendor-specific asynchronous primitive.

- Synchronous `ComputeBackend` implementations remain first-class and valid.
- If asynchronous execution is needed, expose it through a separate optional execution-context/queue capability associated with a session, not by forcing all tensor methods to return futures.
- The abstraction must describe ordering, completion, error propagation, and resource lifetime. It must permit a CPU implementation to provide a direct/synchronous execution context.
- Device memory and tensor ownership must remain explicit. A tensor used by an asynchronous operation cannot be released/reused until the operation's completion contract permits it.
- Cross-device synchronization and transfer are explicit operations; do not imply that two sessions share ordering or memory.
- Introduce this companion contract only when concrete CPU and accelerator use cases define its minimum semantics. Phase 4A.3 establishes the boundary, not the final async API signatures.

## 8. Capability model

Keep provider-level capabilities and operation-level support distinct:

1. `ProviderDescriptor.capabilities` remains a set of stable capability identifiers and versions describing broad provider features.
2. Device/session capability views describe facts that vary by device or initialized session, such as supported data types, memory limits, concurrency, and optional execution modes.
3. Operation support queries answer whether a specific operation is supported for the supplied constraints (operation, dtype, layout, shape, and relevant options).
4. Capability claims may be `DECLARED`, `AVAILABLE`, or `VALIDATED`; callers must not treat declaration as proof of runtime usability.

Do not put every operation, dtype, shape, and device fact into `ProviderDescriptor`. Keep descriptors compact and allow optional capability interfaces so providers can evolve independently.

## 9. Discovery and classloader rules

- `ProviderRegistry` is scoped to an explicit `ClassLoader` (defaulting to the consumer's chosen loader), not a mutable process-global singleton.
- Discovery uses `ServiceLoader<AljabrProvider>` and validates non-null descriptors, stable ids, API compatibility, and duplicate ids.
- A duplicate provider id is a discovery error with both implementation class names reported; silently choosing the first provider is prohibited.
- A provider that fails construction or descriptor validation must not prevent unrelated providers from being discovered. Return per-provider diagnostics alongside usable providers.
- Discovery must not initialize native runtimes or allocate device resources. Initialization occurs only after consumer selection.
- Optional provider artifacts and native libraries remain optional. Absence of CUDA/Metal/etc. must not prevent core and CPU use.
- Service descriptors are added only for provider modules that are included and independently buildable. Source presence in an excluded module is not evidence of a supported provider.

## 10. Fallback and execution diagnostics

Fallback is a consumer policy decision. The provider contract must make actual execution visible:

- Operation delegation is permitted when the implementation advertises it and diagnostics identify the effective backend.
- Provider initialization fallback and execution-failure fallback require explicit consumer permission.
- If fallback is disabled, initialization or execution failure is surfaced; no hidden switch to CPU is allowed.
- If fallback is enabled, the outcome records requested provider/device, effective provider/device/backend, triggering error, and whether the operation was retried.
- A retry is not always safe (especially for mutating or partially completed asynchronous work). The backend must report whether execution began/completed sufficiently to determine retry safety; the consumer policy must not blindly retry ambiguous failures.
- Existing fallback behavior is preserved during the migration until call sites, tests, and compatibility impact are understood. This document does not silently change defaults.

## 11. KernelPlugin disposition

`KernelPlugin` is not the provider SPI and is not interchangeable with `ComputeBackend`. It may remain useful for specialized runner-level/fused-kernel dispatch, but the current interface/module/lifecycle mismatch prevents treating it as a stable cross-backend extension point.

Until a dedicated call-site audit is complete:

- classify it as legacy/internal infrastructure;
- do not add new provider implementations against it;
- do not delete it or break existing users;
- do not move it into a neutral SPI module yet;
- identify concrete consumers and determine whether their operations belong in `ComputeBackend`, a session-owned specialized-kernel service, or private implementation details.

A later ADR/change must explicitly decide public support, neutral module placement, lifecycle contract, compatibility policy, and migration/deprecation before any move or removal.

## 12. API evolution and compatibility rules

- Keep existing `ComputeBackend` method signatures unchanged during provider composition work.
- Evolve `AljabrProvider` additively or introduce a narrowly scoped companion interface only if binary/source compatibility analysis proves the current interface cannot express provider/session composition.
- Do not introduce a second tensor-operation interface or a second provider registry.
- Do not add AQLI classes, request/serving concepts, logging framework dependencies, reactive libraries, or vendor SDK types to `aljabr-spi-core`.
- Preserve direct CPU/Metal/CUDA construction and existing runner paths while adapters are introduced and tested.
- Mark experimental provider modules and APIs clearly; do not advertise excluded or non-compiling modules as supported artifacts.
- Any deprecation requires call-site inventory, replacement guidance, tests, and a documented compatibility window.

## 13. Conformance and validation gates

Before a provider is considered integrated:

1. **Discovery:** provider absent, one provider, multiple providers, duplicate ids, broken provider construction, incompatible API version, and isolated classloaders.
2. **Lifecycle:** initialize success/failure, partial initialization cleanup, multiple sessions, concurrent session rules, idempotent close, provider close after session close.
3. **Devices:** zero devices, one device, multiple devices, stable ids, optional metadata, unavailable device, and device disappearance.
4. **Capabilities:** declared vs available vs validated; operation/dtype/shape support; native/delegated/emulated/unsupported/unknown reporting.
5. **Execution:** tensor backend conformance, numerical tolerances, unsupported operation behavior, resource ownership, and failure diagnostics.
6. **Fallback:** disabled fallback, explicitly enabled initialization fallback, explicitly enabled execution fallback, retry-safety ambiguity, and observable effective backend.
7. **Isolation:** two providers and two sessions coexist without process-global backend mutation or cross-session resource invalidation.
8. **Distribution:** CPU-only packaging works without optional provider/native artifacts; optional provider packaging is tested independently.
9. **Boundary:** Aljabr has no dependency on AQLI; consumer selection policy remains outside Aljabr.

Fake-provider and fake-device tests should cover the generic lifecycle and discovery paths without accelerator hardware. Metal/CUDA hardware validation remains a separate gate for their concrete provider adapters.

## 14. Implementation sequence after this proposal

| Order | Work item | Why this order |
|---|---|---|
| 1 | Inventory provider call sites, service descriptors, build inclusion, and legacy constructors | Prevents designing against unused or excluded source |
| 2 | Define immutable device/readiness/support value types and provider-session composition | Smallest shared contract needed by all backends |
| 3 | Add provider registry/discovery tests with fake providers | Validates lifecycle and optional-provider isolation without hardware |
| 4 | Implement CPU provider adapter while retaining direct CPU construction | Establishes the reference implementation and conformance suite |
| 5 | Add compatibility adapters for Metal, then validate on Apple Silicon | Preserves current Metal behavior until parity is demonstrated |
| 6 | Adapt CUDA/ROCm only after their modules/build/runtime paths are verified | Avoids claiming support based on excluded or stale source |
| 7 | Resolve `KernelPlugin` disposition from actual call sites | Prevents premature public API promotion or deletion |
| 8 | Integrate AQLI selection and fallback policy through Aljabr's public contracts | Keeps policy in the inference runtime |

## 15. Non-goals

- No production Java API or backend implementation changes.
- No change to existing CPU, Metal, CUDA, or ROCm fallback behavior.
- No new service descriptor or provider activation.
- No changes to Gradle module inclusion or artifact publication.
- No hardware benchmark or performance claim.
- No AQLI configuration schema or serving policy is defined here.
- No KernelPlugin deletion, move, or public deprecation.

## 16. Acceptance

Phase 4A.3 is complete when this architecture is reviewed and accepted as the basis for additive implementation. The next code change should be a narrowly scoped SPI/value-type design with compatibility tests, not a wholesale backend rewrite.
