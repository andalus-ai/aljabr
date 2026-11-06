# ADR-0003: Capability registry design

- **Status:** Proposed
- **Date:** 2026-11-05
- **Related:** `core:aljabr-spi-core`, `core:aljabr-spi-compute`, `core:aljabr-spi-testkit`, ADR-0002
- **Scope:** General capability metadata and runtime registration; not a serving/runtime-selection policy

## Context

Aljabr already has a minimal provider SPI. `AljabrProvider` exposes a `ProviderDescriptor`, and that descriptor contains a set of provider-declared `CapabilityDescriptor` values. A separate `ComputeProvider` extension exposes devices and device-scoped `ProviderSession` objects. The base provider test kit checks descriptor availability and provider initialize/close lifecycle. Compute sessions can report conservative operation-level support using `OperationSupportDescriptor`.

These mechanisms provide useful provider and operation contracts, but they are not yet a general registry:

- A descriptor is a provider's declaration, not a runtime inventory.
- A device/session query is compute-specific, not a general capability query.
- A provider lifecycle test is not a capability correctness conformance result.
- Provider-declared stability is not runtime availability, compatibility, or validation.
- AQLI runtime provider selection and serving policy must remain outside Aljabr.

A registry should compose these existing contracts. It must not introduce a competing provider SPI or assume every capability is a device-backed compute operation.

## Decision proposal

Introduce a small Aljabr-owned capability registry as a runtime service, after the Phase 0 inventory confirms the correct owning module and existing registration mechanisms.

The registry is responsible for indexing provider declarations and recording Aljabr-observed runtime/validation state. It is not responsible for discovering arbitrary classes by scanning the classpath, downloading extensions, selecting serving providers, or scheduling computation.

### 1. Separate discovery, registration, and initialization

Treat these as distinct steps:

1. **Discovery:** locate provider factories through Java `ServiceLoader` or application-supplied explicit registration. Discovery does not imply provider initialization.
2. **Registration:** validate and index a provider's immutable descriptor and its declared capabilities. Registration should not open devices or initialize heavyweight native resources.
3. **Initialization:** initialize only providers selected by the owning application/runtime. Initialization failure must be recorded and cleanup attempted.
4. **Availability refresh:** query provider-specific runtime facts only through the relevant domain SPI (for example, `ComputeProvider.devices()`). Do not force all providers to expose devices.
5. **Validation:** associate conformance results with provider version, Aljabr API compatibility, capability ID/version, test-suite version, and validation environment.
6. **Unregistration/shutdown:** close initialized provider instances according to the established lifecycle and remove their runtime registrations. Do not close an instance still owned by another runtime.

The registry must not silently initialize every discovered provider just to enumerate metadata.

### 2. Keep provider declarations immutable and runtime observations separate

Do not mutate `ProviderDescriptor` or `CapabilityDescriptor` to store transient state. Keep at least two conceptual records:

- **Declared metadata:** provider ID/version/API compatibility and the provider-declared capability IDs, versions, and maturity labels.
- **Runtime registration record:** the specific provider instance/factory registration, lifecycle state, compatibility assessment, availability observations, validation evidence, and diagnostics.

This prevents transient hardware state or test outcomes from contaminating a provider's immutable published metadata.

### 3. Use explicit, non-collapsed state dimensions

The registry should expose separate dimensions rather than one `available` boolean:

| Dimension | Meaning | Authority |
|---|---|---|
| Declared | Provider descriptor claims this capability | Provider |
| Installed/discovered | Provider artifact/service or explicit registration is present in this runtime | Runtime discovery/registration |
| Compatible | Provider/API/capability versions and declared constraints satisfy the runtime contract | Aljabr compatibility evaluator |
| Initialized | Provider lifecycle initialization completed for this runtime | Runtime lifecycle manager |
| Available | Required runtime resources are presently usable | Relevant provider/domain SPI |
| Validated | Applicable conformance suite passed for the recorded versions/environment | Aljabr conformance process |
| Stable | Aljabr's published capability maturity designation | Aljabr release governance |
| Optimized | Benchmark evidence supports an optimization claim under recorded conditions | Benchmark process |

Do not interpret `CapabilityStability.STABLE` in a provider descriptor as proof of Aljabr stability or validation. Treat it as a provider declaration until a migration policy defines how existing values map to Aljabr-owned maturity.

### 4. Capability identity and versioning

Use stable, namespaced capability identifiers, for example:

- `tensor.compute`
- `tensor.add`
- `linear-algebra.dense`
- `statistics.descriptive`
- `statistics.inference`
- `optimization.gradient`
- `model.execution.transformer`
- `tokenizer.text`

These are illustrative identifiers, not a final catalog. The Phase 0 inventory and Phase 3 semantics audit must identify existing operation/capability names before the catalog is frozen.

Keep three versions distinct:

- Provider implementation version.
- Aljabr provider API compatibility declaration.
- Capability contract version.

Do not assume semantic-versioning range strings are safely comparable until the repository selects and tests a version-range grammar. The current `apiCompatibility` string is a declaration; a future evaluator must define its syntax, invalid-range behavior, and compatibility policy explicitly.

### 5. Registry query model

The registry should support read-only metadata queries without forcing provider initialization, and runtime queries that clearly identify which state dimensions they require.

Conceptual query needs:

- Find providers by provider ID.
- List declared capabilities.
- Find providers declaring a capability ID and compatible capability version.
- Inspect registration, initialization, compatibility, availability, validation, and diagnostics independently.
- Query domain-specific support using the owning SPI when available (for compute, the existing `ProviderSession.operationSupport` API).
- Return deterministic, immutable snapshots suitable for diagnostics and reproducibility.

Do not make a general capability query imply that a provider can execute immediately. A declaration-only query and an executable/readiness query must be visibly different.

### 6. Duplicate and conflict policy

- Provider IDs must be unique within one registry/runtime scope.
- Registering the same provider ID and same implementation identity twice should be idempotent only if the ownership semantics are explicit; otherwise reject duplicate registration.
- Two provider versions with the same provider ID may coexist only if the runtime can represent and isolate them safely. Do not assume this is supported initially.
- Duplicate capability IDs across different providers are expected: providers compete or complement each other.
- A single provider declaring the same capability ID more than once with conflicting versions/metadata is invalid.
- Conflicts must produce deterministic diagnostics; do not resolve them by classpath order.

### 7. Thread safety and lifecycle ownership

The registry must define whether it is scoped to one Aljabr runtime. The initial design should use one explicit runtime/registry scope, with no hidden global singleton.

Registration and state transitions must be thread-safe. Query snapshots must be immutable. Provider initialization and close must not occur while holding broad registry locks; use explicit state transitions and carefully handle races between unregister, initialization, and shutdown.

The registry owns only provider instances/factories explicitly registered into its scope. It must close only instances it owns. If applications supply externally managed instances, ownership must be explicit in the registration API.

### 8. Diagnostics

Every failed or rejected transition should be inspectable with structured information:

- provider ID and version;
- capability ID and contract version where relevant;
- lifecycle/compatibility state;
- failure category;
- concise reason and causal exception where safe;
- timestamp or validation-run identity;
- relevant environment details for runtime availability and conformance.

Diagnostics must not expose secrets from provider configuration. Do not add a mandatory logging framework to the registry; expose structured diagnostic values and let the application decide how to log them.

## Explicit non-goals

The first registry does not:

- Download or install executable provider artifacts.
- Sandbox untrusted in-process Java code.
- Replace Java `ServiceLoader` or explicit application wiring.
- Select AQLI serving providers or own request scheduling/fallback policy.
- Replace `ComputeProvider`, `ProviderSession`, GGUF, Safetensors, model, tokenizer, or plugin domain contracts.
- Claim correctness based on provider-declared stability or self-reported support.
- Become a distributed registry or remote control plane.
- Automatically benchmark providers during registration.
- Require CDI, Mutiny, a dependency-injection framework, a logging implementation, GPU SDKs, or native libraries.

## Proposed module placement

Do not create a new Gradle module until the Phase 0 module/dependency inventory is complete.

The implementation should live in a small runtime/registry module that depends on the base provider SPI, but not on tensor, compute, inference, model, tokenizer, CDI, or accelerator modules. Domain-specific integrations may contribute availability/readiness adapters in their own optional modules.

The registry API should be usable by a statistics-only application without pulling tensor/GPU/DL dependencies.

## Migration and implementation sequence

1. Complete the Phase 0 inventory of existing discovery/registration mechanisms across compute, model, tokenizer, plugin, GGUF, and Safetensors.
2. Confirm provider ownership semantics and the existing `apiCompatibility` version grammar.
3. Define immutable registration snapshots and explicit lifecycle states.
4. Define duplicate-registration and multi-runtime ownership rules.
5. Implement the minimal registry with deterministic metadata queries and no implicit initialization.
6. Add tests for duplicate IDs, descriptor failures, compatibility decisions, initialization/close races, partial initialization cleanup, immutable snapshots, and externally managed instances.
7. Add compute-specific readiness integration using the existing `ComputeProvider` and `ProviderSession` contracts, without adding compute dependencies to the base registry.
8. Extend the conformance test kit with report metadata and capability-specific suite integration; do not make the base lifecycle kit claim numerical correctness.
9. Publish an external-provider example using only released Aljabr APIs and SPIs.

## Acceptance criteria

- Existing `AljabrProvider` and `ComputeProvider` contracts are reused; no duplicate generic provider SPI is introduced.
- A registry can list declarations without initializing providers or loading native libraries.
- Registration, initialization, availability, validation, and maturity are distinguishable.
- Duplicate registration and ownership behavior are deterministic and tested.
- Registry core has no mandatory tensor, GPU, native, CDI, Mutiny, or logging implementation dependency.
- Compute readiness uses the existing compute SPI as an optional integration.
- An external provider can register and be inspected without depending on Aljabr internals.
- AQLI retains serving selection and orchestration policy.

## Open questions for implementation review

1. Should registration accept provider instances, provider factories, or both? Prefer factories for runtime isolation, but preserve explicit instance registration for embedding/testing only if ownership is clear.
2. What version-range grammar should replace informal `apiCompatibility` strings, and is a dedicated dependency justified?
3. Should availability be a pull-based query, cached observation with TTL, or domain-specific event? Start with explicit pull/refresh unless current runtime requirements demonstrate a need for events.
4. How should capability maturity be published and versioned independently of provider declarations?
5. Which existing format/model/plugin registries must remain independent rather than being folded into the general registry?

These questions should be resolved after the Phase 0 inventory and before implementation.