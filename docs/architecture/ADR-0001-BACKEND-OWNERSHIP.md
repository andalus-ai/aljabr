# ADR-0001: Backend Discovery, Configuration, and Profiling Ownership

- **Status:** Accepted
- **Date:** 2026-11-04
- **Scope:** Backend provider discovery, device/capability reporting, initialization, profiling, selection policy, packaging, and operational telemetry

## Context

Aljabr is a reusable Java machine-learning framework and compute engine. AQLI is an inference runtime that consumes Aljabr. Research, desktop, training, and future inference applications must be able to use Aljabr without depending on AQLI.

Backend availability has two distinct dimensions:

1. **Distribution-time availability:** which provider artifacts and native libraries are included in an installation or deployment.
2. **Runtime usability and selection:** which installed providers can initialize on the current host, which devices and operations are usable, and which provider an application chooses for a workload.

Conflating these decisions creates accidental mandatory accelerator dependencies, hidden fallback, and application-specific policy inside reusable libraries.

## Decision

### 1. Aljabr owns reusable backend mechanisms

Aljabr owns:

- Backend provider SPI and provider lifecycle contracts.
- Provider discovery and registration mechanisms.
- Device discovery APIs and provider-reported device information.
- Backend and operation capability declarations.
- Capability and environment readiness queries.
- Backend initialization and release primitives.
- Device memory and execution constraints.
- Kernel registries, kernel selection mechanisms, and backend dispatch.
- Low-level kernel benchmarking and backend-level profiling interfaces.
- Execution diagnostics, typed backend errors, and provider readiness information.
- Backend implementations, native bindings, and resource lifetime management.

Aljabr reports facts and provides mechanisms. It does not make AQLI serving or deployment decisions.

### 2. AQLI owns inference-runtime policy and operations

AQLI owns:

- Runtime configuration files and configuration precedence.
- Backend selection policy, including automatic selection, explicit provider selection, and per-model overrides.
- Model/session execution options and serving-level resource budgets.
- Startup and model-load orchestration that queries Aljabr.
- Whether fallback is permitted for a workload.
- Production profiling and aggregation: request latency, TTFT, throughput, token rates, queueing, concurrency, and serving memory metrics.
- Operational dashboards, alerts, and telemetry export.
- Request scheduling, admission, routing, streaming, and serving lifecycle.
- Persistence and policy for using benchmark results to choose a provider.

AQLI may invoke Aljabr profiling APIs and interpret their results. Aljabr must not depend on AQLI configuration types, dashboards, request metrics, or scheduling policies.

### 3. Distribution and installation own provider composition

Build, packaging, installation, and deployment tooling determine which provider modules and required native libraries are included.

Examples:

- Apple Silicon distribution may include Aljabr core, Java CPU provider, and Metal provider.
- NVIDIA distribution may include Aljabr core, Java CPU provider, and CUDA provider.
- CPU-only distribution may include Aljabr core and Java CPU provider only.

These are package profiles, not runtime selection rules. Aljabr core must remain usable when optional accelerator providers are absent.

### 4. Runtime selection is independent of packaging

At startup or model load, AQLI queries Aljabr for installed providers, discoverable devices, provider readiness, capabilities, and relevant execution constraints. AQLI applies its configured policy and asks Aljabr to initialize the selected provider/runtime.

The following states must not be conflated:

- **Defined:** a capability contract exists.
- **Implemented:** provider code claims to implement it.
- **Installed:** the provider artifact is present.
- **Available:** required host, driver, native library, and device prerequisites are satisfied.
- **Initialized:** provider runtime initialization succeeded.
- **Validated:** correctness and compatibility tests passed for a documented environment and version.

A provider being installed does not imply that hardware is present or usable. A device being discoverable does not imply that every operation, dtype, shape, or model is supported.

### 5. Fallback must be explicit

An accelerator initialization or execution failure must be surfaced with actionable diagnostics. CPU fallback is allowed only when the calling application or runtime policy explicitly permits it.

Aljabr may expose fallback-capable implementations and compatibility information. It must not silently switch a requested accelerator workload to CPU in a way that hides the actual execution backend. If fallback is permitted, execution diagnostics must identify the fallback and its reason.

### 6. Provider configuration is not serving policy

Provider initialization may accept provider-scoped technical configuration, such as native library paths, device identifiers, memory-pool sizing, or debug/validation options.

Such configuration is supplied by the consuming application/runtime. AQLI owns its external configuration schema and policy. Aljabr must not own AQLI YAML/properties files, model-selection policy, serving budgets, or deployment profiles.

### 7. Profiling has two ownership layers

**Aljabr low-level profiling** measures backend mechanisms: kernel latency, dispatch overhead, command submissions, memory allocation/transfer, device utilization where available, and benchmark results for defined workloads.

**AQLI production profiling** measures serving behavior: request latency, TTFT, tokens per second, queueing, concurrency, per-model/session behavior, and operational resource utilization.

Aljabr may expose execution events and measurements needed by consumers. AQLI owns aggregation, retention, dashboards, alerts, and policy decisions based on those measurements. Profiling must be opt-in or explicitly requested where it incurs material overhead.

### 8. No hidden process-global backend policy

Aljabr APIs should support explicit runtime and execution-context ownership. Avoid mutable global APIs such as setGlobalBackend("metal") as the primary selection mechanism.

Applications may choose to share provider runtimes or create multiple contexts. The API must not assume one process-wide backend when different models or workloads may use different providers.

### 9. Dependency direction

- Aljabr core and backend SPIs must not depend on AQLI.
- Backend implementations depend inward on Aljabr contracts.
- AQLI depends on Aljabr APIs and selected provider artifacts.
- Provider-specific SDKs and native libraries remain optional dependencies.
- Model formats and model-family plugins must not own hardware provider selection policy.
- Tafkir and other consumers may use Aljabr backend APIs without introducing an AQLI dependency.

## Consequences

### Positive

- Aljabr remains reusable outside inference serving.
- CPU-only and accelerator-specific distributions can be assembled independently.
- Runtime selection can adapt to actual device readiness and model requirements.
- Low-level kernel measurements remain useful to multiple consumers.
- Serving policy and production observability remain in AQLI.
- Future CUDA, Vulkan, OpenCL, Metal, and other providers can implement common contracts without model-family changes.

### Trade-offs

- Existing overlapping provider/backend/plugin SPIs must be reconciled.
- Provider readiness and capability reporting require explicit semantics and conformance tests.
- AQLI must implement and test selection, fallback, and operational policy.
- Packaging profiles must be validated independently from runtime selection.
- Legacy runner configuration may require staged deprecation rather than immediate removal.

## Initial source audit (2026-11-04)

This is a source-level inventory of the current main branch, not a claim of full runtime validation.

| Existing area | Observed source | Initial assessment |
|---|---|---|
| Generic provider SPI | core/aljabr-spi-core: AljabrProvider, ProviderDescriptor, ProviderContext, CapabilityDescriptor | Useful generic provider lifecycle and metadata foundation. It does not by itself define compute-device discovery or operation-level readiness. |
| Tensor backend contract | core/aljabr-tensor: core/backend/ComputeBackend | Broad tensor operation interface. It has no common provider/device lifecycle or structured capability query in the inspected contract. |
| CPU provider | backend/cpu: CpuBackend | Implements the tensor backend contract; current module build uses the Java plugin and directly depends on tensor. Provider registration/lifecycle integration needs audit. |
| CUDA backend generations | backend/cuda: CUDABackend, CudaComputeBackend, CudaKernelPlugin | Multiple backend/plugin abstractions exist. At least one implementation has a CPU fallback path. Selection and fallback behavior need consolidation and explicit policy boundaries. |
| Metal backend | backend/metal: MetalComputeBackend, MetalKernelPlugin, MetalCapabilities, AppleSiliconDetector | Contains device detection, capability snapshot, optimization manager, runner/configuration classes and FFM bindings. Legacy runner/configuration concepts coexist with reusable backend implementation. |
| GGUF provider SPI | core/aljabr-gguf-api: GgufBackendProvider, GgufBackendSelection | Separate format-specific provider and selection mechanism. It should not be assumed to be the universal compute-provider SPI. |
| Service registration | CUDA and ROCm plugin service descriptors; Java and llama.cpp GGUF provider descriptors | ServiceLoader is already used in several areas, but registration is not yet demonstrated as one unified compute-provider discovery path. |
| Metal documentation | backend/metal/aljabr-backend-metal/README.md | Contains aljabr.runners.metal.* runtime settings and auto/standard/offload/force/disabled selection examples. These are legacy runner-facing settings and conflict with this ownership boundary if treated as canonical public configuration. |
| Distribution guidance | docs/architecture/MODULAR_DISTRIBUTION.md | Already documents optional providers, consumer-selected dependencies, provider readiness states, and the distinction between installed and usable providers. Align implementation with this guidance. |

## Follow-up work

This ADR intentionally does not prescribe a broad refactor. Next implementation work should be small and evidence-driven:

1. Complete a call-site and dependency audit for the existing provider, tensor backend, kernel plugin, and GGUF provider contracts.
2. Determine whether AljabrProvider can be extended for compute providers or whether a narrow compute-specific SPI is required; do not create a duplicate registry.
3. Define the minimum common device/capability/readiness model and provider conformance tests.
4. Audit and classify legacy Metal/CUDA runner configuration and fallback behavior. Deprecate or migrate only after callers and tests are identified.
5. Update backend documentation so provider technical configuration is distinguished from AQLI runtime selection policy.
6. Verify provider-absent, provider-installed/device-missing, provider-ready, unsupported-operation, initialization-failure, and explicitly-permitted-fallback cases.
7. Validate packaging profiles separately from runtime selection.

## Validation requirements

- Unit tests for provider descriptors, capability semantics, discovery, initialization, and release.
- Tests for provider absence and optional native-library absence without breaking core/CPU use.
- Tests proving unsupported operations and initialization failures are explicit.
- Tests proving fallback occurs only when explicitly enabled by the consumer.
- Tests proving multiple provider runtimes/contexts can coexist where supported.
- Clean consumer dependency-graph tests for CPU-only and optional accelerator profiles.
- Hardware-specific correctness and performance validation remains separate for Metal/CUDA execution changes.

## Related architecture documents

- Framework Architecture
- Modular Distribution Architecture
- Unified Aljabr Architecture
