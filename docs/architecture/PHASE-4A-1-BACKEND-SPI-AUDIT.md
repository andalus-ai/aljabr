# Phase 4A.1 — Backend Provider and SPI Audit

**Status:** Audit baseline  
**Date:** 2026-11-04  
**Audited ref:** `main` (tree `3bee521a1c630e271b5606ceab7bed0b9d473d5c`)  
**Related decision:** [ADR-0001 — Backend Ownership](ADR-0001-BACKEND-OWNERSHIP.md)

## Purpose

Record the current provider/backend/plugin contract boundaries and build participation before proposing API consolidation. This is an inventory and risk assessment, not an implementation change. No source behavior is changed by this audit.

## Scope inspected

- `core/aljabr-spi-core`: provider metadata and lifecycle contracts.
- `core/aljabr-tensor`: tensor `ComputeBackend` contract.
- CPU, CUDA, Metal, ROCm and Blackwell backend/plugin source trees.
- CUDA and ROCm `META-INF/services` declarations.
- Root `settings.gradle.kts` and selected backend module build files.
- GGUF provider service descriptors, to distinguish format-specific backend selection from compute-provider discovery.

## Current contract inventory

| Contract | Current responsibility | Observed status |
|---|---|---|
| `spi.provider.AljabrProvider` | Generic provider descriptor, initialization and close lifecycle | Exists in `aljabr-spi-core`; no `META-INF/services/...AljabrProvider` registration was found in the repository tree |
| `spi.provider.ProviderDescriptor` / `CapabilityDescriptor` | Provider identity, API compatibility and declared capability metadata | Metadata only; no device inventory, operation readiness, or initialization state model |
| `spi.provider.ProviderContext` | Provider-scoped string configuration | Exists; no typed device selection or runtime ownership model |
| `core.backend.ComputeBackend` | Tensor operation surface over `Tensor` | Used by the included CPU and Metal tensor backend implementations |
| `spi.tensor.ComputeBackend` | Referenced by CUDA/ROCm kernel compute adapters and their service descriptors | No matching source interface was found in the audited `main` tree |
| `KernelPlugin` | Runner/kernel lifecycle, platform metadata, supported operations, health and execution | Separate plugin abstraction; its inspected source is under the DirectML plugin module |
| `GgufBackendProvider` | GGUF-format-specific backend selection | Separate format SPI; must not be treated as the universal compute-provider contract |

## Findings

### F1 — Two tensor backend type names do not converge

The core tensor API defines `tech.kayys.aljabr.core.backend.ComputeBackend`. CUDA and ROCm adapter sources import `tech.kayys.aljabr.spi.tensor.ComputeBackend`, and their service descriptors name that latter type. No `spi/tensor/ComputeBackend.java` source exists in the audited tree.

**Impact:** The CUDA/ROCm service descriptors do not currently establish a usable, repository-local ServiceLoader contract. The two tensor backend APIs must be reconciled before provider discovery can be standardized.

**Priority:** P0 for CUDA/ROCm provider integration; verify with a focused build/compile probe before changing code.

### F2 — Generic provider lifecycle is not wired to concrete compute providers

`AljabrProvider` supplies `descriptor()`, `initialize(ProviderContext)` and `close()`. The inspected `CpuBackend`, `MetalComputeBackend`, `CUDABackend` and CUDA/ROCm adapter classes implement tensor-operation contracts, not `AljabrProvider`.

No service descriptor for `tech.kayys.aljabr.spi.provider.AljabrProvider` was found.

**Impact:** The generic provider SPI is currently a standalone foundation, not an operational compute-provider discovery path. Avoid claiming that installed compute backends are discoverable through it until an implementation and registration path exist.

**Priority:** P1; first settle the canonical tensor/backend contract and provider composition model.

### F3 — KernelPlugin is a third, distinct abstraction

The inspected `KernelPlugin` contract describes platform identity, availability, architecture/version/operation support, execution, and metadata. CUDA and Metal plugin classes wrap `CudaRunner` and `MetalRunner` and implement validation, initialization, and health behavior.

The inspected `KernelPlugin` source is located in the DirectML plugin module. The CUDA plugin module's Gradle dependency on the former kernel-core project is commented out, and the CUDA plugin module is not included in the root settings. Metal's `MetalKernelPlugin` is not included by the Metal module's explicit main-source include list.

**Impact:** The plugin lifecycle and tensor compute backend lifecycle overlap in platform detection and initialization but represent different execution abstractions. Their build participation is also inconsistent. They should not be merged blindly.

**Priority:** P1; determine whether KernelPlugin is a legacy inference-runner plugin, a kernel dispatch plugin, or an intended compute-provider adapter.

### F4 — Root build excludes several backend/plugin modules

At the audited ref, root `settings.gradle.kts` includes CPU, CUDA's older backend and kernel modules, and the Metal backend. It comments out CUDA kernel plugin, DirectML plugin, ROCm kernel/plugin modules, and Blackwell kernel module. The service descriptors in the excluded CUDA/ROCm plugin modules therefore do not participate in the default root build.

**Impact:** Source presence and ServiceLoader resource presence must not be confused with a buildable, packaged, or installed provider.

**Priority:** P1; document supported build profiles and the actual distribution matrix before declaring backend availability.

### F5 — CPU fallback is embedded inside accelerator implementations

The inspected CUDA tensor backend and CUDA adapter delegate many operations to CPU and fall back on native initialization or operation failure. The Metal backend similarly delegates many operations to CPU and falls back when Metal is inactive or a native operation returns an error. The ROCm adapter currently has `isNative = false` and routes operations to CPU. Blackwell contains standalone CPU fallback kernel routines.

**Impact:** An accelerator backend may report itself as selected while individual operations execute on CPU. This conflicts with an explicit, observable fallback policy at the runtime boundary unless the behavior is intentionally classified as per-operation implementation fallback.

**Priority:** P0 for policy clarity; do not remove fallback code in this audit. First distinguish (a) unsupported-op implementation delegation, (b) initialization fallback, and (c) runtime failure fallback.

### F6 — Device detection and capability reporting are backend-specific

Metal has `AppleSiliconDetector` and `MetalCapabilities`; CUDA has `CudaDetector` and `CudaCapabilities`; Blackwell has its own detector/capabilities. The generic `CapabilityDescriptor` is a provider-declared capability identifier/version/stability tuple and does not normalize device identity, memory, driver/runtime versions, operation support, or readiness.

**Impact:** A common provider API will need to model normalized device/capability facts without erasing vendor-specific details. Detection must remain distinct from successful initialization and operation validation.

**Priority:** P1; define semantics and state transitions before implementing a shared registry.

### F7 — GGUF backend selection is format-specific

The repository has ServiceLoader registrations for `GgufBackendProvider` in the Java and llama.cpp GGUF modules. This is a format execution/loader selection concern, not proof of a universal tensor compute provider registry.

**Impact:** Preserve the GGUF SPI boundary. It may consume a compute backend, but it should not become the owner of device discovery or global backend policy.

**Priority:** P2; retain and document the boundary.

## Build and registration observations

| Area | Main-build participation at audited ref | Registration observation |
|---|---|---|
| CPU tensor backend | Included | No generic `AljabrProvider` registration found |
| CUDA backend | Included | Older tensor backend module; separate CUDA plugin module is excluded |
| CUDA kernel plugin | Excluded | Service descriptor targets missing `spi.tensor.ComputeBackend` type |
| Metal tensor backend | Included | Uses core tensor `ComputeBackend`; Metal runner plugin class is excluded from main source set |
| ROCm plugin/backend | Excluded | Service descriptor targets missing `spi.tensor.ComputeBackend` type |
| Blackwell kernel | Excluded | Has backend-specific detection and CPU fallback routines |
| GGUF Java / llama.cpp | Included | Format-specific `GgufBackendProvider` descriptors |

## Recommended consolidation direction

Do not introduce a new interface in this step. The evidence supports this staged direction:

1. **Canonical operation contract:** retain `core.backend.ComputeBackend` as the current tensor-operation API pending a compatibility review. Treat `spi.tensor.ComputeBackend` references as stale/unresolved until proven otherwise.
2. **Provider adapter:** define how a concrete compute provider owns one or more `ComputeBackend` instances, devices, and lifecycle resources through the generic provider SPI. Avoid making tensor operations themselves responsible for device discovery.
3. **Kernel plugin disposition:** decide whether `KernelPlugin` is retained for model-runner/inference plugins, deprecated, or adapted behind a compute provider. Do not conflate it with the tensor backend contract.
4. **Discovery and readiness:** distinguish declared, implemented, packaged, discovered, available, initialized, and validated states. Device discovery alone must not imply usable execution.
5. **Fallback:** preserve CPU implementations but expose fallback as an explicit, observable policy. Separate unsupported-operation delegation from provider failure fallback.
6. **Build profiles:** define which backend modules are supported in default builds, optional builds, and published distributions. Service descriptors in excluded modules are not active registrations.

## Follow-up tasks

- [ ] Run focused Gradle compile/test probes for the included CPU/CUDA/Metal modules and determine whether any source-set or dependency configuration masks unresolved contracts.
- [ ] Trace all construction and selection call sites for `ComputeBackend`, `KernelPlugin`, and GGUF providers.
- [ ] Audit `ServiceLoader` and custom plugin registries in the repository, including resource descriptors in excluded modules.
- [ ] Inventory operation-level native coverage versus CPU delegation for CUDA and Metal.
- [ ] Specify provider/device/readiness data model and conformance test requirements.
- [ ] Produce the Phase 4A.2 comparison and API consolidation proposal before changing public APIs.

## Non-goals

- No backend implementation was deleted or rewritten.
- No CPU fallback behavior was changed.
- No public SPI was added or renamed.
- No macOS/Metal hardware validation was performed or required for this source/build-configuration audit.
