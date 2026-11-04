# Phase 4A.3a — CUDA/ROCm Provider Contract Audit

**Status:** Audit complete; legacy plugin modules remain excluded  
**Date:** 2026-11-04  
**Depends on:** ADR-0001, Phase 4A.1 audit, Phase 4A.2 comparison, Phase 4A.3 canonical provider/session contract

## Executive decision

Do **not** migrate the excluded CUDA/ROCm plugin modules to `AljabrProvider` in place, and do not merely rename their ServiceLoader descriptors.

The repository contains two distinct generations of backend code:

1. The active CUDA backend path uses the canonical tensor API in `tech.kayys.aljabr.core.backend.ComputeBackend`.
2. The excluded legacy CUDA/ROCm plugin paths still refer to the removed/nonexistent `tech.kayys.aljabr.spi.tensor` API and publish ServiceLoader descriptors for that missing type.

The legacy plugin classes also implement the separate `KernelPlugin` API. They are not `AljabrProvider` implementations and do not currently expose the provider/device/session lifecycle defined by Phase 4A.3.

The correct action is to keep these legacy modules excluded, preserve their sources for a later controlled migration or retirement, and prevent them from being mistaken for supported providers. A future migration must be a dedicated PR after the provider/session API exists and must demonstrate feature parity. This audit makes no production source or build changes.

## 1. Canonical contracts present on main

| Concern | Current canonical location | Finding |
|---|---|---|
| Provider lifecycle | `core/aljabr-spi-core/.../spi/provider/AljabrProvider.java` | Exists; supports descriptor, initialize, close |
| Provider metadata/context | `spi/provider/ProviderDescriptor.java`, `ProviderContext.java` | Existing baseline |
| Tensor execution API | `core/aljabr-tensor/.../core/backend/ComputeBackend.java` | Canonical tensor-operation contract |
| CPU tensor backend | `backend/cpu/aljabr-backend-cpu` | Separate CPU implementation |
| CUDA tensor backend | `backend/cuda/aljabr-backend-cuda` | Active module; imports `core.backend.ComputeBackend` |
| CUDA FFM bindings | `backend/cuda/aljabr-kernel-cuda` | Active module; low-level binding/detection/stream utilities, not a provider |
| ROCm FFM bindings | `backend/rocm/aljabr-kernel-rocm` | Source exists, but project is excluded from default settings |

A recursive main-branch tree inspection found **no source path** for `tech.kayys.aljabr.spi.tensor`. The existence of the new `spi.provider` package does not make the old `spi.tensor` service type valid.

## 2. CUDA module inventory

### Active: `backend/cuda/aljabr-backend-cuda`

- Included by `settings.gradle.kts` as `:backend:cuda:aljabr-backend-cuda`.
- Implements `tech.kayys.aljabr.core.backend.ComputeBackend`.
- Depends on `:core:aljabr-tensor`, `:backend:cpu:aljabr-backend-cpu`, and `:backend:cuda:aljabr-kernel-cuda`.
- It is a tensor backend implementation, **not** an `AljabrProvider`.
- No provider ServiceLoader registration is established by the inspected source/build configuration.
- Its current CPU delegation and native initialization behavior are explicitly outside this audit's change scope. Do not alter them as part of this contract cleanup.

### Active: `backend/cuda/aljabr-kernel-cuda`

- Included by `settings.gradle.kts`.
- Its Gradle source set deliberately compiles the selected binding, detector, memory-pool, accelerator, stream-manager, and runner-mode classes.
- It does not provide a provider lifecycle implementation.
- Keep this low-level FFM module distinct from provider discovery and tensor backend selection.

### Excluded legacy module: `backend/cuda/aljabr-plugin-kernel-cuda`

- Its settings include is commented out.
- `CudaComputeBackend.java` imports `tech.kayys.aljabr.spi.tensor.ComputeBackend` and `tech.kayys.aljabr.spi.tensor.CpuBackend`; neither package exists in the current main-branch source tree.
- Its service descriptor is `META-INF/services/tech.kayys.aljabr.spi.tensor.ComputeBackend`, which names the absent service type and registers `tech.kayys.aljabr.cuda.CudaComputeBackend`.
- The module's `build.gradle.kts` comments out the old kernel-plugin-core dependency, while `CudaKernelPlugin.java` imports `tech.kayys.aljabr.plugin.kernel.*`.
- The module is therefore not a valid implementation of the current tensor API or the new provider SPI.

**Decision:** Do not update its service descriptor to `AljabrProvider`: neither `CudaComputeBackend` nor `CudaKernelPlugin` implements that interface. Do not re-enable the project as part of provider discovery.

## 3. ROCm module inventory

### Excluded low-level module: `backend/rocm/aljabr-kernel-rocm`

- Its settings include is commented out.
- It contains FFM/HIP binding and runner scaffolding.
- Its Gradle file depends on `:core:aljabr-tensor`, but the inspected implementation is not a complete provider/session adapter.
- Its source-level comments explicitly describe parts of the HIP execution path as TODO/scaffolding.

### Excluded legacy module: `backend/rocm/aljabr-plugin-kernel-rocm`

- Its settings include is commented out.
- `RocmComputeBackend.java` imports the absent `tech.kayys.aljabr.spi.tensor.ComputeBackend` and `CpuBackend` types.
- Its service descriptor is `META-INF/services/tech.kayys.aljabr.spi.tensor.ComputeBackend`, registering `tech.kayys.aljabr.rocm.RocmComputeBackend`.
- `RocmKernelPlugin.java` implements the separate legacy `KernelPlugin` contract.
- Its Gradle file depends on `:core:plugin:aljabr-plugin-kernel-core`; that project is not included by the inspected root settings.
- The backend implementation hardcodes `isNative = false` and delegates to CPU. This is scaffolding, not evidence of a functioning ROCm provider.

**Decision:** Keep excluded. Do not register it as an Aljabr provider and do not advertise it as a supported ROCm tensor backend.

## 4. Service descriptor findings

| Descriptor | Current status | Required treatment |
|---|---|---|
| CUDA legacy `spi.tensor.ComputeBackend` | Invalid/stale; module excluded; service type absent | Remove only when legacy module is retired or migrated; never re-label without implementing the target interface |
| ROCm legacy `spi.tensor.ComputeBackend` | Invalid/stale; module excluded; service type absent | Same |
| `spi.provider.AljabrProvider` | Canonical provider service type exists | New provider implementations should register here after they implement the full lifecycle contract |

A service descriptor is part of a runtime contract. Renaming the descriptor without making the implementation assignable to its service type creates a broken ServiceLoader registration. The future provider artifact must contain an implementation of `AljabrProvider`, not just a backend class or `KernelPlugin`.

## 5. Build and validation status

This is a source/build-graph audit based on the main-branch tree, root `settings.gradle.kts`, module build files, service descriptor contents, and implementation imports.

| Check | Result |
|---|---|
| Confirm canonical tensor API package | Confirmed: `core.backend.ComputeBackend` |
| Confirm old `spi.tensor` source package | No matching source path in main tree |
| Confirm active CUDA project inclusion | Confirmed |
| Confirm legacy CUDA plugin inclusion | Excluded/commented out |
| Confirm legacy ROCm project inclusion | Both projects excluded/commented out |
| Confirm legacy service descriptor targets | Both target missing `spi.tensor.ComputeBackend` |
| Compile probe of excluded modules | Not run; no repository checkout/Gradle execution was available in this audit |
| Hardware validation | Not applicable; no runtime behavior changed |

Do not interpret the source audit as a successful compile. Before any future re-enablement, run targeted Gradle compile tasks in a real checkout after restoring the required module dependencies and API contracts.

## 6. Safe next steps

1. Keep current root project inclusion unchanged.
2. Keep the active CUDA tensor backend and low-level CUDA FFM module unchanged.
3. Do not add CUDA/ROCm legacy modules to the default build.
4. Implement provider discovery against `AljabrProvider` independently of these legacy classes.
5. Later add a CUDA provider adapter that composes the active CUDA backend with a provider-owned device/session lifecycle. It must report operation support and CPU delegation honestly.
6. Later add a ROCm provider only after HIP binding and execution are real, independently testable, and hardware-validated.
7. Retire stale service descriptors together with their legacy modules, or migrate them only in the same PR that adds genuine `AljabrProvider` implementations and passing ServiceLoader tests.

## Non-goals

- No changes to CUDA/Metal kernels, GGUF execution, tensor numerical behavior, CPU fallback defaults, native library loading, or AQLI.
- No root settings/build inclusion changes.
- No deletion of legacy CUDA/ROCm source files.
- No claim of CUDA/ROCm provider support based solely on source presence.
