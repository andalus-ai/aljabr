# CPU Backend Responsibility Boundaries

## Purpose

Reduce the responsibility concentration in `CpuBackend` without changing the public `ComputeBackend` contract, tensor semantics, numerical behavior, or backend selection. This is an incremental refactor, not a rewrite and not a requirement to create one class per operation.

## Current inventory

The current `CpuBackend` is approximately 2,274 lines. Its responsibilities currently include:

| Responsibility | Current location / evidence | Intended boundary |
|---|---|---|
| Backend lifecycle and storage allocation | `close`, `allocateStorage`, `allocate`, output allocation and tensor wrapping near the beginning of `CpuBackend` | Keep backend-owned lifecycle and allocation orchestration in the facade; keep storage capability access in the existing storage helper |
| Tensor storage and logical layout access | Existing `CpuTensorStorageAccess` and `CpuTensorLayoutAccess` | Reuse these as the canonical host-storage and logical-indexing boundaries |
| Pointwise kernels | Existing `ops/CpuOps` plus `CpuTensorLayoutAccess` | Keep SIMD/scalar kernels separate from tensor-level validation and output ownership |
| Matrix multiplication | `ops/MatmulCpu` and `CpuBackend.matmul` | Keep shape/contract checks and output lifecycle separate from the numerical kernel |
| Indexing and shape transforms | `reshape`, `slice`, `split`, `indexSelect`, `indexAdd`, `flatten`, `unsqueeze`, `squeeze`, `transpose`, `cat` | Extract only when a cohesive operation group can own validation, layout semantics and execution without duplicating contracts |
| Activations and normalization | Pointwise methods, `layerNorm`, `rmsNorm`, `batchNorm`; existing `ops/NormOps` | Reuse existing kernels; avoid a second normalization implementation |
| Reductions and losses | `sum`, `mean`, `max`, `reduceAlongDim`, `crossEntropy`, `binaryCrossEntropy` | A cohesive reduction/loss execution component, only where it reduces coupling |
| Attention | `attention`, `attentionGqa`, mask validation, parallelization policy and group computation; existing attention kernel helpers | First substantial extraction candidate: one attention execution component shared by standard and grouped-query attention |
| Spatial operations | `conv2d`, `upsampleNearest2d`, `maxPool2d`, `adaptiveAvgPool2d`, `embedding` | Separate spatial/embedding execution only after confirming shared dependencies and tests |

The repository already contains useful boundaries (`CpuTensorStorageAccess`, `CpuTensorLayoutAccess`, `ops/CpuOps`, `ops/MatmulCpu`, `ops/NormOps`, attention helpers and dedicated tests). Do not recreate these responsibilities under new names.

## Target shape

- **`CpuBackend`** remains the `ComputeBackend` implementation and a thin composition/facade point. It owns backend identity, lifecycle, capabilities, output-storage allocation and delegation.
- **Operation components** own cohesive tensor-level validation and execution for a related operation family. They receive explicit dependencies; they do not create hidden global state or select another backend.
- **Kernel classes** own numerical loops and SIMD/scalar dispatch. They do not own public API policy, model/runtime policy or tensor allocation lifecycle.
- **Storage/layout helpers** remain the canonical path for host-accessible storage and logical tensor addressing. Do not bypass them with ad hoc casts or duplicate offset calculations.

Do not introduce interfaces for every component. Prefer package-private final classes and constructor-injected collaborators unless there is a demonstrated second implementation, stable extension point or valuable test seam.

## Incremental sequence

1. **Inventory and baseline (this change):** map method families, existing helpers, tests and lifecycle dependencies. Do not mix numerical optimization with structural extraction.
2. **Attention extraction:** move standard attention, GQA, mask validation, work partitioning and group computation together. Preserve the current accumulation precision, causal/mask behavior, parallelization threshold, failure cleanup and output ownership. Keep the facade methods as delegators.
3. **Spatial extraction:** evaluate the spatial operations as a cohesive group; preserve layout and padding semantics.
4. **Reduction/loss extraction:** separate only after confirming shared validation and allocation behavior.
5. **Indexing/shape extraction:** preserve view-versus-copy semantics, storage offsets, strides and index validation.
6. **Facade review:** remove only dependencies made obsolete by extraction; do not leave forwarding methods or duplicate kernels that create competing paths.

Each extraction should be a small PR with focused tests and no unrelated performance changes.

## Required invariants

- Preserve the public `ComputeBackend` API and documented tensor semantics.
- Preserve storage ownership, scope/lifetime, release-on-failure and output wrapping behavior.
- Preserve logical indexing for contiguous, strided and offset tensors.
- Preserve dtype, shape, broadcasting, error and numerical-precision behavior.
- Keep backend-specific code inside the backend module; do not leak CPU implementation types into public tensor APIs.
- Keep operation diagnostics and failures attributable to the operation and relevant tensor metadata without exposing tensor contents.
- Test the extracted component directly where useful and retain backend contract tests as end-to-end coverage.
- No extraction may silently turn unsupported inputs into success or change fallback behavior.

## Exit criteria

The refactor is complete when `CpuBackend` primarily coordinates backend lifecycle, capabilities, storage and operation delegation; each extracted component has a cohesive reason to change; there is one canonical implementation per operation; and the existing CPU backend contract and numerical tests remain green.
