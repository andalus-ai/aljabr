# Tensor Indexing Contract Audit

**Status:** source-level API audit and implementation proposal; no runtime contract is changed by this document  
**Baseline:** `main` after the shared split-validation change  
**Scope:** public `Tensor` API, `ComputeBackend` SPI, `DefaultTensor`, CPU backend, dtype and autograd contracts

## 1. Executive summary

Aljabr currently exposes rectangular slicing and equal-sized splitting, but it does not expose general indexed selection.

The public `Tensor` interface contains `slice(long[] offsets, long[] sizes)` and `split(int axis, int parts)`. `ComputeBackend` has matching backend operations. There is no `gather`, `indexSelect`, `take`, or equivalent indexed-selection method in these contracts. The CPU backend implements slice and split; split delegates to slice.

This means there is no existing gather/index-select validation contract to centralize. The next step should define the operation semantics before adding a new public API or backend method.

## 2. Source findings

### 2.1 Existing indexing surface

| Surface | Existing behavior |
|---|---|
| `Tensor.slice(offsets, sizes)` | Rectangular region described by one offset and size per dimension |
| `Tensor.split(axis, parts)` | Equal-sized chunks along one axis |
| `ComputeBackend.slice` | Backend-owned storage/layout implementation |
| `ComputeBackend.split` | Backend-owned chunk implementation; CPU currently delegates each chunk to slice |
| Gather / index-select | Not present in the public API or backend SPI |
| Scatter / index-add | Not present in the inspected tensor operation surface |

Slice and split are structural range operations. They cannot express arbitrary repeated, reordered, or sparse positions along a dimension.

### 2.2 Dtype and index representation

The current `DType` enum includes `I32`, `I8`, and `INT8`, but no `I64`. The public `Tensor.of` factory currently accepts `float[]`; there is no corresponding public integer-index tensor factory in the inspected API.

The existence of `I32` in the enum is not sufficient evidence that integer tensors can be constructed and consumed consistently across all providers. The CPU slice path currently documents and enforces F32 input support. Index values must not be read through float-oriented helpers or by assuming every backend exposes a host `MemorySegment`.

### 2.3 Layout and ownership

`Tensor` exposes logical shape and an immutable `TensorLayout`; `TensorStorage` is backend-neutral and host access is an optional `HostAccessibleStorage` capability. An indexing implementation must preserve that boundary.

A backend may produce a materialized output or a representable view where semantics permit. It must not require public callers to access CPU buffers or host memory. Output ownership must follow the existing rule: each returned tensor owns one storage reference, and partially created outputs must be released if a later step fails.

### 2.4 Autograd

The operation graph has structural operations such as `SLICE` and `SPLIT`, but no indexed-selection operation. A differentiable index-select needs a defined backward rule: gradients from output positions are accumulated into the corresponding input positions. Duplicate indices therefore require accumulation, not overwrite. This is a correctness requirement, not an optimization detail.

A gather operation has a different backward mapping because each output element can select a different coordinate. It should not be conflated with axis-wise index-select.

## 3. Distinguish the operations

### 3.1 Index-select

Index-select chooses complete slices along one dimension using a one-dimensional index sequence.

For input shape `[d0, ..., d(axis), ..., dn]` and index count `k`, output shape is identical except that the selected axis becomes `k`.

Example: selecting positions `[2, 0, 2]` from axis 1 of a `[2, 4]` tensor produces shape `[2, 3]`; output order follows the index sequence and duplicate positions are preserved.

### 3.2 Gather

Gather selects individual elements using an index tensor whose rank and non-selected dimensions participate in the output shape. It is not equivalent to selecting whole axis slices.

These APIs should remain separate. A single ambiguous `index` method would make shape, validation, and gradient semantics difficult to understand.

## 4. Recommended initial contract: index-select

Implement axis-wise index-select first, as the smaller and more composable primitive.

Recommended semantics:

1. Input rank must be at least one; scalar input is invalid.
2. Axis accepts negative indexing and is normalized once in shared validation.
3. Indices are a one-dimensional integer tensor. Initially support `I32` only if the integer storage path is verified end-to-end; do not reinterpret F32 values as integers.
4. Every index must satisfy `0 <= index < input.shape().dim(axis)`. Do not silently wrap negative indices.
5. Preserve index order and duplicates.
6. An empty index tensor is valid and produces a zero-sized selected axis.
7. A non-empty index tensor cannot select from a zero-sized input axis.
8. Output dtype/device/backend follow the input tensor; indices are metadata/control input and must be compatible with the operation's provider contract.
9. The backend owns the data movement, output allocation, layout, and cleanup. The public API does not expose host-memory assumptions.
10. During autograd, backward performs scatter-add into the input gradient. Repeated indices accumulate contributions.

The initial implementation should reject unsupported index dtypes and backend combinations explicitly. It must not silently copy indices through a host-only path when the selected backend cannot support that transfer.

## 5. Shared validation plan

When the API is approved, add a pure shape/metadata validator in `TensorOperationValidator`. It should produce an immutable plan containing:

- normalized axis;
- input shape and output shape;
- selected-axis size;
- index count;
- validated index dtype;
- any backend-independent constraints.

Index-value bounds validation belongs in a layer that can read the index representation correctly. Do not put host segment access into the shared shape validator.

The CPU implementation should use a typed integer read path for I32 indices, support non-contiguous input layouts only when it can correctly honor their strides, and create an independently-owned output. GPU/device backends may implement native indexed selection or explicitly report unsupported capability.

## 6. Incremental implementation sequence

1. **Index representation audit:** define typed integer tensor construction, storage encoding, and value access without adding an implicit F32 conversion.
2. **Index-select contract:** add API documentation and pure shared validation tests for axis normalization, scalar rejection, output shape, empty indices, and bounds.
3. **CPU reference implementation:** implement I32 index-select with layout-aware reads, duplicate/order tests, and failure cleanup tests.
4. **Autograd:** add a dedicated graph operation and scatter-add backward tests, especially duplicate indices.
5. **Provider conformance:** define optional capability support and tests for CPU, Metal, CUDA, HAT, and other providers. Unsupported providers must fail explicitly.
6. **Gather:** design and implement separately after index-select is stable.

Do not require every provider to implement indexed selection in the first API release. The backend SPI should have an explicit unsupported default until a provider has a correct implementation.

## 7. Non-goals

- This audit does not add a public method or backend SPI signature.
- It does not assume I32 support is complete merely because the enum value exists.
- It does not define gather as an alias for index-select.
- It does not make host-readable storage mandatory for device tensors.
- It does not promise zero-copy indexed selection.
- It does not silently add negative-index wrapping semantics.
- It does not implement scatter or scatter-add as a separate public operation.

## 8. Acceptance criteria for implementation

Before index-select is considered complete:

- Public API and backend contract agree on axis and index semantics.
- Shared validation is independent of storage access and execution.
- CPU correctness tests cover rank, axis, bounds, order, duplicates, empty indices, and non-contiguous layouts.
- Autograd tests prove duplicate-index gradients accumulate.
- Output and temporary resource ownership is correct on success and partial failure.
- Unsupported providers fail explicitly and do not silently transfer device data to the host.
- CI and provider conformance results are reported separately from declared capability metadata.
