# Tensor View and Aliasing Contract Audit

**Status:** implementation audit; no runtime contract is changed by this document  
**Scope:** eager and lazy tensor shape operations, storage ownership, and the current CPU backend  
**Baseline:** `main` at the time of this audit

## Executive summary

Aljabr already uses reference-counted storage views, but view behavior is currently an implementation detail of individual backends rather than a discoverable `Tensor` API contract.

The current CPU backend has four materially different behaviors:

| Operation | Current CPU behavior | Storage relationship |
|---|---|---|
| `reshape` | Validates element count, constructs a new tensor | Shares the same storage and retains one reference |
| `slice` | Creates a view for a contiguous rectangular region; copies a non-contiguous region | Conditional alias |
| `split` | Repeatedly delegates to `slice` | Conditional alias for each output |
| `transpose` | Allocates and copies elements | Independent storage |
| `flatten` | Delegates to `reshape` | Shares storage |
| `unsqueeze` / `squeeze` | Delegate to `reshape` | Shares storage |

Lazy tensors record shape operations and defer execution. Lazy `split` is explicitly unsupported. Consequently, the eager CPU behavior is not a universal guarantee for every backend or execution mode.

## Implementation findings

### 1. Storage ownership is separate from tensor aliasing

`DefaultTensor` owns one reference to its `TensorStorage`. Its constructor takes ownership of an existing reference, and `release()` releases that reference once.

The CPU `reshape` implementation calls `retain()` before constructing the result. The new tensor therefore owns an independent reference to the same storage. Releasing either tensor does not invalidate the other while its reference remains live.

The CPU `slice` implementation uses `CpuBuffer.view` for legacy buffers or `HostAccessibleStorageView.view` for neutral host storage. Both view paths retain their parent storage and release that reference when the view's final reference is released.

This ownership model is a sound foundation for independently-lived views. It does not, by itself, describe which operations alias.

### 2. Slice is conditionally zero-copy

The CPU slice implementation validates rank, bounds, and F32 support, computes row-major input strides, and determines whether the requested rectangular region can be represented as one contiguous byte range.

- Contiguous regions become reference-counted storage views.
- Non-contiguous regions are copied element-by-element into a newly allocated output.
- A split is implemented as a series of slices, so splitting along the leading dimension of a row-major tensor can produce views, while splitting along an inner dimension generally materializes copies.

This is a reasonable performance strategy, but callers cannot discover whether a particular result aliases its source through the current public API.

### 3. Transpose is currently a materializing operation

The CPU transpose implementation computes row-major input/output strides, allocates an output buffer, and writes the transposed values into it. It does not create a strided view.

There is no public stride or storage-offset metadata on `Tensor`. A zero-copy transpose would therefore require a broader layout contract rather than only changing the implementation to share storage.

### 4. Reshape assumes the current storage is reshape-compatible

The CPU reshape checks only that the element count is unchanged, then shares the source storage. This is correct for the current contiguous row-major representation. The tensor abstraction does not yet expose strides or a contiguity predicate, so it cannot express whether an arbitrary future non-contiguous tensor can be reshaped without copying.

Do not extend this implementation to arbitrary strided tensors without first defining reshape compatibility and materialization behavior.

### 5. Lazy behavior is not yet aligned with eager tuple operations

Lazy `reshape` and `slice` record graph nodes. Evaluation eventually delegates to the execution backend. Lazy `split` currently throws `UnsupportedOperationException`.

The lazy path therefore cannot yet promise the same complete shape-operation surface as eager CPU tensors. The aliasing contract should describe evaluated results and should not imply that graph recording itself creates storage views.

### 6. Existing tests are incomplete for the public aliasing contract

The current test suite includes a tensor-view lifetime test for a legacy `CpuBuffer.view`, along with storage ownership conformance tests. These establish important lifetime properties, but do not establish a complete cross-operation matrix for:
- storage identity / shared backing allocation;
- mutation visibility through aliases where mutable host access is available;
- view lifetime after releasing the source tensor;
- source lifetime after releasing a view;
- contiguous versus non-contiguous slice behavior;
- split aliasing by axis;
- eager versus lazy evaluated behavior;
- backend differences.

## Contract decisions recommended before implementation

### A. Distinguish ownership from aliasing

Keep the existing explicit ownership model:
- every tensor owns one storage reference;
- every independently-lived view owns its own storage reference;
- borrowed accessors do not retain or transfer ownership;
- releasing a tensor releases only the reference it owns.

Document aliasing separately. Two tensors may own independent references while observing the same underlying elements.

### B. Define current operations honestly

For the current CPU backend, document these behaviors as implementation facts, not cross-backend guarantees:
- `reshape`, `flatten`, `squeeze`, and `unsqueeze` share storage;
- `slice` and `split` may share storage or materialize, depending on contiguity;
- `transpose` materializes.

Until a backend-neutral contract exists, callers should not rely on storage identity or mutation visibility across these operations.

### C. Introduce layout metadata before general strided views

A future layout model should make at least the following concepts explicit:
- logical shape;
- element strides;
- storage offset;
- element type / byte width;
- contiguity (or a well-defined layout predicate);
- storage ownership independent of the logical view.

A view should be a logical tensor layout over a retained storage reference, not a copied allocation disguised as a view.

### D. Make copy behavior explicit

The eventual API should distinguish intent:
- a view operation that either returns a representable alias or fails;
- a materialization / contiguous operation that guarantees independent contiguous storage;
- convenience operations such as `reshape` that may select a view or copy according to documented compatibility rules.

Avoid promising that every reshape, slice, or transpose is zero-copy.

### E. Preserve backend-neutrality

Do not expose `CpuBuffer`, `MemorySegment`, or host-only assumptions in the public tensor view contract. Device storage may not be host-addressable. Layout metadata and view creation belong at the tensor/storage abstraction boundary, while each backend decides whether it can execute a view directly or must materialize.

## Recommended incremental implementation sequence

1. **Contract tests:** add backend-focused tests for alias identity, retained lifetime, copy independence, and eager/lazy parity where supported.
2. **Public documentation:** document the current CPU behavior without claiming it is universal.
3. **Layout value type:** introduce an immutable, validated layout descriptor (shape, strides, offset, and contiguity) only after agreeing on overflow, zero-sized dimensions, and dtype units.
4. **Storage-backed tensor views:** add a single internal view-construction path that retains storage and validates the view's reachable byte range.
5. **Backend conformance:** require each backend to declare or test support for view execution and materialization.
6. **API evolution:** only then decide whether to add explicit `view`, `contiguous`, `clone`, or alias-inspection methods to the public SDK.

## Non-goals

- No public API signatures are changed by this audit.
- No backend implementation is changed.
- No blanket promise of zero-copy transpose or arbitrary reshape is made.
- No assumption is made that device storage can be accessed through a host `MemorySegment`.
- No change is made to autograd semantics for views; view-gradient propagation needs its own explicit contract and tests.

## Exit criteria

The view/aliasing contract is ready for implementation when:
- ownership and aliasing are specified independently;
- shape, strides, and storage-offset units are unambiguous;
- each operation has defined view/copy/error behavior;
- lifetime and mutation visibility are tested;
- eager and lazy paths agree on supported semantics;
- backend capability and fallback behavior are explicit.
