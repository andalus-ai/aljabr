# Tensor API and Memory Ownership Contract Audit

> **Status:** Source-level contract audit; recommendations only  
> **Baseline:** `main` at the time of review  
> **Scope:** `plugins/ml/aljabr-tensor`, core memory contracts and the tensor-facing `ComputeBackend` SPI  
> **Compatibility rule:** Do not change public signatures until implementations, downstream consumers and published artifacts have been mapped.

## 1. Executive summary

Aljabr has made meaningful progress toward a backend-neutral tensor and storage design. The current code already separates `TensorStorage` from host memory, uses explicit retain/release ownership, provides FFM-backed storage and reference-counted views, and keeps device-only storage possible. `DefaultTensor` also owns its storage reference and releases its owned gradient.

The main gaps are not a missing tensor abstraction. They are **contract consistency, metadata completeness, lifecycle safety at API boundaries, and cross-implementation semantics**.

The recommended direction is to preserve the current storage migration and strengthen it incrementally. In particular:

1. Make tensor metadata describe logical indexing/layout independently from allocation capacity.
2. Make ownership transfer and borrowed access impossible to misunderstand.
3. Separate compute/storage representation from numerical compute dtype, especially for quantized formats.
4. Decouple the high-level tensor API from the full backend implementation interface without removing advanced escape hatches.
5. Define one behavior matrix for eager, lazy, host and device implementations.
6. Turn memory and numerical semantics into conformance tests.

This document is not a proposal to immediately replace `Tensor`, `DType` or `ComputeBackend`.

## 2. Current implementation evidence

### 2.1 Tensor facade

`Tensor` currently combines:
- Logical metadata: `shape()`, `dtype()`, `device()`.
- Execution/provider access: `backend()`.
- Optional physical storage access: `storage()`.
- Arithmetic, matrix, reduction, activation, normalization, convolution, pooling, attention and embedding operations.
- Autograd controls and gradient ownership.
- Eager/lazy materialization: `eval()`.
- Resource lifecycle: `release()` / `close()`.
- Factory methods and convenience exports.

This is convenient for current model implementations, but it makes the interface broad and combines several capability domains. Not every implementation necessarily supports every operation. The interface uses default unsupported behavior for selected methods, while other methods are mandatory.

**Recommendation:** retain the facade for compatibility now. Over time, define discoverable capability interfaces (for example differentiable, host-readable, strided, sparse, quantized, asynchronous/materializable) where they reduce implementation burden. Do not split every method into a separate interface pre-emptively. A capability query must be stable and must not require callers to inspect implementation class names.

### 2.2 Shape

`Shape` is immutable in practice: constructor input and returned dimensions are cloned; element-count multiplication is checked; negative dimensions are rejected; zero dimensions are accepted. Rank-zero shape currently represents a scalar with one element.

**Recommendations:**
- Preserve checked arithmetic and defensive copying.
- Document rank-zero scalar and zero-sized dimension semantics explicitly.
- Add canonical axis normalization and tests for negative axes at the public contract boundary.
- Avoid treating `Shape` as layout. Equal shapes can have different strides, offsets and physical storage requirements.
- Keep shape equality purely logical; do not fold device, dtype or layout into it.

### 2.3 DType and quantized storage

The current `DType` enum includes F32/F16/BF16, I32/I8/INT8 and block formats Q8_0/Q4_K/Q4_0. It carries block size and bytes per block and calculates storage footprint with checked multiplication.

Review findings:
- `I8` and `INT8` currently have identical block metadata and appear semantically duplicated.
- Common scientific types such as F64 and I64 are not represented in this enum.
- Quantized block encodings are represented in the same enum as ordinary numeric types.
- `memoryFootprintBytes` requires element counts to be divisible by block size. This is reasonable for some packed representations, but makes logical tensor shape and physical encoding constraints inseparable.
- A storage format and an arithmetic/compute dtype are related but not identical concepts.

**Recommendation:** design a compatibility-safe type model that distinguishes:
1. **Logical element type** (what values the tensor represents).
2. **Compute type** (precision used by operations/accumulation).
3. **Storage encoding** (primitive or block-quantized representation, including block metadata).
4. **Quantization parameters** (scales, zero points, grouping, axis and encoding-specific metadata).

Do not simply add enum constants before defining promotion, conversion, arithmetic and serialization semantics. Maintain adapters for existing GGUF/Safetensors format enums until the canonical model is established.

### 2.4 TensorStorage and ownership

`TensorStorage` is deliberately small: capacity in bytes plus `retain()` and `release()`. Its documentation states that each successful retain creates one ownership reference and each owner must balance it. It does not assume heap, host, FFM or device-local memory.

This is a strong foundation. However, the interface does not itself express whether a caller has borrowed a reference or owns one. The ownership protocol currently depends on documentation and disciplined call sites.

`DefaultTensor` takes ownership of one existing storage reference after successful constructor validation. Its `release()` is guarded by an atomic flag and releases its owned gradient and storage. `storage()` returns the underlying storage object directly, but the returned reference is not explicitly documented as borrowed in the public `Tensor` contract.

**Recommendations:**
- State explicitly that `Tensor.storage()` returns a borrowed view/reference; it does not transfer ownership and callers must not call `release()` unless they explicitly retained it.
- Add a clear owned/borrowed/retained terminology section to memory package docs.
- Add explicit factory/handoff helpers only where they reduce ambiguity; avoid wrapping every storage in an allocation-heavy owner object.
- Document constructor failure behavior and ownership transfer for every public constructor/factory.
- Define whether storage access remains valid only while the owning tensor is open.
- Consider a future `TensorMetadata` / `StorageDescriptor` for read-only capacity, alignment, memory kind and accessibility without exposing mutable lifetime operations to ordinary consumers.

### 2.5 ArenaTensorStorage and views

`ArenaTensorStorage` uses a dedicated FFM arena and CAS-based reference counting. It rejects retain/release after final release and checks reference-count overflow. `HostAccessibleStorageView` retains its parent, uses a checked byte range and releases the parent when the view's own count reaches zero. Existing tests cover parent lifetime, sibling/nested views, aliasing and invalid ranges.

These are good patterns to preserve.

Open questions:
- The view is byte-oriented; it does not carry tensor shape, dtype, strides or offset metadata. A storage view alone is not a complete tensor view.
- `HostAccessibleStorageView` uses synchronized reference counting while `ArenaTensorStorage` uses atomics. This is not inherently wrong, but the concurrency guarantees should be explicit and consistent at the abstraction level.
- Verify the race between final release and segment access across all storage implementations, not only the current host view.
- Ensure device/provider storage has equivalent ownership tests even where it cannot expose a host segment.

### 2.6 ManagedArena requires a focused correctness fix

`ManagedArena` uses `AtomicInteger.getAndIncrement()` for retain and `decrementAndGet()` for close. It rejects retaining after the count has reached zero, but `close()` does not reject over-release. Calling close more times than the initial reference count can drive the count negative. It also lacks the explicit overflow guard used by `ArenaTensorStorage`.

This is a concrete lifecycle inconsistency, not just a documentation issue.

**Recommended next code change:** make `ManagedArena` reject over-release and reference-count overflow with a CAS-based transition (or an equivalent atomic state protocol), validate the wrapped arena, and add tests for:
- Double close / over-release.
- Retain after close.
- Retain at reference-count overflow (using a testable counter strategy rather than billions of calls).
- Concurrent retain versus final close.
- Exactly-once underlying arena closure.

Keep its intended shared and confined arena thread-access rules explicit. Reference counting does not make a confined FFM arena safe to access from other threads.

### 2.7 Layout and indexing are missing as first-class tensor metadata

The public tensor metadata currently exposes shape, dtype and device, but no canonical stride, offset, memory order or layout contract. Storage views can represent byte ranges, while tensor operations such as transpose, slice and reshape are expressed only as operations returning tensors.

This leaves important questions implementation-specific:
- Is transpose a view or materialized copy?
- Is reshape guaranteed to share storage when possible?
- Are slices contiguous?
- What are the stride and offset semantics for non-contiguous tensors?
- How do broadcasting and indexing behave across providers?
- Can a backend accept arbitrary strides or does it silently materialize?

**Recommendation:** define a logical `TensorLayout` descriptor (shape remains separate) with at least strides, storage offset and a documented contiguous/strided classification. Keep allocation capacity separate from the logical view range. Introduce only after auditing all existing backends and kernels; use compatibility defaults for existing contiguous implementations.

### 2.8 Backend coupling

`Tensor.backend()` exposes the full `ComputeBackend` interface. The backend SPI currently combines a large mathematical operation set, allocation, host/device copy operations and provider lifecycle. This is useful internally, but forces public tensor consumers to know about a backend implementation contract.

**Recommendation:** separate concepts gradually:
- Tensor operations remain the primary semantic API.
- A read-only execution descriptor can expose provider identity, device capabilities and diagnostics.
- Explicit advanced interop may expose a provider/session handle where needed.
- Allocation and transfer remain provider responsibilities.
- Backend selection and lifecycle are managed by the runtime/application layer, not by an arbitrary tensor consumer.

Do not remove `backend()` in a breaking change until all callers and external artifacts are mapped. Deprecation, adapters and migration documentation would be required.

### 2.9 Eager and lazy semantics

`DefaultTensor` executes operations immediately through its backend and records autograd graph metadata. `LazyTensor` records a graph and materializes through `eval()`. Lazy wrappers can borrow constant leaves and release only their owned materialized result. Some operations (for example split) are explicitly unsupported; device reporting for unevaluated lazy tensors currently defaults to CPU.

**Recommendations:**
- Define whether `eval()` returns the same object, a borrowed result or an independently owned result for each implementation.
- Define ownership of lazy graph inputs and the lifetime requirement for borrowed leaves.
- Do not report CPU as the actual execution device merely because the graph has not run. Represent unknown/deferred placement explicitly or expose planned placement separately.
- Define operation support and failure behavior through capability metadata or a support matrix.
- Add tests for closing inputs before evaluation, closing lazy wrappers before/after materialization, repeated evaluation, evaluation failure cleanup and device placement.
- Specify whether graph optimization preserves autograd metadata and observable numerical behavior.

### 2.10 Autograd ownership

The Tensor contract documents that `grad()` is borrowed and `setGrad()` transfers ownership. `DefaultTensor` implements replacement by releasing the previously owned gradient and releases the current gradient when the tensor is released. This is a good explicit ownership pattern.

Remaining audit requirements:
- Test self-gradient and cyclic ownership rejection beyond the direct self-reference.
- Clarify whether assigning a gradient requires matching shape, dtype and device; enforce consistently.
- Clarify graph lifetime and whether gradients may outlive graph intermediates.
- Define accumulation versus replacement semantics for repeated backward calls.
- Define gradient clearing/zeroing and graph-retention behavior.
- Ensure lazy `grad()` does not unexpectedly materialize or change ownership without documentation.

## 3. Proposed canonical contract matrix

| Concern | Canonical rule to specify | Current status |
|---|---|---|
| Shape | Immutable logical dimensions; scalar and empty semantics | Implemented; document edge semantics |
| Layout | Strides, offset, order, contiguous/strided | Gap in public tensor metadata |
| DType | Logical and compute types, promotion, conversion | Partial; enum conflates quantized storage |
| Storage encoding | Physical primitive/block representation and metadata | Partial; block formats in DType |
| Device | Logical placement and transfer semantics | Present; device capability model needs review |
| Ownership | Owned, borrowed, retained, transferred | Partially documented |
| Views | Parent retention, aliasing, mutation and view range | Host storage views present; tensor view contract incomplete |
| Copy | Explicit copy/materialization/transfer semantics | Backend copy APIs present; observability needed |
| Lifecycle | Idempotence, use-after-close and cleanup | DefaultTensor/storage mostly guarded; ManagedArena over-release gap |
| Backend | Semantic operations vs provider identity/lifecycle | Coupled through Tensor.backend() |
| Eager/lazy | Evaluation, placement, input lifetime and ownership | Different semantics; contract incomplete |
| Autograd | Gradient ownership, shape/device validation, accumulation | Partially documented |

## 4. Conformance tests to add before API redesign

### P0 — Lifecycle correctness

- `ManagedArena` double close and over-release.
- Retain after final release for every storage implementation.
- Concurrent retain/release and exactly-once destruction.
- `DefaultTensor` use-after-close behavior for every public data/operation method.
- Constructor failure leaves caller ownership intact.
- Gradient replacement and tensor close release each owned resource exactly once.
- Lazy leaf borrow and materialized-result ownership.

### P1 — View and data movement semantics

- View retains parent after parent tensor closes.
- Nested and sibling views preserve lifetime and aliasing.
- View range and tensor logical range cannot exceed capacity.
- Mutation visibility for aliased views is explicit.
- Transpose/slice/reshape view-versus-copy behavior.
- Host-to-device and device-to-host copies preserve shape, dtype and values.
- Copy/transfer failures release temporary resources.
- Device-only storage never accidentally assumes host accessibility.

### P2 — Numerical metadata semantics

- Scalar rank-zero and zero-sized dimensions.
- Negative axis normalization and out-of-range rejection.
- Dtype promotion matrix and scalar overload behavior.
- Quantized block alignment and shape/storage compatibility.
- Layout/stride correctness for non-contiguous operations.
- Broadcasting and matmul shape validation.
- Consistent behavior across eager/lazy and all supported providers.

### P3 — API usability and diagnostics

- Actionable errors include operation, shapes, dtype, layout, device and unsupported capability.
- No accidental backend-specific exception leakage from public APIs.
- Copy/materialization/transfer can be observed or profiled.
- Public examples demonstrate try-with-resources and borrowed versus owned results.
- API docs state whether each operation is eager/lazy, differentiable, view-producing and supported by which providers.

## 5. Recommended implementation sequence

Keep each change small and independently reviewable.

1. **Fix ManagedArena lifecycle** with tests. This is a concrete bug and does not require a public Tensor API redesign.
2. **Document ownership terminology** and correct any ambiguous public Javadocs, especially `Tensor.storage()`, `Tensor.grad()`, `setGrad()`, constructors and `eval()`.
3. **Build a storage conformance test suite** reusable by host, device and view implementations.
4. **Audit all Tensor implementations and backend callers** before introducing layout/capability interfaces.
5. **Design logical layout metadata** and test compatibility across CPU/Metal/CUDA/HAT and format adapters.
6. **Design logical dtype / compute dtype / storage encoding separation** with conversion and promotion matrices.
7. **Introduce an execution descriptor** as a non-breaking alternative for ordinary consumers to inspect provider identity/capabilities.
8. **Publish API migration guidance** only after the compatibility path is implemented and tested.

## 6. Non-goals

- Do not replace FFM with JNI or heap arrays.
- Do not assume every tensor is host-accessible.
- Do not force device storage to expose a `MemorySegment`.
- Do not equate zero-copy with no synchronization, no ownership transfer or no lifetime constraints.
- Do not introduce a DataFrame, sparse, distributed or symbolic tensor API as part of this lifecycle audit.
- Do not remove `ComputeBackend`, `backend()`, `Buffer` or existing DType constants in this increment.
- Do not claim cross-backend parity without provider-specific conformance results.

## 7. Exit criteria

This audit phase is complete when:
- The ManagedArena over-release inconsistency is fixed and tested.
- Owned/borrowed/transferred semantics are documented consistently.
- A reusable storage lifecycle conformance suite covers all current implementations.
- Tensor metadata gaps and migration options are mapped against all implementations and downstream callers.
- No public API breaking change is introduced without a separate compatibility proposal and test evidence.
