# Shared Attention Context Contracts

## Purpose

This document defines Aljabr's model- and algorithm-neutral foundation for
reusing target attention context in sparse attention and speculative drafting.
DFly2 is an intended consumer, not the owner of these contracts.

## Ownership

- The target sequence/cache owner remains responsible for allocation, writes,
  truncation, reset, and final storage release.
- A consumer receives a borrowed `KVCacheReadView`; it never owns or closes the
  underlying `KVCacheStorage`.
- The view is bounded to the committed prefix captured when it is opened.
- The session rejects writes into the borrowed committed prefix and rejects
  truncation/reset while views are active. Appends beyond the snapshot remain
  possible.
- Direct mutation through the underlying storage bypasses session protection
  and is forbidden while a view is active.

## Sparse index selection

`AttentionIndexSelection` stores absolute sequence positions selected from a
source prefix. It is immutable, validates bounds and duplicate positions, and
preserves indexer order. It intentionally does not encode a model family,
attention layer, top-k policy, scoring function, or kernel layout.

A consumer validates that the selected source prefix is visible through its
`KVCacheReadView` before applying the selection.

## Capability and fallback

`KVCacheSharingCapability` is optional. A storage/session path that cannot
provide a safe borrowed view must not advertise it. Dense attention and
independent-drafter-cache execution remain valid without this capability.

## Deliberate limitations of this phase

- The initial read view exposes logical K/V range reads and may copy into
  caller-owned arrays. It does not claim end-to-end zero-copy.
- Native pointers, device tensors, page tables, cache quantization metadata,
  sparse-index production, and attention kernels are not part of this contract.
- Thread/device synchronization beyond the session's mediated access is a
  provider responsibility and must be documented by any optimized provider.
- The selection represents positions only; query-specific, per-layer and
  multi-head selection metadata can be added only when a concrete model/API
  requires it.

## Validation

The core tests cover immutable selection, range validation, snapshot visibility,
borrow lifetime, append behavior, mutation exclusion, and storage ownership.
Provider-specific conformance tests must additionally validate device ordering,
resource lifetime, and concurrent execution before advertising optimized
sharing.

## Sparse attention indexing contract

`AttentionIndexer` selects absolute sequence positions for one query and
returns an immutable `AttentionIndexSelection`. It does not access KV storage,
compute relevance outside its own indexing strategy, or execute attention.

`AttentionIndexingRequest` carries the visible source sequence length, query
position, selection budget, and causal constraint. The caller validates every
result with the request before using it. Selection order is preserved and need
not be sorted. Causal requests reject future positions. Indexers must not
silently exceed the requested selection budget or return a selection for a
different source prefix.

Models expose indexing only when they support it through the optional
`AttentionIndexingCapability`. Dense models remain valid without this
capability. A caller may request separate selections for different layers or
heads; this phase intentionally does not impose a universal layer/head scope
model or assume all heads share one selection.

The indexing contract remains independent of KV sharing. A selection can be
validated against a `KVCacheReadView` and then consumed by an attention kernel
that understands the model's cache layout. The indexer itself never receives
or owns the cache view.

### Deliberate non-goals

- No built-in scoring algorithm or learned indexer weights.
- No top-k implementation or sorting policy.
- No model-family-specific index tensors or sparse-attention kernels.
- No assumption that selections are reusable across queries, layers, heads,
  requests, or cache generations. Reuse requires the caller to establish that
  the selection's source prefix and scope still match.

## Backend-aware KV buffer borrowing

Logical reads through `KVCacheReadView` remain the portable compatibility
path. Providers may separately advertise `KVCacheBufferSharingCapability`
when they can safely lend native allocation regions to compatible consumers.
This capability is optional and does not replace logical reads.

A `KVCacheBufferReadView` is a lease over target-owned storage. Its K/V
`KVCacheBufferRegion` descriptors identify an opaque allocation handle,
memory domain, backend/device identity, element encoding, and byte range.
The descriptor does not expose a pointer or permit arbitrary dereferencing.
Only a compatible backend may interpret the handle while the lease remains
open. Closing the lease releases the borrow, never the target allocation.

Memory domains distinguish Java heap, native host, and accelerator device
memory. Device regions require a device identity; host regions do not. Element
encodings are explicit so consumers cannot silently interpret FP16/BF16/FP8
storage as float32. Region offsets and lengths are byte-based and element
aligned.

### Interoperability and fallback rules

- Same-backend consumers may use a borrowed region without a host round trip
  when the provider/backend explicitly supports that operation.
- Cross-backend consumers must use an explicit transfer/import path. They must
  not claim zero-copy merely because a descriptor is available.
- Providers that cannot guarantee allocation lifetime, mutation exclusion,
  and synchronization must not advertise buffer sharing.
- Consumers must close leases on success, failure, cancellation, and timeout.
- The existing logical `KVCacheReadView` remains the fallback for providers
  that can safely copy values but cannot expose native/device allocations.
- This phase defines the interop contract only. CPU, Metal, and CUDA storage
  providers must implement and pass provider-specific conformance tests before
  advertising the capability.

No JDK Foreign Function & Memory type is exposed in the core contract. A CPU
provider may internally use `MemorySegment`; Metal/CUDA providers may use their
own device allocation handles. This keeps the common API usable across memory
domains without pretending device pointers are host-dereferenceable.
