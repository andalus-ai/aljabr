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
