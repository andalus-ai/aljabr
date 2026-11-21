# KV-cache tensor integration

This module adapts the format- and backend-neutral KVCacheSession API to Aljabr tensors without introducing a dependency from the tensor module back to aljabr-core.

## Snapshot semantics

KVCacheTensorSnapshots.keys(...) and values(...) copy a committed range into a newly allocated F32 tensor with shape [tokenCount, kvHeads, headDimension].

The returned tensor owns independent storage. It remains valid after the cache session is reset or closed. Snapshot creation is intentionally not zero-copy: KVCacheStorage currently defines read/write transfer operations, and heap-backed implementations may grow and relocate their arrays. A future zero-copy capability must be explicit, provider-supported, and carry a retained storage lease for the entire tensor-view lifetime.

Only committed positions are visible through KVCacheSession reads. The adapter does not expose the in-progress decode position.
