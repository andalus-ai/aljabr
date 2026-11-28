# Metal KV storage provider

## Scope

The optional `MetalKVCacheStorage` provider is an implementation of the format-neutral KV storage contract in the Metal backend module. It allocates one retained `MTLBuffer` for K and one for V per layer, using `MTLResourceStorageModeShared`. The provider is opt-in and currently requires an active Metal runtime with unified memory.

The Objective-C bridge retains each `MTLBuffer` independently of dispatch-scoped tensor wrappers. The Java storage owner releases those references only after all borrowed leases have closed. The generic cache API continues to expose only opaque `KVCacheBufferHandle` values and region metadata.

## Ownership and access

- `MetalKVCacheStorage` owns the native Objective-C buffer references and their CPU-visible contents segments.
- `KVCacheSession` opens leases bounded to the committed sequence prefix.
- Writes into a prefix visible to an active lease are rejected; appends beyond that prefix remain allowed.
- Clear, truncate-through-prefix, and storage close are rejected while affected leases are active.
- `MetalKVCacheBufferAccess.withBuffer(...)` provides a callback-scoped view containing the opaque Objective-C buffer reference and a read-only CPU-visible contents slice.
- The callback must not retain either view value. The Objective-C reference must not be released by consumers.

## Synchronization boundary

The provider currently writes K/V from the CPU into shared-storage buffers. Those writes are synchronous. Its readiness stage is therefore completed when the lease is opened. The existing Metal dispatch path submits and waits for command completion before returning; consumers must preserve that ordering when using these buffers.

This provider does not yet expose asynchronous command-buffer submission, GPU-side KV mutation, events/fences, or cross-queue synchronization. A future asynchronous Metal consumer must add explicit command-buffer dependency handling before advertising readiness for GPU-produced cache data.

## Limitations

- This is a shared-memory Metal provider for unified-memory devices, not a private-device-memory or discrete-GPU provider.
- A Metal kernel must explicitly consume the MTLBuffer handle; creating a region does not automatically route existing attention kernels to it.
- No end-to-end zero-copy or performance improvement is claimed until the attention path is integrated and measured on Apple hardware.
- Heap and CPU-native FFM providers remain independent alternatives.

## Direct borrowed-buffer decode attention

`MetalKVCacheAttentionConsumer` provides an opt-in single-query decode path over the committed prefix of a Metal cache lease. It consumes the existing `aljabr_sdpa_flat_kernel` pipeline, whose K/V addressing matches contiguous sequence-major `[sequence, kvHead, headDimension]` storage.

The Objective-C bridge receives the retained K/V `MTLBuffer` object handles and binds those objects directly at the kernel's K/V buffer slots. It does not wrap the CPU contents pointer as a second Metal buffer and does not copy K/V into temporary storage. Query and output remain caller-owned buffers wrapped for the dispatch.

The consumer validates the lease backend, device identity, FLOAT32 encoding, byte span, and exact sequence-major stride requirements. It waits for the lease readiness stage and keeps both scoped buffer views—and therefore the originating lease—open until the command buffer completes. The current kernel supports one decode query, up to 4096 visible positions, and synchronous completion.

This is a direct K/V binding path, not a claim that every Metal attention workload is zero-copy. Paged/block-table attention, multi-token prefill, non-FLOAT32 caches, incompatible layouts, and other backends continue to use their existing paths. Hardware correctness and performance still require Apple Silicon testing.

HAT remains optional and separate. Its current integration stages values into HAT-owned arrays; this Metal consumer does not imply HAT/Metal external-memory interoperability.
