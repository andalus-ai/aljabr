## CPU native-memory KV provider

NativeKVCacheStorage is an explicitly selected CPU provider backed by the JDK 25 Foreign Function & Memory API. It allocates per-layer K/V regions from one storage-owned shared Arena; it does not use JNI or expose native addresses through the generic cache contract.

The provider exports HOST_NATIVE / FLOAT32 regions using opaque handles. NativeKVCacheBufferAccess.withReadOnlySegment(...) supplies a read-only MemorySegment slice to a synchronous callback. The segment must not be retained or used by work that outlives the callback and active lease. Closing storage is rejected while leases are active. Mutations into a borrowed committed prefix are rejected, while append writes beyond that prefix remain available.

Heap storage remains the portable default. Native memory is opt-in and does not imply that a kernel is FFM-compatible or that an operation is zero-copy end-to-end. Provider selection and kernel compatibility remain explicit. The shared arena permits access from multiple threads, while storage operations and scoped access are serialized by the provider.

The implementation uses standard JDK 25 APIs and does not require --enable-native-access for ordinary arena allocation and segment access. Native linking/downcalls, if introduced by a future provider, must remain separately gated and documented.
