# KV buffer consumer compatibility

A provider's region descriptor describes where memory lives; it does not grant
universal access to that memory. Direct borrowing is valid only when the
consumer's requirements match the provider's exported metadata.

`KVCacheBufferCompatibilityValidator` centralizes these checks for both K and V:

- backend identifier,
- memory domain (host heap, host native, or device),
- encoded element type, and
- exact device identity (empty only for host memory).

The validator also invokes the shared logical-layout/region-span validator.
Consumers still apply their kernel-specific layout, alignment, offset, and
synchronization requirements before dispatch.

A mismatch must be rejected before interpreting an opaque handle. Moving data
between backends requires an explicit transfer path and must never be described
as direct borrowing or zero-copy.
