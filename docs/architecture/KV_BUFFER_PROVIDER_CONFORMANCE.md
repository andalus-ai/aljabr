# KV buffer provider conformance

The heap, FFM-native, and Metal KV-buffer providers share one reusable JUnit
contract in `aljabr-core` test fixtures. Each provider supplies only:

- its storage allocation,
- the expected backend and memory-domain identifiers, and
- a scoped reader that copies values out while the lease is active.

The contract checks per-layer logical geometry (including heterogeneous KV head
counts and dimensions), FLOAT32 region sizes, backend/domain metadata, readiness,
and actual K/V values. It also verifies that an active lease remains bounded to
its original committed prefix, permits appends beyond that prefix, and rejects
mutation of positions visible through the lease.

This is a test contract, not a production abstraction. Backend-specific access
remains in each provider's scoped access API. Metal execution is skipped when a
unified-memory Metal device and the required buffer allocation capability are
unavailable; a skipped test is not evidence of hardware correctness.
