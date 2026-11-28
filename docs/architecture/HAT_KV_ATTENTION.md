# Optional HAT KV Attention

The optional HAT backend now has a staged single-query GQA decode consumer.
It accepts the format-neutral `KVCacheReadView`, reads only the committed
visible prefix into caller-independent temporary arrays, copies those arrays
into HAT `F32Array` buffers, dispatches a stable online-softmax kernel, and
copies the output back to a Java array.

This is intentionally a separate optional execution path:

- It requires the Babylon JDK 26 HAT toolchain and is not part of the JDK 25
  default build.
- It uses the logical read-view contract, not Metal `MTLBuffer` handles or
  native/device pointers.
- K/V are staged on every invocation. It does not claim zero-copy or direct
  Metal/HAT interoperability.
- The current kernel is correctness-first and computes one output element per
  work item with an online softmax recurrence. It is not a tiled or
  high-throughput attention implementation.
- Dispatch is serialized while the current HAT Java backend's mutable
  compute-context behavior remains unverified for concurrent use.
- The upstream HAT Accelerator has no close contract; the consumer reuses its
  accelerator rather than creating one per call.

The consumer is explicit and opt-in. Existing HAT backend operations continue
to use their established dispatch/fallback behavior unless an application
selects this consumer. The portable logical read-view path remains available
when no HAT backend is installed.

Before promoting this path as a production optimization, add parity coverage
for multi-head GQA, nontrivial head dimensions, longer sequences, numerical
edge cases, and compare it against CPU reference attention. Benchmark staging
cost separately from HAT kernel execution.
