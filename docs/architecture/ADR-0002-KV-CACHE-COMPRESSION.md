# ADR-0002: KV-Cache Compression and Quantization Ownership

- **Status:** Proposed
- **Date:** 2026-11-22
- **Scope:** Reusable KV-cache compression contracts, quantization algorithms, execution integration, and inference-runtime policy

## Context

Aljabr already provides a format- and backend-neutral KV-cache storage contract in `tech.kayys.aljabr.cache`:

- `KVCacheLayout` describes per-layer KV-head geometry and head dimensions.
- `KVCacheStorage` owns K/V tensor storage and range reads/writes.
- `KVCacheStorageFactory` creates storage for a logical layout.
- `KVCacheSession` coordinates committed-prefix visibility and sequence-local writes.

These contracts deliberately separate tensor storage from request/session lifecycle. They currently expose float-array reads and writes; they do not define a compressed representation, quantization metadata, or quantized attention operation.

Aljabr also has a separate quantization SPI for model/tensor quantizers. Optional Tafkir adapters include a TurboQuant-named weight-quantizer provider, but that adapter is not evidence of TurboQuant KV-cache support. The existing adapter has been gated pending algorithmic and serialization validation. Weight quantization and KV-cache compression have different data lifetimes, metadata, access patterns, and execution requirements.

TurboQuant research combines vector quantization techniques (including PolarQuant and QJL) for KV-cache compression and attention-related workloads. It must not be represented as a GGUF/Safetensors weight format or treated as a generic INT4 alias.

## Decision

### 1. Aljabr owns reusable KV-cache compression mechanisms

Aljabr owns:

- Backend-neutral contracts for compressed KV-cache representations, when introduced.
- Quantization/compression algorithms and their numerical semantics when they are reusable computational capabilities.
- Reference implementations and correctness oracles.
- Per-layer/per-head geometry validation, encoded-state metadata, and lifecycle rules.
- Backend capability declarations and execution primitives for supported compressed layouts.
- Reusable attention integration required to consume compressed K/V data without forcing AQLI to implement numerical algorithms.

These capabilities must remain independently usable by applications other than AQLI.

### 2. AQLI owns serving policy, not algorithms

AQLI owns:

- Runtime configuration and user-facing CLI/API options.
- Whether KV-cache compression is enabled for a serving profile or request.
- Selection policy among compatible strategies, subject to explicit user constraints.
- Serving memory budgets, fallback policy, and operational reporting.
- Request/session allocation, admission, scheduling, eviction, and serving lifecycle.

AQLI may request a strategy and provide constraints to Aljabr. It must not implement TurboQuant transforms, codebooks, residual correction, quantized attention math, or duplicate Aljabr cache contracts.

### 3. Keep model-weight quantization distinct

The existing tensor/model quantization SPI remains responsible for its current model/tensor quantization use cases. A TurboQuant provider exposed through that SPI must not be assumed to implement KV-cache compression.

Do not:
- Reuse a weight-quantizer adapter as a KV-cache codec merely because both are named TurboQuant.
- Store KV-cache algorithm state in GGUF or Safetensors weight metadata by default.
- Conflate weight bit width with KV-cache key/value precision.
- Advertise a strategy as available or validated until encode, decode, metadata, and numerical behavior are tested.

Shared low-level primitives may be reused only where their semantics genuinely match.

### 4. Preserve the existing cache lifecycle boundary

Compressed storage must integrate with the existing logical cache/session semantics:

- Writes to an in-progress position remain invisible to ordinary committed-prefix reads.
- Attention may read the current position only under the existing layer-local visibility rules.
- Commit, truncate, reset, and close must preserve encoded-state consistency.
- Truncation and reset must release or invalidate all associated compressed payloads and metadata.
- A failed encode/write must not publish a partially encoded position.
- Storage ownership remains explicit; a session must not unexpectedly close caller-owned storage.

Do not move request admission, paging policy, eviction, scheduling, or per-tenant memory policy into Aljabr.

### 5. Make compressed representation metadata explicit

A future compressed-cache contract must be able to describe, as applicable:

- Algorithm and format version.
- Key and value encoding parameters independently.
- Layer/head geometry and grouping/block geometry.
- Rotation or transform identity and reproducibility requirements.
- Codebook/scales/residual metadata and their ownership.
- Encoded payload size and alignment.
- Supported read/decode granularity.
- Numerical error characteristics and compatibility constraints.

Metadata required to decode a payload must travel with that payload or be held by an explicitly owned immutable codec state. It must not depend on hidden process-global mutable state.

### 6. Separate codec support from fused attention support

A provider may support encoding and decoding compressed K/V without supporting direct attention over the encoded representation. These are distinct capabilities.

The reference implementation should first establish correct encode/decode and attention-output parity through a transparent decode path. Fused or approximate attention kernels are separate optimizations and must be advertised only when validated for the relevant dtype, shape, backend, and algorithm version.

### 7. Provider and capability states remain explicit

Use Aljabr's existing provider/capability vocabulary. Keep these states distinct:

- **Declared:** a provider claims to implement the strategy.
- **Installed:** the implementation artifact is present.
- **Available:** required runtime resources are present.
- **Compatible:** algorithm version, layout, dtype, and constraints match.
- **Validated:** reference-backed correctness and lifecycle tests pass.
- **Optimized:** representative benchmarks demonstrate an improvement under documented conditions.

AQLI applies serving policy using these facts. Neither a provider declaration nor a configuration string proves support.

## Consequences

### Positive

- Aljabr remains the single source of truth for reusable numerical and cache-compression semantics.
- AQLI remains a thin policy/serving consumer and can select among multiple algorithms.
- Weight quantization and KV-cache compression cannot accidentally share incompatible payload assumptions.
- CPU reference correctness can be established before backend-specific acceleration.
- Future algorithms can reuse the same cache integration boundary without model-family-specific implementations.

### Costs and constraints

- A compressed representation needs explicit metadata and lifecycle handling beyond the current float-array storage contract.
- A generic codec API must be designed from actual algorithm access patterns rather than speculative abstractions.
- Quantization/dequantization overhead may outweigh memory savings for short contexts or low-concurrency workloads.
- Approximate attention must be validated independently from reconstruction quality.
- Backend support will be incremental; unsupported combinations must fail clearly or use an explicitly permitted fallback.

## Implementation sequence

1. Audit the current Aljabr KV-cache contracts and the active AQLI cache integration; preserve existing lifecycle and ownership semantics.
2. Define the smallest Aljabr-owned compressed-cache representation and codec boundary that supports independently owned metadata and transactional position/block writes.
3. Add a pure-Java reference implementation and deterministic tests for round-trip error, inner-product error, shape/layout handling, truncation, reset, failed writes, and close.
4. Implement TurboQuant's documented KV-cache algorithm as an Aljabr strategy, including PolarQuant transforms and QJL correction where supported by the reference.
5. Integrate the strategy with attention through a correctness-first decode path.
6. Add backend-specific kernels only after the reference path passes and profiling identifies a worthwhile bottleneck.
7. Add AQLI configuration and policy integration after the Aljabr API is stable; test explicit fallback and observability.
8. Benchmark against existing FP32/FP16/BF16 and supported quantized cache paths across context lengths, batch sizes, head geometries, and target backends.

## Validation gates

A TurboQuant KV-cache implementation must not be called production-ready until it has:

- Reference-backed tests for the exact algorithm and payload metadata.
- Deterministic encode/decode behavior for fixed inputs and configuration.
- Bounded reconstruction and attention-output error with documented metrics.
- Correct session commit/truncate/reset/close behavior, including failure paths.
- Tests for heterogeneous per-layer KV geometry and supported head dimensions.
- Explicit capability reporting for each backend and attention path.
- Memory accounting that includes payloads, codebooks, rotations, scales, and temporary workspaces.
- Benchmarks that include encode/decode/attention overhead, not only compressed byte counts.

## Alternatives considered

### Implement TurboQuant in AQLI

Rejected. This would put reusable vector-quantization mathematics and cache representation semantics in the serving runtime, duplicate logic for non-AQLI consumers, and create pressure for AQLI-specific numerical kernels.

### Reuse the existing model-weight TurboQuant provider

Rejected. Weight quantization and KV-cache compression have different representations, lifecycles, access patterns, and correctness requirements. A shared algorithm name is not a compatible storage contract.

### Add a TurboQuant-specific cache implementation directly to each model family

Rejected. This would duplicate numerical behavior, cache lifecycle rules, and backend compatibility logic across model families.

### Add a large generic quantization framework before validating the cache use case

Rejected. The initial API should be the smallest contract justified by the existing `KVCacheStorage` lifecycle and the reference algorithm. Broader vector-quantization abstractions can follow demonstrated reuse.
