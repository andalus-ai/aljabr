# Quantized Tensor Interoperability Contract

## Status

This document defines the cross-repository contract for quantized tensor data exchanged by Aljabr, Tafkir, and AQLI. It is a format and ownership specification, not a claim that every listed format or algorithm is implemented or production-ready.

The contract is intentionally separate from any one quantization algorithm. GPTQ, AWQ, AutoRound, NF4, FP8, INT8, and future formats may use it when their representation is fully specified and validated.

## Ownership

| Responsibility | Owner | Required behavior |
|---|---|---|
| Canonical tensor descriptor, format identifiers, metadata validation, packing/codecs, generic numerical primitives | Aljabr | Define stable, algorithm-neutral representation contracts. Do not depend on Tafkir or AQLI. |
| Calibration, algorithm-specific optimization, quantization decisions, conversion workflows, producer provenance | Tafkir | Produce tensors/artifacts that conform to an Aljabr format contract; retain algorithm-specific workflow state where it is not needed to decode the stored tensor. |
| Artifact loading, compatibility checks, execution-provider selection, native quantized kernels, runtime fallback/dequantization | AQLI | Consume supported formats without depending on Tafkir implementation classes or training dependencies. |

An algorithm name is not a storage format. AQLI must select decoding and kernels from the persisted format identifier and version, not infer a layout from a producer's algorithm label.

## Tensor-level descriptor

Every persisted quantized tensor must be accompanied by a descriptor with the following semantic fields. Concrete Java types and serialization encoding are to be selected by the Aljabr API implementation, while preserving these meanings.

| Field | Required | Meaning |
|---|---:|---|
| `formatId` | Yes | Stable namespaced format identifier, e.g. `org.bitsandbytes.nf4` or `org.ggml.q4_k`. Must identify the actual byte layout and decoding semantics. |
| `formatVersion` | Yes | Positive version of that format contract. Never silently reinterpret old payloads after a layout change. |
| `logicalShape` | Yes | Original tensor dimensions, in a documented dimension order. |
| `logicalDType` | Yes | Element type represented before quantization. |
| `storageDType` | Yes | Physical payload element/storage type (including packed-byte representation). |
| `bitWidth` | Yes | Bits per logical value, where applicable; formats with mixed/variable coding must define this explicitly or mark it as not fixed. |
| `packingOrder` | Conditional | Bit/nibble order, endianness, lane ordering, and padding rules whenever values are packed. |
| `blockSize` / `groupSize` | Conditional | Exact block/group semantics, including grouping axis and behavior for incomplete final groups. |
| `quantizationAxis` | Conditional | Axis for per-channel/per-axis metadata. |
| `scales` | Conditional | Scale values, dtype, shape, ordering, and association with blocks/groups/channels. |
| `zeroPoints` | Conditional | Zero-point values, dtype, shape, ordering, and signedness/offset convention. |
| `codebook` | Conditional | Codebook identity/version or the codebook data itself when not globally standardized. |
| `auxiliaryState` | Conditional | Required decoder state such as rotations, permutation/index maps, nested scales, or format-specific parameters. |
| `payloadLength` | Yes | Exact encoded payload length; readers must reject inconsistent lengths. |

Conditional fields are mandatory whenever the selected format requires them. A descriptor must not rely on undocumented defaults for decoding-critical values.

### Separation of format and provenance

The tensor descriptor describes how to decode the bytes. Producer provenance describes why and how the bytes were created. Keep these concerns distinct.

Producer provenance should record, where available:
- algorithm and implementation identifier/version (for example GPTQ or AWQ);
- quantizer/library version and relevant configuration;
- calibration dataset identity/checksum and sample policy, without embedding private data;
- source model and source tensor checksums;
- conversion timestamp and reproducibility/seed information.

Provenance is valuable for audit and reproducibility, but must not be required to decode a tensor if the format descriptor is complete.

## Model artifact boundary

A model artifact is larger than a quantized tensor. A model-level manifest should reference tensor descriptors and payload locations and preserve the model configuration, tokenizer, shard/index information, and other model assets required by the model format.

Tafkir owns producing a complete converted artifact. Aljabr owns reusable tensor-level format and codec contracts. AQLI owns validating and loading the artifact for execution. Do not make Aljabr's tensor descriptor responsible for model packaging, repository upload, tokenizer lifecycle, or serving policy.

## Validation requirements

A producer must not publish an artifact as supported unless it can demonstrate all of the following for the exact format/version:

1. Descriptor validation: required fields, valid dimensions, legal bit width, axis/group consistency, and bounded metadata sizes.
2. Payload validation: exact lengths, legal padding, packing order, endianness, and no out-of-bounds reads.
3. Decode validation: reference-backed numerical comparison against the format's authoritative implementation, including non-divisible dimensions and partial final groups.
4. Round-trip validation: quantize/dequantize error metrics appropriate to the algorithm; a self-consistent encoder/decoder pair alone is insufficient evidence of external format compatibility.
5. Serialization validation: write/read across process boundaries and across supported library versions.
6. Negative validation: unknown format IDs/versions, missing auxiliary state, corrupt lengths, unsupported dtypes, and malformed metadata fail closed with actionable errors.

AQLI must reject unsupported format IDs or versions before executing kernels. It may use a dequantization fallback only when a validated decoder exists. It must not silently reinterpret unknown data as INT8, FP16, or another convenient type.

## Runtime execution behavior

AQLI should prefer a native quantized operator when the selected backend advertises support for the exact format/version, shape constraints, and required auxiliary state. Dequantization to a higher precision representation is a fallback, not the default definition of quantized inference.

Runtime capability selection belongs to AQLI. Low-level unpacking, conversion, and math kernels may be supplied by Aljabr. The presence of an Aljabr codec does not imply that every AQLI backend can execute the format natively.

## Compatibility and evolution

- Format IDs are stable and namespaced; never recycle an identifier for different semantics.
- Increment the format version for incompatible payload or metadata changes.
- Readers may support multiple versions; writers emit only versions they explicitly implement.
- Unknown optional metadata may be preserved, but unknown decoding-critical metadata must cause rejection.
- Metadata and payload must be treated as untrusted input: validate lengths, dimensions, offsets, and resource limits before allocation or access.
- In-memory descriptors should be immutable from callers' perspective. Array/buffer inputs and outputs must use defensive copies or read-only views with explicit lifetime ownership.
- Do not expose mutable arrays or memory segments whose lifetime is shorter than the descriptor or payload consumer.

## Current API gap and migration guidance

Aljabr currently has `QuantizedTensorInfo` fields for shape, original/quantized dtype, strategy, scales, zero points, group size, channel axis, and opaque metadata. This is a useful starting point but is not yet a complete interoperable format descriptor: it does not explicitly identify a versioned byte layout, packing order, payload length, or structured codebook/auxiliary state. Its array fields are also exposed without defensive copying.

The current `QuantConfig` contains both representation parameters and algorithm/workflow parameters (including calibration sample count, sequence length, activation ordering, and dampening). Keep algorithm-specific configuration in Tafkir workflows; do not make it part of the minimum runtime decoding contract.

The existing `Quantizer` SPI is an algorithm provider interface. It should not be treated as the persisted tensor format contract. Evolve the descriptor and quantizer SPI deliberately and compatibly rather than renaming or replacing existing APIs in one breaking change.

## Acceptance criteria for implementation

- Aljabr publishes an algorithm-neutral, versioned quantized tensor descriptor and validation API.
- Tafkir adapters emit descriptors whose format identifiers describe the actual payload layout, and attach all required decoding state.
- AQLI validates format/version and backend capabilities before loading or executing quantized weights.
- Cross-repository fixtures prove a Tafkir-produced artifact can be read by Aljabr and consumed by AQLI without a Tafkir runtime dependency.
- Unsupported or unvalidated formats remain unavailable; documentation distinguishes mathematical primitives, artifact-format support, and native inference-kernel support.
