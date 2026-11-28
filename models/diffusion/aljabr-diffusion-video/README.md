# Aljabr Diffusion Video Conditioning

Optional reusable per-frame conditioning integration for the generic diffusion conditioning SPI.

## Frame preparation

- One VIDEO_FRAME input produces one VideoFrameConditioningData result.
- Each input carries a non-negative sequenceIndex and required timestamp metadata parsed as an ISO-8601 instant.
- Decoding and preprocessing are injected. Codec, source resolution, temporal sampling, resize/crop, normalization, and color-space policy remain application-owned.
- Decoded pixels are immutable heap-owned float arrays with explicit HWC/CHW layout and finite-value validation.

## Sequence view

VideoFrameSequence groups prepared frame results into a validated temporal view.

- Caller order is retained; the sequence does not sort or resample.
- Sequence indices and presentation timestamps must both be strictly increasing.
- Sparse sequence indices are allowed, supporting intentionally sampled frames.
- Duration is elapsed presentation time between the first and last timestamps, not an estimated frame rate.
- The sequence is non-owning. PreparedDiffusionConditioningSet remains the single owner and must stay open while the sequence is in use.
- Mixed prepared conditioning data can be filtered to a video-only view; every VIDEO_FRAME entry must use VideoFrameConditioningData.

## Temporal sampling

Sampling is an independent policy capability applied after frame preparation.

- `VideoFrameSamplingPolicy` allows applications to provide custom deterministic selection strategies.
- `EvenlySpacedVideoFrameSamplingPolicy` selects a requested count across the sequence, including both endpoints when count is greater than one. A one-frame request selects the lower middle frame. Requests larger than the available sequence are rejected.
- `TimestampIntervalVideoFrameSamplingPolicy` keeps the first frame and then selects the first available frame at or after each fixed interval target measured from the first timestamp. Targets missed by a large gap are skipped rather than causing dense catch-up selection.
- `TimestampWindowVideoFrameSamplingPolicy` selects frames inside an inclusive timestamp window. A window containing no frames yields an empty sampling result.
- `VideoFrameSampler` validates custom policy output: selected frames must be original instances from the source sequence, unique, and in source order.
- Sampling returns a non-owning sequence view. It preserves original sequence indices and timestamps, does not copy pixel data, and does not change ownership.

No FFmpeg, ImageIO, Python, PyTorch, ComfyUI, tensor backend, or device-selection dependency is introduced.
