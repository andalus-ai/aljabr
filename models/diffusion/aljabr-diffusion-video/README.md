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

## Packed frame batches

VideoFrameBatchPacker materializes a sequence into contiguous, backend-neutral heap memory.

- Output layouts are explicit: THWC or TCHW. Per-frame HWC/CHW source layouts are normalized during packing.
- Frames must have identical width, height, and channel count. Geometry changes are rejected rather than silently resized or padded.
- The batch retains each frame's original input ID, sequence index, and timestamp in matching temporal order.
- The batch owns an independent packed buffer and remains valid after the source prepared conditioning set is closed.
- Public array access is defensive; scalar access is available through logical time/height/width/channel coordinates.
- Packing performs a data copy intentionally. It does not create Aljabr tensors, choose a compute backend, normalize pixel ranges, or assume a model-specific channel convention.

No FFmpeg, ImageIO, Python, PyTorch, ComfyUI, tensor backend, or device-selection dependency is introduced.
