# Aljabr Diffusion Video Conditioning

Optional reusable per-frame conditioning integration for the generic diffusion conditioning SPI.

## Frame preparation

- One VIDEO_FRAME input produces one VideoFrameConditioningData result.
- Each input carries a non-negative sequenceIndex and required timestamp metadata parsed as an ISO-8601 instant.
- Decoding and preprocessing are injected. Codec, source resolution, temporal sampling, resize/crop, normalization, and color-space policy remain application-owned.
- Decoded pixels are immutable heap-owned float arrays with explicit HWC/CHW layout and finite-value validation.

## Lazy sequence chunking

`VideoFrameSequenceChunker` exposes contiguous frame chunks through a repeatable, pull-based `Iterable`.

- Each iterator creates one non-owning sequence view when the consumer requests it; it does not materialize a list of all chunks.
- `INCLUDE_PARTIAL` retains the final non-empty short chunk; `DROP_PARTIAL` emits only complete chunks.
- Each iterator starts independently at the first source frame, and chunks preserve original frame objects, indices, and timestamps.
- Pull-based iteration lets a downstream consumer control how quickly it requests more work. It does not create threads, schedule execution, or provide asynchronous backpressure.
- Chunks do not own or close frame resources; the original prepared conditioning set remains the owner and must stay open while chunks are consumed.

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
## Temporal sampling

Sampling is an independent policy capability applied after frame preparation.

- `VideoFrameSamplingPolicy` allows applications to provide custom deterministic selection strategies.
- `EvenlySpacedVideoFrameSamplingPolicy` selects a requested count across the sequence, including both endpoints when count is greater than one. A one-frame request selects the lower middle frame. Requests larger than the available sequence are rejected.
- `TimestampIntervalVideoFrameSamplingPolicy` keeps the first frame and then selects the first available frame at or after each fixed interval target measured from the first timestamp. Targets missed by a large gap are skipped rather than causing dense catch-up selection.
- `TimestampWindowVideoFrameSamplingPolicy` selects frames inside an inclusive timestamp window. A window containing no frames yields an empty sampling result.
- `VideoFrameSampler` validates custom policy output: selected frames must be original instances from the source sequence, unique, and in source order.
- Sampling returns a non-owning sequence view. It preserves original sequence indices and timestamps, does not copy pixel data, and does not change ownership.
## Sequence windowing

VideoFrameSequenceWindowing creates immutable, non-owning windows for bounded-memory or chunked processing.

- Window size and start stride are expressed in frame counts; both must be positive.
- A stride smaller than the window size creates overlap; a larger stride intentionally skips frames between windows.
- INCLUDE_PARTIAL retains a final non-empty short window; DROP_PARTIAL emits only complete windows and may return an empty list.
- Windows preserve original frame instances, sequence indices, timestamps, and source ownership. They do not copy pixels, decode, interpolate, or alter temporal metadata.
- Windowing is independent of temporal sampling and can be applied to an already sampled sequence.

No FFmpeg, ImageIO, Python, PyTorch, ComfyUI, tensor backend, or device-selection dependency is introduced.
