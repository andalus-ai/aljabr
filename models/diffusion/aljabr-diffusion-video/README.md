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

## Sequence windowing

VideoFrameSequenceWindowing creates immutable, non-owning windows for bounded-memory or chunked processing.

- Window size and start stride are expressed in frame counts; both must be positive.
- A stride smaller than the window size creates overlap; a larger stride intentionally skips frames between windows.
- INCLUDE_PARTIAL retains a final non-empty short window; DROP_PARTIAL emits only complete windows and may return an empty list.
- Windows preserve original frame instances, sequence indices, timestamps, and source ownership. They do not copy pixels, decode, interpolate, or alter temporal metadata.
- Windowing is independent of temporal sampling and can be applied to an already sampled sequence.

No FFmpeg, ImageIO, Python, PyTorch, ComfyUI, tensor backend, or device-selection dependency is introduced.
