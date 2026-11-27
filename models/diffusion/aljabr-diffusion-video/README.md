# Aljabr Diffusion Video Conditioning

Optional reusable per-frame conditioning integration for the generic diffusion conditioning SPI.

- One VIDEO_FRAME input produces one VideoFrameConditioningData result.
- DiffusionConditioningSet preserves caller-declared order and validates unique frame sequence indices. Supply frames in intended order; this adapter does not sort or resample.
- Each input carries a non-negative sequenceIndex and required timestamp metadata parsed as an ISO-8601 instant, for example 2026-11-27T10:15:30.250Z. No constant frame rate is assumed.
- VideoFrameDecoder and VideoFrameConditioningTransform are injected. Codec, source resolution, temporal sampling, resize/crop, normalization, and color-space policy remain application-owned.
- Decoded pixels are immutable heap-owned float arrays with explicit HWC/CHW layout and finite-value validation. Pixel range and color semantics remain decoder/transform-defined.

No FFmpeg, ImageIO, Python, PyTorch, ComfyUI, tensor backend, or device-selection dependency is introduced. Whole-container decoding and timestamp inference from frame rates are outside this adapter.
