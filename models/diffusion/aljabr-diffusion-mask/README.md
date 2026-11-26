# Aljabr Diffusion Mask Integration

Optional reusable adapter for mask conditioning.

## Semantics

- BINARY masks contain discrete zero/one labels. They use nearest-neighbor resampling and an explicit threshold when decoded values are converted to labels.
- SOFT masks contain continuous weights in [0, 1]. They use bilinear resampling by default and preserve fractional values.
- Mask data is single-channel, row-major, finite, and range-validated. It is not treated as an RGB image.

## Extension boundaries

MaskDecoder resolves and decodes the opaque source reference. MaskConditioningTransform owns mask-aware resize/crop operations and receives the selected resampling policy plus the generic preparation context. The transform must honor nearest-neighbor for binary masks and bilinear for soft masks. The adapter applies the binary threshold after the transform.

No image library, tensor backend, device API, Python, PyTorch, or ComfyUI dependency is introduced. The adapter handles MASK only; image/reference-image and video-frame conditioning retain their own semantics.
