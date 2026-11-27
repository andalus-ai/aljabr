# Aljabr Diffusion Image Integration

Optional, reusable image-conditioning adapter for diffusion pipelines.

## Responsibilities

- Accept opaque IMAGE and REFERENCE_IMAGE conditioning inputs.
- Delegate source decoding to an injected ImageDecoder.
- Delegate resize, crop, channel conversion, normalization, and other image operations to an injected ImageConditioningTransform.
- Return caller-owned ImageConditioningData with explicit width, height, channel count, layout, and defensive pixel storage.

## Representation boundary

DecodedImage is a backend-neutral heap value containing finite floating-point pixels and an explicit HWC or CHW layout. Pixel values are not implicitly normalized or assigned a color space. Applications must configure their transform to produce the range, channel semantics, and dimensions expected by the selected model.

The integration does not select a decoder, prescribe Java ImageIO, depend on Python/PyTorch/ComfyUI, allocate Aljabr tensors, or select a compute device. A Java ImageIO adapter or another decoder can be supplied independently.

Masks and video-frame sequencing are intentionally not treated as ordinary images; they require distinct semantic preprocessing contracts.
