# Aljabr MiniMax-H3 diffusion family

Optional family metadata and provider discovery for
[`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3).

## Capability declaration

The family module describes the task and component vocabulary without baking
ComfyUI filenames, quantization choices or hardware assumptions into the
family-neutral diffusion API.

Declared tasks:

- Text-to-video
- Image-to-video
- Reference-to-video
- Multiframe reference-to-video
- Text-to-image
- Image editing

Logical component roles:

- Diffusion model
- Qwen3-VL text encoder
- Video autoencoder
- Audio autoencoder
- Adapter / LoRA
- Model patch
- Prompt embedding

The provider can be discovered with Java `ServiceLoader` or supplied directly
to `DiffusionFamilyRegistry`. A descriptor may use the family and architecture
IDs `minimax-h3`; an optional `task` attribute must match one of the task IDs
declared by `MiniMaxH3Task`.

## Deliberate support boundary

This artifact currently provides **family identification and capability
metadata only**. It does not claim to load MiniMax-H3 checkpoints or execute
generation. The existing diffusion API does not yet define model construction,
multimodal conditioning, video/audio latent state, component loading or
pipeline execution contracts.

A complete native Aljabr implementation still requires, in dependency order:

1. Reusable component-manifest and checkpoint-role resolution over Aljabr's
   Safetensors APIs, including variant/quantization metadata.
2. The Qwen3-VL text/vision conditioning path and the MiniMax-H3-specific
   conditioning contract.
3. The MiniMax-H3 diffusion transformer architecture and verified weight
   mapping for supported FL2VA / REF2VA variants.
4. Video and audio autoencoder interfaces, temporal latent layout and decode.
5. A compatible flow/diffusion integration contract, deterministic noise and
   scheduler reference fixtures.
6. End-to-end parity tests against trusted reference tensors and generated
   media fixtures.

The family module must reuse Aljabr tensor, operator, memory, Safetensors and
backend contracts. It must not introduce a second compute backend, depend on
ComfyUI at runtime, or pull Python/PyTorch/LibTorch into native inference.

## Dependency

```kotlin
implementation("tech.kayys.aljabr:aljabr-diffusion-minimax-h3:0.1.0-SNAPSHOT")
```

The module is independently included by the existing nested diffusion project
discovery in `settings.gradle.kts`.
