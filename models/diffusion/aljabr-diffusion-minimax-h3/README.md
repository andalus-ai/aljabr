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

This artifact currently provides **family identification, component requirement profiles, and conditioning capability metadata**. It does not claim to load MiniMax-H3 checkpoints or execute generation. Model construction, modality-specific preprocessing, video/audio latent state, and pipeline execution remain outside this increment.

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


## Task component requirements

`MiniMaxH3PipelineRequirements` translates a task plus explicit runtime
conditioning/output choices into the family-neutral
`DiffusionComponentRequirement` contract. The generic
`DiffusionPipelineLoader` remains responsible for resolution, artifact
materialization, format loading and resource ownership.

```java
var requirements = MiniMaxH3PipelineRequirements.forTask(
    MiniMaxH3Task.TEXT_TO_VIDEO,
    MiniMaxH3PipelineRequirements.Conditioning.LOCAL_TEXT_ENCODER,
    MiniMaxH3PipelineRequirements.Output.VIDEO);

// Include this only when the requested result contains generated audio.
requirements = MiniMaxH3PipelineRequirements.withAudioOutput(requirements);

var loaded = pipelineLoader.load(manifest, requirements, resolutionContext);
try (loaded) {
    // Pass loaded components to the selected execution implementation.
}
```

The profile always requires the diffusion model. It selects either a local text
encoder or precomputed prompt embeddings, and requires the video autoencoder
only for video output. Audio decoding is opt-in. Reference images and frames
are request inputs, not checkpoint components in this role inventory; their
conditioning path remains a separate execution capability. These profiles
describe component prerequisites and do not claim that generation is
implemented.


## Request conditioning capabilities

The family module maps tasks onto the generic request-input descriptors in the
diffusion API:

| Task | Declared request inputs |
|---|---|
| Text-to-video | Prompt |
| Image-to-video | Prompt, image |
| Reference-to-video | Prompt, reference image |
| Multiframe reference-to-video | Prompt, reference image, ordered video frames |
| Text-to-image | Prompt |
| Image edit | Prompt, image |

Use `DiffusionConditioningSet` to pass ordered inputs, then validate the
declared modality compatibility:

```java
var conditioning = new DiffusionConditioningSet(List.of(
    new DiffusionConditioningInput("ref-0",
        DiffusionConditioningInput.Modality.REFERENCE_IMAGE,
        "file:///inputs/reference.png"),
    new DiffusionConditioningInput("frame-0", "file:///inputs/frame-0.png", 0),
    new DiffusionConditioningInput("frame-1", "file:///inputs/frame-1.png", 1)));

MiniMaxH3ConditioningCapabilities.validate(
    MiniMaxH3Task.MULTIFRAME_REFERENCE_TO_VIDEO, conditioning);
```

The capability check rejects unsupported modalities; it deliberately does not
infer cardinality, prompt requirements, frame sampling, image preprocessing,
or tensor layouts. Those belong to later execution and conditioning adapters.
