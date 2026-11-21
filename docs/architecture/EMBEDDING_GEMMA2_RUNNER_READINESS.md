# EmbeddingGemma 2 runner readiness

**Status: reference components available; registered inference runner pending.**

This note distinguishes implemented model operations from a model that applications can actually load and execute through Aljabr's inference APIs.

## Implemented reference components

The `aljabr-model-embedding-gemma2` module contains:

- Text configuration validation and task formatting.
- Checkpoint tensor mappers for text, vision and multimodal projection weights.
- Float32 reference operations for the bidirectional text backbone.
- A Gemma4-compatible image processor, vision tower, spatial pooling and image-to-text soft-token projection.
- `EmbeddingGemma2VisionImagePipeline`, which accepts decoded image samples and returns text-hidden soft-token sequences.
- `EmbeddingGemma2MultimodalEncoder`, which combines projected image tokens with an expanded text prompt and executes the reference text backbone.

These components are useful foundations for parity work and runner integration. They do not, by themselves, constitute a loadable inference model.

## Explicit integration gaps

The model-family plugin currently reports:

- `supported_modalities=text`
- `deferred_modalities=image,audio,video`
- `runner_status=not_registered_until_forward_path_is_complete`

The module README also states that no `EmbeddingModelRunner` is registered. This conservative declaration must remain until the registered runtime path has been tested. Do not change the modality descriptor simply because a reference image pipeline exists.

Remaining work includes:

1. Complete tokenizer execution and prompt/image-placeholder expansion from the selected checkpoint's processor contract.
2. Complete checkpoint deserialization and connect decoded weights to the text and vision reference components.
3. Complete the model's embedding task path, including pooling, Matryoshka dimension selection and normalization, with exact model-defined behavior.
4. Implement and register an `EmbeddingModelRunner` through Aljabr's public inference SPI.
5. Validate the complete forward pass against a published reference implementation and real checkpoint tensors.
6. Verify text-query and image-document vectors inhabit the same trained embedding space. Matching dimensions are not sufficient.
7. Validate CPU and any claimed accelerator backend, memory ownership, cancellation and deterministic cleanup.
8. Only then advertise image support in the model-family descriptor.

## Java baseline

The model module currently configures a Java 25 toolchain. Downstream projects targeting Java 21 cannot consume a Java 25-compiled artifact. Before publishing this module as a reusable integration dependency, either maintain a compatible Java 21 API/artifact where feasible or document and coordinate an intentional ecosystem baseline change. Do not silently raise downstream requirements.

## Acceptance checklist

- [ ] Model family discovered by Aljabr's normal model registry.
- [ ] A real checkpoint can be loaded by the supported format loader.
- [ ] Registered runner is selected through public inference APIs.
- [ ] Text-only embeddings match reference outputs within a documented tolerance.
- [ ] Image embeddings match reference outputs within a documented tolerance.
- [ ] Text-to-image retrieval is validated using paired examples from one checkpoint.
- [ ] Unsupported modalities and missing runtime capabilities fail explicitly.
- [ ] Model metadata reports only validated modality support.
- [ ] Java baseline and published dependency coordinates are documented.
- [ ] Tests run in CI on the supported JDK and backend matrix.

## Consumer guidance

AQLI should consume the registered Aljabr runner through public inference APIs; it should not reimplement EmbeddingGemma 2 tensor operations. Pinisi should consume an optional AQLI provider or a deliberately supported embedded Aljabr adapter, and should not depend on Aljabr's internal model implementation packages.

Related:
- [EmbeddingGemma 2 module README](../../models/aljabr-model-embedding-gemma2/README.md)
- [AQLI multimodal embedding integration contract](https://github.com/andalus-ai/aqli/blob/main/docs/architecture/MULTIMODAL_EMBEDDING_INTEGRATION.md)
- [Pinisi integration gate](https://github.com/andalus-ai/pinisi/blob/main/docs/implementation/ALJABR_AQLI_INTEGRATION.md)
