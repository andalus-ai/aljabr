# EmbeddingGemma 2 model family

This module registers the distinct `embedding_gemma2` family in Aljabr. It is
not an alias for the causal `gemma2` or `gemma4` modules.

## Current scope

- Model-family identity: `embedding_gemma2` and nested text type
  `embedding_gemma2_text`.
- SentencePiece tokenizer routing.
- Validated nested text configuration through `EmbeddingGemma2Config`.
- Preservation and validation of zero-padded per-layer overrides and
  attention-type-specific RoPE parameters.
- Resolution of effective per-layer attention geometry, mixed full/sliding
  attention, and attention-specific RoPE metadata.
- Text task formatting for search, question answering, fact checking, code
  retrieval, classification, clustering, and semantic similarity.
- Checkpoint-backed float32 Q/K/V/O projection primitives and a composed
  bidirectional attention sub-block with per-head RMSNorm, Q/K RoPE, GQA,
  and per-layer full/sliding attention geometry.
- A complete text encoder-layer reference operation with all four residual-path
  RMSNorms, gated GELU MLP, and the third PLE residual sub-block.
- A format-neutral `EmbeddingGemma2CheckpointMapper` that validates Hugging Face text tensor names and shapes and assembles the typed checkpoint bundle; storage-format decoding remains outside the model module.
- One canonical text-backbone reference path that owns vocabulary lookup,
  sqrt(hidden_size) embedding scaling, projection-only PLE, construction and
  execution of every encoder layer, final RMSNorm, token-wise embedding
  projection, and optional mask-aware mean pooling.
- Projection-only Per-Layer Embeddings (PLE) reference operations: input
  projection, per-layer reshape/RMSNorm, and gated residual injection.
- Matryoshka truncation and final L2 normalization for already-projected
  768-dimensional vectors.
- A Gemma4 vision patch embedder that applies the reference pixel scaling, bias-free RGB patch projection, and learned x/y positional-table lookup.
- `EmbeddingGemma2VisionImageProcessor` accepts decoded RGB, RGBA, grayscale, or BGR samples without an AWT or image-codec dependency, applies Gemma4's patch-budget aspect-ratio resize and bicubic antialias filter, rescales RGB to [0,1], extracts row-major patches, emits (x,y) position IDs, and pads each image to the configured common patch axis. It returns the immutable `EmbeddingGemma2VisionProcessorOutput` consumed by the composed tower.
- A Gemma4 vision spatial pooler that groups patch states by their 2D coordinates, applies fixed-area k×k averaging with zero-valued padding, returns a pooled padding mask, and scales pooled states by sqrt(hidden_size) in float32.
- `EmbeddingGemma2VisionSoftTokenStandardizer` applies the optional checkpoint-provided feature-wise `(x - std_bias) * std_scale` operation after spatial pooling and sqrt(hidden_size) scaling. It computes in float32; the caller casts to its working dtype after standardization and enables it only when `vision_config.standardize=true`.
- A configurable Gemma4 vision encoder stack that validates the checkpoint layer count, executes every vision encoder layer in order, supports either RGB patch vectors or precomputed patch embeddings, and returns unpooled final hidden states without adding a non-reference final normalization.
- A Gemma4 vision encoder-layer reference with the four checkpoint-backed RMSNorm scales, injected bidirectional attention operation, gated GELU MLP, and exact two-residual ordering. Gemma4-compatible axial 2D RoPE and a concrete float32 bidirectional vision attention operation are implemented. The attention path includes bias-free Q/K/V/O projections, Q/K learned RMSNorm, unscaled V RMSNorm, masked non-causal GQA, and no KV cache.
- A format-neutral `EmbeddingGemma2VisionCheckpointMapper` that validates the Hugging Face `vision_tower.*` state-dict namespace, patch/position tensors, every encoder layer's attention/MLP/norm weights, optional legacy `pooler.scale` metadata, and optional standardization vectors. It consumes format-decoded dense tensors and rejects unknown vision tensors. The legacy pooler tensor is retained only when present: the current Hugging Face Gemma4 vision pooler forward path does not consume it.
- `EmbeddingGemma2VisionTower` composes the patch embedder, ordered vision encoder layers, spatial pooler, and optional standardizer from mapped checkpoint weights. It removes padded pooled tokens before standardization, matching the Hugging Face forward path, and exposes a separate final operation for modality-to-text projection. Image preprocessing and patch extraction remain processor responsibilities.
- A format-neutral multimodal soft-token embedder that applies unscaled RMSNorm
  followed by the checkpoint-backed bias-free modality-to-text projection, while
  preserving per-image token sequences. The `EmbeddingGemma2MultimodalCheckpointMapper` maps
  the Hugging Face `embed_vision.embedding_projection.weight` tensor
  (output-major [text hidden][vision hidden]) and rejects unknown tensors in
  the image-bridge namespace. `EmbeddingGemma2VisionTower.fromCheckpoint`
  can receive this mapped embedder so `encodeAndProject` executes the
  complete vision-to-text soft-token path without a separately injected
  projection at call time. A separate splitter restores per-image boundaries
  from padded patch-position metadata and the configured pooling kernel.
- `EmbeddingGemma2VisionImagePipeline` composes the vision checkpoint mapper, multimodal projection mapper, Java image processor, executable vision tower, and image-to-text bridge from one named checkpoint tensor map. `fromCheckpoint(...)` accepts the vision spec, text hidden size, norm/RoPE settings, soft-token budget, and pooling kernel; `encode(image)` and `encodeBatch(images)` return separate text-hidden soft-token sequences per image.
- `EmbeddingGemma2VisionProcessorOutput` is an immutable adapter for Gemma4-compatible processor tensors (`pixel_values`, `image_position_ids`, and `num_soft_tokens_per_image`). `EmbeddingGemma2VisionTower.encodeAndProject(processorOutput)` executes each padded image sequence, checks the actual valid pooled-token count against processor metadata, applies the checkpoint-backed bridge, and returns one text-space soft-token sequence per image without flattening away image boundaries. Callers can now create this output directly with `new EmbeddingGemma2VisionImageProcessor(patchSize, maxSoftTokens, poolingKernelSize).process(rgbImage)` or process a list of images.
- Text and vision now have separate float32 reference paths. The image pipeline can compose decoded RGB inputs through checkpoint-backed vision weights into text-space image soft tokens; registered model-runner integration remains a separate milestone.

The attention, encoder-layer, and assembled text-backbone operations are
float32 references, not yet a complete registered model runner. The backbone
accepts token IDs and explicit position IDs; it does not perform tokenizer
preprocessing or checkpoint-file deserialization. The checkpoint mappers accept
format-decoded named tensors, validate the supported tensor sets, and create
typed weight bundles. Safetensors/GGUF decoding, tokenizer execution,
Matryoshka post-processing integration, and end-to-end runner registration
remain separate work items.

The plugin deliberately does **not** declare
`DIRECT_SAFETENSOR_INFERENCE`, and this module does not yet register an
`EmbeddingModelRunner`. A metadata/config module is not evidence of executable
inference. The runner must not be registered until the complete text path is
implemented and validated.

## Configuration contract

`EmbeddingGemma2Config.fromJson` requires the family-level
`embedding_gemma2` identity and nested `embedding_gemma2_text` identity.
It reads the checkpoint's `dtype` field (with `torch_dtype` compatibility),
base head dimension, embedding dimension, and both full/sliding RoPE parameter
sets. It validates positive dimensions, query/KV head divisibility, the
supported BF16/FP32 precision contract, layer-type count, and per-layer
override index ranges.

Checkpoint layer keys such as `"05"`, `"11"`, `"17"`, and `"23"` are
normalized to zero-based integer-string keys, so the runtime can reliably
resolve an override by layer index. The raw override objects and RoPE
parameters are defensively copied and retained; both the backbone and text
checkpoint mapper resolve the effective attention geometry per layer.

The documented input budget (8,192 tokens) is kept distinct from the raw
`max_position_embeddings` field.

## Text task formatting

Use `EmbeddingGemma2InputFormatter` with an explicit
`EmbeddingGemma2InputRole`. Asymmetric tasks intentionally use different
query and document templates. Document inputs use
`title: none | text: ...` when no title is supplied. Symmetric tasks use the
same task prefix on each side. `UNSPECIFIED` preserves raw input for legacy
callers.

The formatter does not tokenize or execute the model. Those remain separate
stages of the model runner.

## Vector post-processing

`EmbeddingGemma2VectorPostProcessor` accepts only the learned projection's
native 768-dimensional output. It supports the documented Matryoshka
dimensions (768, 512, 256, 128), truncates leading dimensions first, then
normalizes the retained vector using L2 normalization. It rejects non-finite,
zero-norm, malformed, and unsupported-dimension inputs rather than returning
invalid vectors.

This utility is separate from mask-aware mean pooling and the learned
512-to-768 projection.

## Required execution constraints

The eventual text runner must implement bidirectional mixed full/sliding
attention, projection-only PLE, mask-aware mean pooling, learned projection,
Matryoshka truncation/renormalization, and L2 normalization. The reference
operations and encoder-layer composition are not a registered model runner.

Image decoding and end-to-end registered runner execution remain deferred. The Java image processor accepts already-decoded RGB samples; codec integration remains the responsibility of the application or a separate image-I/O module. The bicubic antialias path now has a deterministic torchvision CPU reference fixture. The composed image pipeline has synthetic named-checkpoint integration coverage; full parity against a published EmbeddingGemma 2 checkpoint and registration in the model runner still require real-checkpoint validation. Current vision components
cover patch projection and learned 2D position lookup, the encoder tower and
spatial pooler, optional post-pool standardization, post-tower modality
projection, and reference per-image soft-token splitting.
Audio and video execution also remain deferred.
