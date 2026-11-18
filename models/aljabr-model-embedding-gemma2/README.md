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
- A Gemma4 vision patch embedder that applies the reference pixel scaling, bias-free RGB patch projection, and learned x/y positional-table lookup. Image preprocessing and patch extraction remain processor responsibilities.
- A format-neutral multimodal soft-token embedder that applies unscaled RMSNorm
  followed by the checkpoint-backed bias-free modality-to-text projection, while
  preserving per-image token sequences. A separate splitter restores per-image
  boundaries from padded patch-position metadata and the configured pooling kernel.
- Text-only is the first intended execution milestone.

The attention, encoder-layer, and assembled text-backbone operations are
float32 references, not yet a complete registered model runner. The backbone
accepts token IDs and explicit position IDs; it does not perform tokenizer
preprocessing or checkpoint-file deserialization. The checkpoint mapper accepts
format-decoded named tensors, validates the complete text tensor set, and creates
the typed weight bundle. Safetensors/GGUF decoding, tokenizer execution,
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
parameters are defensively copied and retained; both the backbone and checkpoint
mapper resolve the effective attention geometry per layer.

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

The Gemma 4 vision tower, image preprocessing/patchification, vision checkpoint
mapping, and end-to-end image execution remain deferred. Current vision components cover checkpoint-backed patch projection and learned 2D position lookup, plus post-tower projection and reference per-image soft-token splitting. The bidirectional vision transformer, 2D RoPE, pooling, and standardization still remain to be implemented.
Audio and video execution also remain deferred.
