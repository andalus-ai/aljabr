# EmbeddingGemma 2 model family

This module registers the distinct `embedding_gemma2` family in Aljabr. It is
not an alias for the causal `gemma2` or `gemma4` modules.

## Current scope

- Model-family identity: `embedding_gemma2` and nested text type
  `embedding_gemma2_text`.
- SentencePiece tokenizer routing.
- Validated nested text configuration through `EmbeddingGemma2Config`.
- Preservation and validation of per-layer configuration overrides.
- Text-only is the first intended execution milestone.

The plugin deliberately does **not** declare
`DIRECT_SAFETENSOR_INFERENCE`, and this module does not yet register an
`EmbeddingModelRunner`. A metadata/config module is not evidence of executable
inference. The runner must not be registered until the complete text path is
implemented and validated.

## Configuration contract

`EmbeddingGemma2Config.fromJson` requires the family-level
`embedding_gemma2` identity and nested `embedding_gemma2_text` identity.
It validates positive dimensions, query/KV head divisibility, the supported
BF16/FP32 precision contract, layer-type count, and per-layer override index
ranges. Raw per-layer JSON is preserved so layer-specific head dimensions,
KV-head counts, and RoPE settings are not lost while parsing the base text
configuration.

The documented input budget (8,192 tokens) is kept distinct from the raw
`max_position_embeddings` field. The runner must resolve per-layer overrides
before constructing attention projections or loading corresponding weights.

## Required execution constraints

The eventual text runner must implement bidirectional mixed full/sliding
attention, per-layer embeddings (PLE), mean pooling, learned projection,
Matryoshka truncation/renormalization, and L2 normalization.

Image, audio, and video execution remain deferred. They must not be inferred
from the broader model family's multimodal embedding-space description.
