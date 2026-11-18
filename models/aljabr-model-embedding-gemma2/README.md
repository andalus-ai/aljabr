# EmbeddingGemma 2 model family

This module registers the distinct `embedding_gemma2` family in Aljabr. It is
not an alias for the causal `gemma2` or `gemma4` families.

## Current scope

- Model-family identity: `embedding_gemma2` and nested text type
  `embedding_gemma2_text`.
- Architecture metadata: `EmbeddingGemma2Model`.
- SentencePiece tokenizer routing.
- Text-only is the first intended execution milestone.

The plugin deliberately does **not** declare
`DIRECT_SAFETENSOR_INFERENCE`, and this module does not yet register an
`EmbeddingModelRunner`. A metadata plugin is not evidence of executable
inference. The runner must not be registered until the complete text path is
implemented, including bidirectional mixed attention, per-layer embeddings,
mean pooling, learned projection, output-dimension truncation, and L2
normalization.

## Required execution constraints

The eventual text runner must validate the nested text configuration and
per-layer overrides rather than assume a uniform Gemma 4 causal decoder. The
supported initial contract is:

- 24 transformer layers, hidden size 512, intermediate size 2048;
- bidirectional attention with the checkpoint's full/sliding layer schedule,
  RoPE settings, and per-layer attention overrides;
- required per-layer embedding (PLE) projection, normalization, gating, and
  residual contribution;
- mean pooling, learned 512-to-768 projection, then L2 normalization;
- output dimensions 768, 512, 256, or 128, with renormalization after
  truncation;
- maximum input length 8192 tokens;
- BF16 or FP32 only; do not advertise FP16.

Image, audio, and video execution remain deferred. They must not be inferred
from the broader model family's multimodal embedding-space description.

The model-family project is auto-included by the root Gradle settings because
it is located under `models/aljabr-model-*`.
