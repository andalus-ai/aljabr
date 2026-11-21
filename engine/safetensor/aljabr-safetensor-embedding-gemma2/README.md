# EmbeddingGemma 2 — Safetensors adapter

This module connects the shared Aljabr Safetensors loader to the
format-neutral EmbeddingGemma 2 checkpoint mapper and assembles a complete
text-only model bundle.

## Responsibilities

- Open single-file or indexed multi-shard checkpoints through
  `SafetensorShardLoader`.
- Select only `language_model.*` tensors from the composite checkpoint.
- Enforce the configured checkpoint precision: BF16 for `bfloat16` configs,
  F32 for `float32` configs.
- Convert BF16 and F32 tensors to detached float32 arrays using the shared
  loader's typed tensor accessors.
- Delegate tensor-name and shape validation to
  `EmbeddingGemma2CheckpointMapper`.
- Load the Hugging Face `config.json` and shared `tokenizer.json` from a model
  directory, then compose the config, tokenizer adapter, and text backbone as
  one `EmbeddingGemma2TextModel`.
- Close mapped checkpoint storage after model weights are materialized.

The adapter does not implement the SafeTensors binary format, mmap, shard
indexing, tokenizer algorithms, tensor-name mapping, or model execution.

## Usage

Construct `EmbeddingGemma2ModelBundleLoader` with the shared
`ObjectMapper` and `SafetensorShardLoader`, then call
`load(modelDirectory)`. The directory must contain `config.json`,
`tokenizer.json` (or `tokenizer/tokenizer.json`), and a single-file or
indexed Safetensors checkpoint.

The returned `EmbeddingGemma2TextModel` exposes an end-to-end text embedding
operation: task/role formatting, shared tokenization, BOS insertion,
bidirectional text encoding, mask-aware mean pooling, Matryoshka truncation,
and final L2 normalization. It supports native 768-dimensional vectors and
the documented 512, 256, and 128-dimensional outputs.

This is a text-only model bundle. Image, audio, and video execution and
registered inference-runner integration remain separate milestones. The
model-family module has no dependency on this adapter or on the Safetensors
loader, so the same text model API can be assembled by future format adapters.
