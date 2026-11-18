# EmbeddingGemma 2 — Safetensors adapter

This module connects the shared Aljabr Safetensors loader to the
format-neutral EmbeddingGemma 2 checkpoint mapper.

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
- Close the mapped shard session after the model's weight arrays are
  materialized.

The adapter does not implement the SafeTensors binary format, mmap, shard
indexing, tensor-name mapping, or model execution. It does not accept F16,
F64, FP8, or integer checkpoint weights for this model family.

## Usage

Construct this adapter with the shared `SafetensorShardLoader` and call
`load(checkpointPath, config)`. The returned text backbone owns detached
heap arrays and remains usable after the mapped session is closed.

The model-family module has no dependency on this adapter or on the
Safetensors loader, so the same checkpoint mapper can be reused by GGUF or
other future format adapters.
