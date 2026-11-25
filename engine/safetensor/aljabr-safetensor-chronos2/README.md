# Chronos-2 Safetensors adapter

This module connects the Chronos-2 model-family contract to Aljabr's shared
Safetensors loader. It supports single-file and indexed multi-shard checkpoints.

## Ownership and zero-copy behavior

`Chronos2SafetensorLoader.open(path, config)` returns an
`AutoCloseable` checkpoint session. The adapter validates every required
tensor's checkpoint name, F32 dtype and shape, then retains the original
`SafetensorTensor` views. It does not convert payloads to Java arrays or
create a second tensor reader.

The checkpoint owns the underlying shard session. Tensor views returned by
`checkpoint.tensor(parameterName)` are valid only until the checkpoint is
closed. Keep the checkpoint open for as long as execution needs its weights.

## Scope

This is the storage binding layer, not a complete Chronos-2 execution engine.
The current float32 reference encoder and embedding projections still consume
array-based reference weights; a subsequent execution integration will add
backend-neutral tensor parameter views so kernels can consume these mapped
segments without materialization.
