# BERT Safetensors adapter

This module connects the reusable BERT encoder model to Aljabr's shared Safetensors
shard loader. It supports single-file and sharded checkpoints and converts F32,
F16, and BF16 parameters to backend-owned F32 tensors.

The model module remains independent of Safetensors; checkpoint-name mapping and
shape validation live in `aljabr-model-bert`. This adapter only handles storage
and tensor materialization.

The loader expects canonical Hugging Face BERT encoder tensor names. It intentionally
loads only encoder parameters; unused pooler and task-head weights are not required.
