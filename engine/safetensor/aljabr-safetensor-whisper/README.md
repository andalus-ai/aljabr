# Aljabr Whisper Safetensors adapter

Optional storage adapter for Hugging Face Whisper checkpoints.

## Usage

Add `tech.kayys.aljabr:aljabr-safetensor-whisper` and open the model directory:

```java
try (var checkpoint = loader.open(Path.of("/models/whisper-large-v3"))) {
    var encoderConv1 = checkpoint.tensor("model.encoder.conv1.weight");
    // Consume the zero-copy view while the checkpoint remains open.
}
```

The shared `SafetensorShardLoader` handles single-file and indexed sharded
checkpoints. The adapter validates the Whisper architecture config before
opening weights and preserves the loader's resource lifetime semantics.

## Encoder parameter contract

`WhisperEncoderCheckpointMapper` in the model module defines the canonical
Hugging Face large-v3 encoder tensor names, logical Aljabr parameter names,
and exact shapes. It is storage-format independent and accepts F32, F16, and
BF16 metadata. The optional Safetensors adapter's
`WhisperSafetensorBindings` resolves those names to zero-copy tensor views
and validates the complete encoder contract before exposing the binding.

The encoder contract intentionally covers only `model.encoder.*`. Full
checkpoints also contain decoder and output-projection tensors; those are not
silently treated as encoder parameters.

## Scope

This is checkpoint validation and parameter binding, not yet an executable
Whisper model. Encoder/decoder graph construction, CPU numerical kernels,
KV cache, generation, and numerical conformance remain separate implementation
gates. Tensor views are owned by the open checkpoint and must not outlive it.
