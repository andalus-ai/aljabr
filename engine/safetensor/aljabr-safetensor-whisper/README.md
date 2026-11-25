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

## Scope

This module deliberately exposes original named Safetensors views; it does not
copy weights into a second representation or invent a Whisper-specific tensor
runtime. It is not yet an executable Whisper model. Encoder/decoder graph
construction, parameter-name-to-Aljabr-operation mapping, KV cache, generation,
and numerical conformance remain separate implementation gates.
