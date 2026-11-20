# FunctionGemma SafeTensors adapter

This module connects the FunctionGemma model-family policy to Aljabr's shared
SafeTensors shard loader. It does not implement a separate storage engine or
execute generated function calls.

## Load a checkpoint

```java
@Inject
FunctionGemmaSafetensorCheckpointLoader loader;

try (var checkpoint = loader.load(Path.of("/models/functiongemma-270m-it"))) {
    ModelConfig config = checkpoint.config();
    FunctionGemmaArchitecture architecture = checkpoint.architecture();
    FunctionGemmaTokenizer tokenizer = checkpoint.tokenizer();

    // Tensor views are backed by the open checkpoint session.
    SafetensorTensor embeddings = checkpoint.tensors().tensor(architecture.embedTokensWeight());
    // Use config + architecture with the shared Gemma 3 text execution path.
}
```

The loader:

1. Verifies the checkpoint's FunctionGemma metadata and control-token protocol.
2. Parses the Hugging Face model configuration.
3. Selects the FunctionGemma architecture adapter.
4. Loads the checkpoint tokenizer and validates its FunctionGemma token inventory.
5. Opens single-file or indexed multi-shard SafeTensors using the shared loader.
6. Checks the required Gemma 3 text embedding, attention, FFN, and final-normalization tensors.

The returned `LoadedCheckpoint` owns the open mmap-backed tensor session. Close it
after inference to release mapped resources. Tensor views must not outlive it.

Function/tool dispatch is deliberately not part of this module; the consuming
runtime interprets and validates model-emitted function calls.
