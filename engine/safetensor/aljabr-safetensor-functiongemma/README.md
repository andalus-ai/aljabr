# FunctionGemma SafeTensors adapter

This module connects FunctionGemma checkpoint metadata and tokenizer policy to
Aljabr's shared SafeTensors loader and Gemma 3 text decoder. It does not
implement a separate tensor store, decoder, or function-call dispatcher.

## Load an executable model

```java
@Inject
FunctionGemmaSafetensorCheckpointLoader loader;

// loadModel materializes the required weights into detached float32 arrays.
// The memory-mapped SafeTensors session is closed before the method returns.
FunctionGemmaModel model = loader.loadModel(Path.of("/models/functiongemma-270m-it"));

long[] inputIds = model.tokenizer().encode("Who are you?", EncodeOptions.defaultOptions());
float[][] logits = model.forward(inputIds);
```

The returned `FunctionGemmaModel` exposes the checkpoint configuration,
FunctionGemma tokenizer, and the shared `Gemma3TextBackbone` reference path.
Its current execution contract is full-sequence prefill: one vocabulary-logit
vector per input token. It is a correctness/reference path, not an optimized
backend or incremental KV-cache session.

## Open a checkpoint without materializing model weights

```java
try (var checkpoint = loader.load(Path.of("/models/functiongemma-270m-it"))) {
    ModelConfig config = checkpoint.config();
    FunctionGemmaArchitecture architecture = checkpoint.architecture();
    FunctionGemmaTokenizer tokenizer = checkpoint.tokenizer();

    // Tensor views are backed by the open checkpoint session.
    SafetensorTensor embeddings = checkpoint.tensors().tensor(architecture.embedTokensWeight());
}
```

The lower-level `load` method:

1. Verifies the checkpoint's FunctionGemma metadata and control-token protocol.
2. Parses the Hugging Face model configuration.
3. Selects the FunctionGemma architecture adapter.
4. Loads the checkpoint tokenizer and validates its FunctionGemma token inventory.
5. Opens single-file or indexed multi-shard SafeTensors using the shared loader.
6. Checks the required Gemma 3 text projection and normalization tensors.

The returned `LoadedCheckpoint` owns the open mmap-backed tensor session.
Close it after tensor access; tensor views must not outlive it.

The executable-model path converts F32, F16, and BF16 SafeTensors weights to
the mapper's canonical float32 representation before closing the mapping.
Function/tool dispatch remains the responsibility of the consuming runtime,
which interprets and validates model-emitted function calls.
