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
FunctionGemma tokenizer, and shared `Gemma3TextBackbone` execution path.
It supports full-sequence logits and greedy incremental generation through
Aljabr's shared KV-cache session. This is a CPU correctness/reference path,
not yet an optimized accelerator backend.


## Generate from a structured conversation

The model uses the registered FunctionGemma chat template when passed SPI
messages. Tool declarations can be built with the typed Java API and included in the developer message:

```java
FunctionGemmaChatTemplate template = new FunctionGemmaChatTemplate();
var getWeather = FunctionGemmaToolDeclaration.of(
    "get_weather",
    "Get current weather.",
    FunctionGemmaToolDeclaration.Schema.object()
        .property("location", FunctionGemmaToolDeclaration.Schema.string()
            .description("City or place").minLength(2))
        .property("units", FunctionGemmaToolDeclaration.Schema.string()
            .enumValues("celsius", "fahrenheit"))
        .required("location")
        .additionalProperties(false));

String instructions = template.formatDeveloperContent(
    "You are an assistant that can call functions.",
    getWeather);

var result = model.generate(List.of(
    Message.system(instructions),
    Message.user("What is the weather in Bandung?")
), 96);

List<FunctionGemmaToolCallParser.FunctionCall> calls = model.parseFunctionCalls(result);
```

The parser extracts one or more native call blocks, including escaped string
values, booleans, numbers, nulls, nested objects, and arrays. It preserves
call order. Parsed arguments are untrusted model output: validate them against
your registered tool schema and authorization policy before dispatch. The
`FunctionGemmaToolSchemaValidator` checks required and optional properties,
nested objects and arrays, primitive types, enums, constants, numeric bounds,
string constraints, and unexpected properties when `additionalProperties`
is false. Its diagnostics identify the failing argument path. Aljabr does not
execute tools from this model adapter.

For direct prompting, `generate(String, int)` accepts an already formatted
prompt. The model stops on EOS or the FunctionGemma function-response marker;
the returned `GenerationResult` includes generated token IDs, decoded text,
and the stop reason.

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
