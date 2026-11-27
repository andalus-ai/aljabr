# BERT Safetensors adapter

This module connects the reusable BERT encoder model to Aljabr's shared Safetensors
shard loader. It supports single-file and sharded checkpoints and converts F32,
F16, and BF16 parameters to backend-owned F32 tensors.

The model module remains independent of Safetensors; checkpoint-name mapping and
shape validation live in `aljabr-model-bert`. This adapter only handles storage
and tensor materialization.

The loader expects canonical Hugging Face BERT encoder tensor names. It intentionally
loads only encoder parameters; unused pooler and task-head weights are not required.

## MiniLM-L6-v2 bundle

This module also loads the Sentence Transformers bundle for
`sentence-transformers/all-MiniLM-L6-v2` from a local Hugging Face snapshot.
It contains no Python or PyTorch runtime dependency.

### Obtain the model files

Use the Hugging Face CLI to download the official repository into a local
directory (the CLI is only a download tool; Aljabr performs inference):

```bash
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/all-MiniLM-L6-v2
```

The bundle loader expects the repository's standard files:

- `config.json`
- `sentence_bert_config.json`
- `modules.json`
- `1_Pooling/config.json`
- `tokenizer.json`
- `model.safetensors` (or a standard sharded Safetensors checkpoint)

It validates the BERT architecture, Sentence Transformers module sequence,
mean-pooling configuration, sequence length, and tokenizer vocabulary before
loading weights. The supported profile is the official 384-dimensional,
six-layer MiniLM model with a 256-token sentence input limit.

### Load and encode

In a CDI-managed application, inject the bundle loader and pass the compute
backend selected by the application:

```java
@Inject
MiniLmL6V2BundleLoader miniLmLoader;

float[][] embeddings;
try (MiniLmL6V2Bundle bundle = miniLmLoader.load(
        Path.of("./models/all-MiniLM-L6-v2"), computeBackend)) {
    embeddings = bundle.encode(List.of(
            "That is a happy person",
            "That is a happy dog",
            "Today is a sunny day"));
}
// embeddings.length == 3
// embeddings[i].length == 384
```

The returned vectors are attention-mask-aware mean-pooled and L2-normalized.
Sequences are truncated to 256 tokens while preserving the SEP token; batches
are padded to their longest encoded sentence.

### Ownership and verification

- The caller owns `ComputeBackend`; closing the bundle closes the model's
  registered module/tensor state, not the backend.
- Keep the bundle open while encoding. Do not use it after its backend is closed.
- The adapter accepts F32, F16, and BF16 checkpoint tensors and copies weights
  into backend-owned model parameters before closing the mapped checkpoint session.
- The BERT encoder, checkpoint name mapper, tokenizer SPI, masked mean pooler,
  and L2 normalizer remain reusable and independent of the Safetensors format.

Unit tests cover model configuration, tokenizer behavior, encoder contracts,
checkpoint tensor-name/shape mapping, pooling, normalization, and input
validation. An official-checkpoint embedding parity run against Sentence
Transformers remains a separate integration-validation milestone; unit tests
are not numerical parity evidence.

References:
- [Official Hugging Face model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Sentence Transformers modular model architecture](https://sbert.net/docs/sentence_transformer/usage/custom_models.html)
