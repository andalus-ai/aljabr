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


### Opt-in official-checkpoint parity test

An integration test loads the official local checkpoint through the Java
Safetensors, tokenizer, encoder, pooling, and normalization path, then compares
its embeddings with independently generated Sentence Transformers reference
vectors. It is skipped in ordinary CI unless both environment variables below
are set. The model and reference files are not downloaded by the test.

Generate an independent reference file with the official Python package. This
is a reference-generation tool only, not an Aljabr runtime dependency:

```python
import json
from sentence_transformers import SentenceTransformer

sentences = [
    "That is a happy person",
    "That is a happy dog",
    "Today is a sunny day",
    "The weather is pleasant outside.",
]
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(
    sentences, normalize_embeddings=True, convert_to_numpy=True
).tolist()
with open("minilm-reference.json", "w", encoding="utf-8") as output:
    json.dump({"sentences": sentences, "embeddings": embeddings}, output)
```

Run the test with JDK 25:

```bash
ALJABR_MINILM_MODEL_DIR=./models/all-MiniLM-L6-v2 \
ALJABR_MINILM_REFERENCE_FILE=./minilm-reference.json \
./gradlew :engine:safetensor:aljabr-safetensor-bert:test \
  --tests '*MiniLmOfficialCheckpointParityTest'
```

Default thresholds are cosine similarity >= 0.999 and maximum absolute error
<= 0.005. They can be overridden with `ALJABR_MINILM_MIN_COSINE` and
`ALJABR_MINILM_MAX_ABS_ERROR`. A passing run is the numerical parity evidence;
the presence of this harness alone does not imply that parity has already
been measured.

References:
- [Official Hugging Face model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Sentence Transformers modular model architecture](https://sbert.net/docs/sentence_transformer/usage/custom_models.html)
