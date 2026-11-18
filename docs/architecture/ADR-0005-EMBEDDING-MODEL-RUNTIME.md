# ADR-0005: Embedding Model Runtime and EmbeddingGemma 2

- **Status:** Proposed
- **Date:** 2026-11-18
- **Scope:** Aljabr embedding contracts, model-family integration, reusable embedding execution, and AQLI serving integration

## Context

The goal is to add a real EmbeddingGemma 2 model runner to Aljabr without creating a second embedding framework or moving model execution into AQLI.

The current source tree already contains several embedding-related pieces:

| Existing source | Verified responsibility | Gap / caution |
|---|---|---|
| `spi/aljabr-spi-inference/.../embedding/EmbeddingRequest.java` | Aljabr-owned request record with model id, text inputs, and an untyped parameter map | Text-only; no typed task, modality, truncation, or output-dimension contract |
| `spi/aljabr-spi-inference/.../embedding/EmbeddingResponse.java` | Aljabr-owned response record with `List<float[]>`, dimension, and metadata | No explicit per-item result/index or normalization semantics |
| `spi/aljabr-spi-model/.../ModelFamilyCapability.java` | Already declares `EMBEDDING` as a model-family capability | Capability declaration alone does not prove an executable runner exists |
| `models/aljabr-model-jina-embeddings-v3` | Model-family plugin and tokenizer metadata for Jina Embeddings v3 | Its descriptor explicitly says metadata-only and direct Safetensors runtime is pending |
| `models/aljabr-model-poolformer` | Declares vision and embedding capabilities | Its descriptor explicitly marks direct runtime as pending |
| `engine/safetensor/aljabr-safetensor-api/.../EmbeddingsResource.java` | Existing OpenAI-compatible HTTP resource | Uses reflection to call an `EmbeddingEngine`; this is an adapter/resource, not a reusable model execution contract |

The current Gradle settings discover model-family projects under `models/aljabr-model-*`, while the canonical embedding request and response are already in `spi/aljabr-spi-inference`. There is no existing `aljabr-model-embedding-api` or `aljabr-model-embedding-core` project in the audited tree, and no general `ModelRunner` interface was found at the expected SPI model paths.

The earlier idea of creating separate `aljabr-model-embedding-api` and `aljabr-model-embedding-core` modules is therefore not accepted as an immediate implementation step. First evolve and reuse the contracts that already exist. Add a separate module only if implementation proves that a stable, independently reusable embedding abstraction cannot reasonably live in the existing inference/model SPI modules.

## Decision

### 1. Aljabr owns the canonical embedding model contract and computation

Aljabr owns:

- Model-family identity, architecture metadata, and embedding capability declarations.
- Typed model-execution input semantics that are reusable outside a particular serving product.
- Tokenization and model-specific input preparation required by the model.
- Embedding model graph/forward execution and reusable tensor operations.
- Pooling, projection, dimensionality selection, and normalization when required by the model.
- Embedding output values and execution metadata.
- Format adapters and backend integration for supported weight formats and devices.

The existing Aljabr embedding request/response types are the starting point. Audit their consumers and compatibility requirements before changing their public shape. Do not create another generic embedding request/response pair in a new module.

### 2. AQLI owns serving and request orchestration

AQLI owns:

- Public SDK capability invocation and runtime command semantics.
- Runtime model aliases, model loading policy, provider selection, and fallback policy.
- Admission, batching/scheduling policy, concurrency, cancellation, and request lifecycle.
- REST, gRPC, CLI, and external wire DTOs.
- OpenAI-compatible response formatting and serving-level usage accounting.
- Operational metrics, deployment, and runtime configuration.

AQLI maps its runtime/transport request to Aljabr's canonical embedding contract and invokes the Aljabr model execution capability. It must not implement EmbeddingGemma 2 tokenization, transformer layers, pooling, projection, or normalization itself.

### 3. Keep transport adapters separate from model execution

The existing `EmbeddingsResource` is not the model runner. Its reflection-based lookup and HTTP DTOs must not become the reusable embedding API. AQLI's SDK/REST/gRPC/CLI adapters should converge on the same AQLI capability invocation behavior, then delegate model computation to Aljabr.

### 4. Do not assume capability metadata means runtime support

A model-family plugin may declare `EMBEDDING`, but the runtime must report the actual readiness of:
- weight format and loader;
- tokenizer and input preprocessing;
- architecture/forward implementation;
- pooling/projection/normalization path;
- backend and dtype combination;
- supported task and modality.

Metadata-only model plugins must remain explicitly non-executable until the complete path is validated.

### 5. Keep the first runner milestone narrow and evidence-based

Implement the smallest end-to-end EmbeddingGemma 2 path supported by the paper, model configuration, tokenizer assets, and reference implementation. Start with text input only if the model's documented architecture and checkpoint permit a genuinely correct text-only path. Do not silently approximate or omit required components.

Multimodal input support (image, audio, video, or interleaved inputs, if required by the selected model/checkpoint) is a separate milestone. The initial runner must not claim those modalities are supported until their preprocessing, encoders, fusion, and correctness tests exist.

Task prefixes/instructions, output dimensions, pooling, projection, normalization, maximum sequence length, and special-token behavior must be derived from the authoritative model configuration/model card/reference implementation and encoded as tested model behavior—not hard-coded into AQLI.

## Verified EmbeddingGemma 2 execution requirements

The paper and official model card establish the following constraints. These are implementation requirements, not optional optimizations:

| Area | Verified behavior | Implementation consequence |
|---|---|---|
| Text backbone | 24-layer Gemma 4-derived transformer with **bidirectional attention**, model dimension 512, hidden dimension 2048 | A normal causal Gemma 4 generation runner is not a drop-in implementation; the attention mask and forward path must be embedding-model correct |
| Per-layer embeddings | Each layer receives a learned projected/RMS-normalized path from the input embeddings, gated from the layer output and added residually | The PLE path must be implemented and validated; omitting it is an architecture mismatch |
| Pooling/head | Mean-pool final token states, apply learned 512→768 linear projection, then L2-normalize | This is part of model inference, not a generic postprocessing option that may be skipped |
| Output dimensions | Native 768; Matryoshka dimensions 512, 256, and 128 | Truncation is supported only at these dimensions; re-normalize after truncation |
| Context | Up to 8,192 tokens shared across modalities | Enforce the shared context budget and test truncation/padding behavior |
| Precision | Use BF16 or FP32; FP16 can produce NaNs or silently degraded vectors | Do not advertise FP16 execution as supported; default to BF16 only when the backend is verified, otherwise FP32 |
| Modular encoders | Text-only model is approximately 270/271M parameters; vision and audio encoders are separately loadable | Text-only execution is a legitimate first milestone, but must use the actual text-only configuration/checkpoint semantics |
| Modalities | Text, image, audio, video, and ordered/interleaved combinations share one embedding space | Full model support requires modality-specific encoders, input processors, delimiters, position alignment, and token-budget accounting |

### Task instruction behavior

The official model card defines seven text task families. The model API should represent task intent and query/document role explicitly rather than expose a free-form prompt-prefix string as its only abstraction.

| Task | Query/input instruction | Target/document instruction |
|---|---|---|
| Web/document search | `task: search result | query: {query}` | `title: {title} | text: {content}` |
| Question answering | `task: question answering | query: {question}` | `title: {title} | text: {passage}` |
| Fact checking | `task: fact checking | query: {claim}` | `title: {title} | text: {evidence}` |
| Code search | `task: code retrieval | query: {query}` | `title: {title or filename} | text: {code}` |
| Classification | `task: classification | query: {content}` | Same instruction |
| Clustering | `task: clustering | query: {content}` | Same instruction |
| Semantic similarity | `task: sentence similarity | query: {content}` | Same instruction |

For asymmetric retrieval tasks, query and document instructions differ. A document without a title uses `title: none`. Symmetric tasks apply the same instruction to both inputs. Prefixes apply to text only; media inputs are not prefixed.

### Multimodal processing constraints

- Images are converted to vision soft tokens, with a configurable token budget (70, 140, 280, 560, or 1120); the documented default is 280 tokens per image.
- Audio input is mono 16 kHz and is encoded in 40 ms chunks, producing 25 soft tokens per second.
- Video defaults to 1 frame per second. Frames are processed by the vision encoder and may be interleaved with audio segments and timestamps.
- The text sequence uses modality placeholders such as `<|image|>`, `<|video|>`, and `<|audio|>`; the processor must align placeholders with supplied media in order.
- The shared context budget is 8,192 tokens, with documented accounting rates of 280 tokens/image, 140 tokens/video frame, and 25 tokens/audio second at the default settings.

### Authoritative references

- [EmbeddingGemma 2 technical report (arXiv:2610.37364)](https://arxiv.org/html/2610.37364v2)
- [Google DeepMind official model card](https://gemma.ai/models/embeddinggemma/model-card-v2/)
- [Official Hugging Face checkpoint: google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)

The paper reports 744M total parameters (271M text backbone/embedder, 167M vision, 305M audio); the official model card rounds these figures to 740M (270M, 170M, and 300M). Use checkpoint tensor/config metadata as the final source of truth for exact parameter and tensor-shape validation.

## Target responsibility map

| Concern | Owner | Notes |
|---|---|---|
| Canonical embedding request/result semantics | Aljabr inference SPI | Evolve existing `EmbeddingRequest` / `EmbeddingResponse` after consumer audit |
| Model-family descriptor and capability | Aljabr model SPI + model-family plugin | Declaration and support report only; no serving policy |
| EmbeddingGemma 2 architecture and model-specific execution | Aljabr model/runtime implementation | Reuse tensor, tokenizer, model, format, and backend contracts |
| Generic reusable pooling/projection/normalization primitives | Aljabr ML/tensor modules | Add only where existing primitives are insufficient |
| Weight loading and tensor access | Aljabr format engines | Keep Safetensors/other formats distinct |
| Runtime model selection, batching, scheduling | AQLI | Runtime policy, not model math |
| SDK/CLI/REST/gRPC invocation semantics | AQLI SDK | Adapters share the SDK capability behavior |
| Wire DTOs and OpenAI-compatible response mapping | AQLI adapters | Do not leak transport types into Aljabr model APIs |

## Implementation sequence

### Phase 0 — source and reference audit (this ADR)

- Confirm existing embedding request/response contracts and all consumers.
- Inspect model-family plugin conventions and the actual Gradle module layout.
- Inspect EmbeddingGemma 2 paper, model card, configuration, tokenizer, checkpoint tensor names/shapes, and reference implementation.
- Produce a component/weight map and identify reusable Aljabr primitives versus missing execution capabilities.
- Confirm supported checkpoint formats and the first target backend.

### Phase 1 — canonical contract alignment

- Inventory every Aljabr and AQLI consumer of the existing embedding request/response types.
- Define typed task/instruction, input, output-dimension, and result semantics only where required by the model and downstream use cases.
- Preserve existing source/binary and wire compatibility through explicit adapters or additive evolution.
- Keep AQLI runtime command/transport types separate from Aljabr's model-execution contract.
- Add contract tests for validation, batching order, dimensions, and normalization metadata where applicable.

### Phase 2 — model-family registration and artifact metadata

- Add an EmbeddingGemma 2 model-family plugin using the existing `ModelFamilyPlugin` extension point.
- Declare only verified model types, architectures, tokenizer requirements, and capabilities.
- Add deterministic fixtures for configuration/tokenizer metadata and weight-name/shape validation.
- Do not label the family direct-Safetensors-ready until its complete execution path is implemented.

### Phase 3 — text embedding execution

- Implement model-specific text preprocessing/tokenization using the existing tokenizer SPI where compatible.
- Implement the exact architecture and forward path using Aljabr tensor/model primitives.
- Implement the documented pooling, projection, output-dimension, and normalization behavior.
- Add reference-output tests for tokenization, intermediate shapes, embedding dimensions, and numerical tolerances.
- Validate batch behavior and edge cases (empty input, truncation, padding, special tokens, and invalid task settings).

### Phase 4 — format/backend conformance

- Validate each supported format independently; a loader being able to parse weights is not proof that the model executes correctly.
- Validate CPU correctness first, then add accelerator support through existing backend contracts.
- Record dtype, quantization, backend, and hardware coverage.
- Do not enable automatic provider selection for unvalidated combinations.

### Phase 5 — AQLI integration

- Add/extend the AQLI embedding capability implementation to map SDK/runtime commands to the Aljabr embedding contract.
- Make CLI, REST, and gRPC adapters invoke the same SDK capability semantics.
- Keep batching/scheduling, cancellation, streaming (if applicable), and usage accounting in AQLI.
- Add cross-adapter contract tests proving identical model behavior and response semantics.

### Phase 6 — multimodal expansion

- Add modality-specific preprocessing/encoders and fusion only after the text path is correct.
- Add per-modality fixtures and reference outputs.
- Advertise each modality independently based on tested readiness.

## Acceptance criteria

- No duplicate generic embedding API/core module is introduced without a demonstrated contract gap.
- Aljabr builds and executes the embedding model without an AQLI checkout or dependency.
- AQLI owns invocation orchestration and serving policy while delegating all model computation to Aljabr.
- Model-family metadata does not overstate executable capabilities.
- Embedding outputs match the authoritative reference within documented tolerances for every claimed configuration.
- Task instructions, pooling, projection, normalization, and output dimensions are covered by tests.
- Unsupported modalities, formats, dtypes, and backend combinations fail explicitly rather than returning plausible but incorrect vectors.
- CLI, REST, and gRPC share the same AQLI SDK capability behavior.
- No hardware-specific test is claimed unless it has actually been run on that hardware.

## Explicit non-goals

- Moving embedding model execution into AQLI.
- Creating a second Aljabr embedding framework alongside the existing inference SPI.
- Treating the current Safetensors HTTP resource as a reusable model runner.
- Claiming full multimodal support from model-family metadata alone.
- Replacing or merging the GGUF and Safetensors format engines.
