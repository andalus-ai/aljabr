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

### Checkpoint configuration findings

The public checkpoint `google/embeddinggemma-2` currently declares:

- Top-level `model_type: embedding_gemma2` and `architectures: [EmbeddingGemma2Model]`.
- Text sub-config `model_type: embedding_gemma2_text`, 24 layers, hidden size 512, intermediate size 2048, four attention heads, two KV heads, vocabulary size 262,144, and BF16 dtype.
- Per-layer configuration overrides at layers 05, 11, 17, and 23, including a different head dimension/KV-head count. The implementation must read these per-layer overrides rather than flattening all layers to one attention shape.
- A mixed sequence of sliding-attention and full-attention layers, separate RoPE parameters for each attention type, and a configured sliding window of 512 in this checkpoint.
- Nested `vision_config` and `audio_config` sections, plus image/audio/video token IDs and the default 280 vision soft-token budget.

The config's `max_position_embeddings` value is 262,144, while the official model card documents an 8,192-token effective input context. Do not use the raw config field as the product input limit without reconciling the processor/model implementation and published model guidance.

The checkpoint uses the distinct model identifiers `embedding_gemma2` and `embedding_gemma2_text`; it is not registered as ordinary `gemma4` / `gemma4_text`. Aljabr's existing Gemma 4 family adapter contains relevant Gemma-family weight-name and PLE knowledge, but the EmbeddingGemma 2 family must have its own descriptor/config and explicit weight mapping. Do not alias it to the Gemma 4 causal-LM runner: the embedding model has a distinct architecture class, bidirectional attention behavior, and embedding head.

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

## Execution seam audit

The source audit confirms that the current Aljabr tree has model metadata and weight-loading contracts, but not yet a reusable embedding model runner:

| Existing source | What it provides | What it does not provide |
|---|---|---|
| `spi/aljabr-spi-model/.../ModelArchitecture.java` | Architecture identity and weight-name resolution, including Gemma-style per-layer input weight names | Model forward execution, bidirectional attention policy, embedding head, or pooling |
| `spi/aljabr-spi-model/.../ModelConfig.java` | Flat Hugging Face model configuration fields and a fallback that infers root model type/architecture from nested `text_config` | A typed nested text config or parsing of `per_layer_config` entries |
| `engine/safetensor/aljabr-safetensor-spi/.../SafetensorEngine.java` | Loaded-model lifecycle, raw weight map, tokenizer, flat config, and runtime traits | A forward/embedding execution method |
| `spi/aljabr-spi-inference/.../InferenceEngine.java` | Existing top-level `executeEmbedding(modelId, EmbeddingRequest)` invocation contract | A family-specific embedding runner contract or a concrete Aljabr implementation |
| `engine/gguf/.../GgufRunnerSession.java` | Format-specific generation session contract | A format-neutral embedding execution seam |

This matters for the EmbeddingGemma 2 checkpoint: its root model type is `embedding_gemma2`, but the transformer settings live under `text_config` with model type `embedding_gemma2_text`. The current `ModelConfig` only falls back to nested `text_config.model_type` and `text_config.architectures` when root identity fields are missing; it does not map the nested dimensions, attention layer types, RoPE settings, or per-layer overrides into typed configuration. A flat-config-only load would therefore lose essential model shape information.

### First SPI increment

The first executable contract increment is `EmbeddingModelRunner` in the existing `spi/aljabr-spi-inference` module:

- `modelFamily()` identifies the canonical family handled by a runner (for example, `embedding_gemma2`).
- `embed(EmbeddingRequest)` returns a JDK `CompletionStage<EmbeddingResponse>`, preserving input order.
- The runner owns model-specific forward execution and embedding post-processing.
- Model discovery, alias resolution, scheduling, admission, transport, and serving lifecycle remain outside this SPI.
- The contract deliberately uses JDK concurrency types rather than Mutiny so the SPI does not expose a reactive implementation dependency.

This is an execution seam, not a complete typed embedding API redesign. The existing request/response records remain the canonical contract for this increment. Typed task, modality, and output-dimension options will be introduced additively after their consumers and compatibility constraints are audited.

### Consumer compatibility audit (2026-11-18)

The cross-repository audit of andalus-ai/aljabr and andalus-ai/aqli found that the canonical Aljabr embedding contract is already bridged into AQLI, but AQLI still exposes a compatibility mirror to its callers.

| Consumer / boundary | Current contract | Finding and required treatment |
|---|---|---|
| Aljabr inference SPI | `tech.kayys.aljabr.spi.embedding.EmbeddingRequest/EmbeddingResponse` | Canonical model-execution DTOs. The request is a four-component record: request id, model, text inputs, and untyped parameters. |
| AQLI inference SPI | `tech.kayys.aqli.spi.embedding.EmbeddingRequest/EmbeddingResponse` | Structurally mirrors the Aljabr records. This is a compatibility copy, not a second model contract. |
| AQLI inference compatibility adapter | `AljabrInferenceDtoAdapter` | Performs a direct field-for-field conversion in both directions; no embedding-specific data is currently lost. Keep this adapter as the migration boundary while legacy AQLI implementations remain. |
| AQLI inference engine | `aqli.spi.inference.InferenceEngine` | Retains its legacy AQLI-DTO method and provides a default overload accepting Aljabr's canonical request/response. This is the intended direction of travel: Aljabr types at the model-execution boundary. |
| AQLI public SDK | `AqliSdk.createEmbedding(aqli.spi.embedding.EmbeddingRequest)` | Still exposes the AQLI mirror. SDK callers do not yet invoke the canonical Aljabr embedding DTO directly. Do not break this public surface as part of the model runner work. |
| AQLI REST API | `EmbeddingsResource` + `AgenticApiMapper` | Parses wire JSON into the AQLI compatibility DTO, calls the SDK, then maps the result to OpenAI-compatible or native response JSON. HTTP DTOs and trace/usage metadata remain AQLI concerns. |
| AQLI CLI | `EmbedCommand` | Builds the AQLI SPI request and invokes the SDK. Task and dimension CLI options must eventually map through the SDK contract, not bypass it. |
| AQLI multimodal service SPI | `EmbeddingService` / `MultimodalContent` | A higher-level multimodal service abstraction. It is not the same contract as the low-level Aljabr model-family runner and should not be folded into it. |
| AQLI legacy client SDK model | `aqli.sdk.model.EmbeddingRequest/EmbeddingResponse` | A separate single-input/single-vector convenience DTO used by the older client abstraction. Treat it as a legacy adapter surface, not a model-execution API. |
| Aljabr Safetensors HTTP resource | `EmbeddingsResource` nested request/response records | Transport DTOs and reflective access to a legacy `EmbeddingEngine`; not a consumer of the canonical SPI runner. It should not define model task semantics. |

#### Compatibility decisions

1. Keep `tech.kayys.aljabr.spi.embedding` as the only canonical model-execution embedding contract.
2. Do not add a second Aljabr embedding API/core module.
3. Do not add task/dimension fields only to the Aljabr record while leaving AQLI's mirror and conversion adapter unaware of them. That would silently lose options at the existing bridge.
4. The next typed-contract change must update the canonical Aljabr DTO, the AQLI compatibility DTO, and `AljabrInferenceDtoAdapter` together, with explicit tests proving round-trip preservation. Preserve the existing four-argument constructors and builder methods for compatibility.
5. Keep the AQLI public SDK, REST, and CLI as adapters during migration. A separate SDK API migration can later move the public signature to Aljabr types once downstream compatibility and release policy are confirmed.
6. Do not conflate the model-family runner with AQLI's multimodal service, old client DTO, or HTTP wire representations.

This audit also changes the order of Phase 1: before introducing typed task and output-dimension options, first define the shared value types in Aljabr and decide how the AQLI SDK's legacy request and the `aqli.spi` compatibility mirror map to them. A field-for-field adapter test is a release gate.

### Consequence for implementation

Do not create a model-family descriptor and call that a runner. Before executable model code is added, define the smallest embedding-specific execution seam that connects the existing `InferenceEngine.executeEmbedding` entry point to a family-specific implementation. Keep it in an existing Aljabr SPI module unless dependency analysis proves a new module is necessary.

That seam must not become another general serving engine. It should express only reusable embedding-model execution and model resource ownership; AQLI continues to own scheduling, admission, transport, and runtime policy. The design must also decide how nested model configuration and per-layer overrides are represented without breaking existing flat `ModelConfig` consumers.

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
