# Phase 0.2 — Implementation Boundary Audit

- **Status:** Initial static audit; findings require targeted consumer/test validation before code movement.
- **Audit date:** 2026-11-21
- **Repository baseline:** `main`, tree `ddb3387fc024698d2f40d7602e29095e36699b33`
- **Scope:** Gradle project discovery, root build conventions, inference/model SPIs, format API modules, selected provider implementations and architecture documentation.

This is an evidence log and prioritization document, not authorization to delete public APIs or move classes without a compatibility and consumer audit. The normative ownership rules remain in [AGENTS.md](../../AGENTS.md), the [Unified Architecture](ALJABR_UNIFIED_ARCHITECTURE.md), and the accepted ADRs.

## 1. Repository structure snapshot

The recursive Git tree at the baseline contains 7,461 entries (3,143 files and 4,318 directories) and 318 Gradle build files, including the root build and buildSrc. The repository uses directory-driven project discovery in `settings.gradle.kts` for most project groups, while model-family projects are discovered separately.

Top-level project groups include `core/`, `spi/`, `engine/`, `integration/`, `plugins/`, `backend/`, `models/`, and `examples/`.

This scale is not itself a defect. In particular, independently installable model-family modules can be justified when they keep dependencies opt-in. The audit must evaluate actual dependency edges, published artifacts, consumer value and repeated implementation—not optimize for a smaller module count.

## 2. Confirmed findings

### F-001 — Root architecture document contradicted accepted ownership and the live project graph

**Severity:** High · **Type:** Documentation / source-of-truth conflict

The former root `ARCHITECTURE.md` proposed moving request batching/scheduling and serving-oriented inference contracts into Aljabr, described obsolete module paths, and identified the tensor module as the backend contract owner. This conflicted with the accepted inference-ownership ADR and current compute SPI project layout.

**Action:** Retain the root path for link compatibility, but make it a navigation page pointing to the unified architecture and focused ADRs. Detailed architecture rules must have one authoritative home.

**Tracking:** PR [#922](https://github.com/andalus-ai/aljabr/pull/922).

### F-002 — The inference SPI contains request-serving scheduling contracts

**Severity:** High · **Type:** Ownership / public API boundary

Observed files:

- `spi/aljabr-spi-inference/src/main/java/tech/kayys/aljabr/spi/batch/BatchScheduler.java`
- `spi/aljabr-spi-inference/src/main/java/tech/kayys/aljabr/spi/batch/BatchConfig.java`
- `spi/aljabr-spi-inference/src/main/java/tech/kayys/aljabr/spi/batch/BatchStrategy.java`

`BatchScheduler` explicitly accepts inference requests, queues and flushes them, and changes STATIC/DYNAMIC/CONTINUOUS scheduling configuration. `BatchConfig` owns queue wait time, maximum concurrent batches and prefill batch policy. These are request-serving scheduling concerns, which [ADR-0002](ADR-0002-INFERENCE-CONTRACT-OWNERSHIP.md) assigns to AQLI.

**Direction:** Audit all implementations, imports, published consumers and tests. Keep reusable tensor batching/operation scheduling distinct from request admission, queueing and continuous-batching policy. Do not move or delete these public types until the compatibility path and AQLI consumers are proven.

### F-003 — The generic inference engine SPI mixes execution contracts with runtime lifecycle and serving operations

**Severity:** High · **Type:** Contract cohesion / public API boundary

Observed file:

- `spi/aljabr-spi-inference/src/main/java/tech/kayys/aljabr/spi/inference/InferenceEngine.java`

The interface combines inference execution and streaming with model selection, asynchronous job submission, initialization/shutdown, health, and runtime statistics. Some of these may be appropriate for a runtime-facing facade, but they should not automatically define the minimum reusable model-execution contract of a computational framework.

**Direction:** Classify each method by owner and consumer. Preserve reusable model execution/stream event semantics where justified; keep serving job IDs, service lifecycle and serving-level health/metrics in AQLI. Do not create a replacement interface until the existing call graph demonstrates the smallest missing contract.

### F-004 — Inference SPI contains application identity/authentication semantics

**Severity:** High · **Type:** Separation of concerns / contract leakage

Observed file:

- `spi/aljabr-spi-inference/src/main/java/tech/kayys/aljabr/spi/inference/StreamingSession.java`

The record carries a `RequestContext` and exposes `apiKey()`, including a fallback to `ApiKeyConstants.COMMUNITY_API_KEY`. API-key defaults and caller identity are application/security policy, not reusable tensor or model-computation semantics.

**Direction:** Trace all consumers and wire-compatibility requirements. Keep authentication and API-key policy in the consuming runtime; do not silently change current defaults in a public API migration.

### F-005 — Safetensors format API module contains HTTP resources and RAG/control-plane workflows

**Severity:** Critical · **Type:** Incorrect placement / product boundary / dependency pollution

Observed module:

- `engine/safetensor/aljabr-safetensor-api`

Observed classes include:

- `InferenceApiResource` — neutral inference HTTP resource exposing chat/completion/model operations and SSE streaming. Its Javadoc and OpenAPI tag explicitly state that the wire contract is OpenAI-compatible; that phrase documents protocol interoperability only and is not part of the Java type name.
- `EmbeddingsResource` — HTTP embeddings endpoint.
- `ModelRouterResource` — REST control-plane endpoints for primary/secondary model selection, canary mode and promotion.
- `RagResource` and `RagPipeline` — document ingest, retrieval and answer-generation workflow API.

The module's build declares JAX-RS, CDI, MicroProfile Config/OpenAPI and Quarkus REST dependencies. Several resources use reflection to reach engine/RAG objects and maintain their own model registries or runtime policy. These are not Safetensors format contracts or reusable weight-loading primitives.

**Additional source and consumer evidence (follow-up audit, 2026-11-21):**

- `aljabr-safetensor-api/build.gradle.kts` directly declares Mutiny, MicroProfile Config/OpenAPI, JAX-RS, CDI, and Quarkus REST/Jackson dependencies. A nominal Safetensors API dependency therefore brings in an HTTP/CDI stack.
- The package contains seven transport resources: `OpenAiCompatibleResource`, `EmbeddingsResource`, `ModelRouterResource`, `RagResource`, `AudioResource`, and `QuantizationResource`, plus the shared API package DTOs. (The resource inventory also includes `RagPipeline`, a workflow contract.)
- `OpenAiCompatibleResource` owns alias-to-path state, Hugging Face download orchestration, model load/unload calls, chat/completion DTOs, SSE framing, and token estimation. It obtains engine and hub services through `Instance<Object>` and reflective method lookup.
- `EmbeddingsResource` injects `OpenAiCompatibleResource` to reuse its registry and reflectively invokes an embedding engine. The two HTTP resources are coupled through a CDI resource instance rather than a transport-neutral model registry/runtime service.
- `ModelRouterResource` owns admin-key checks and primary/secondary model, canary-mode, and promotion endpoints. These are serving/deployment control-plane operations.
- `RagResource` owns collection/model registries and HTTP ingest/query operations. `RagPipeline` models document ingest, retrieval, and answer generation; it is not a Safetensors format contract.
- AQLI's `runtime/runner/safetensor/aqli-safetensor-engine/build.gradle.kts` declares `aljabr-safetensor-api` as an implementation dependency alongside the Safetensors core/SPI/loader/bridge artifacts. AQLI root dependency substitution also maps the legacy Alkhawarizm Safetensors API coordinate to this artifact. This proves a runtime dependency edge, but not which API classes AQLI imports or whether the dependency is currently needed at runtime.
- The AQLI Safetensors engine source tree has its own direct generation, routing, session, model lifecycle, and execution packages. Its runtime responsibilities should not be inferred from the Aljabr HTTP resource API.
- Import-level follow-up checked AQLI's primary execution classes (`DirectInferenceEngine`, `DirectModelLoader`, `DirectSafetensorBackend`, `SafetensorInferenceEngineAdapter`, and `TextInferenceEngine`): none imports `tech.kayys.aljabr.safetensor.api`. Repository code search for the package and resource types returned no AQLI matches.
- Based on that evidence, AQLI PR [#706](https://github.com/andalus-ai/aqli/pull/706) removes the direct implementation dependency from `aqli-safetensor-engine`. It intentionally retains the legacy coordinate substitution and Aljabr artifact pending external-consumer/compatibility audit. Gradle CI remains the validation gate.

**Direction:** Complete the exact AQLI import/runtime-use check and inspect tests, published POMs, and external compatibility before moving anything. Separate format parsing/loading and model/tensor capabilities from HTTP transport, model routing/deployment policy and RAG workflow. The latter belong to AQLI or an explicitly optional downstream integration artifact. Do not add HTTP/Quarkus dependencies to the core Safetensors loader path. Preserve any genuine format execution contract through a focused format API/SPI artifact rather than moving it with the HTTP resources.

### F-006 — GGUF API artifact exposes a generic inference-engine facade and a legacy service descriptor

**Severity:** High · **Type:** Format/runtime coupling / legacy integration

Observed files:

- `engine/gguf/aljabr-gguf-api/src/main/java/tech/kayys/aljabr/gguf/api/GgufInferenceEngine.java`
- `engine/gguf/aljabr-gguf-api/src/main/resources/META-INF/services/tech.kayys.gollek.plugin.runner.RunnerPlugin`

`GgufInferenceEngine` implements the general Aljabr `InferenceEngine`, discovers GGUF backends, owns a model-to-backend map, and tracks engine-level lifecycle/statistics. The service descriptor registers `GgufRunnerPlugin` under a legacy `tech.kayys.gollek` service type.

A format adapter may legitimately own GGUF-specific parsing and backend creation, but a format API should not become the competing generic inference runtime. A stale or externally owned ServiceLoader contract in a published artifact is also a discoverability and packaging risk.

**Direction:** Verify whether the legacy service interface is intentionally supplied by a compatibility dependency, whether the descriptor is consumed, and which AQLI code constructs this facade. Then separate format-specific backend creation from generic serving orchestration without losing the Java GGUF execution path.

### F-007 — Root build applies preview/native JVM flags globally

**Severity:** Medium · **Type:** Build boundary / developer experience

Observed file: `build.gradle.kts`.

The root convention adds `--enable-preview` to every JavaCompile task and `--enable-preview` plus `--enable-native-access=ALL-UNNAMED` to every test and JavaExec task, including modules that may not use preview or native APIs. The comment refers to Java 21 FFM, while the project toolchain is JDK 25. The Vector API module allowlist also contains a duplicate clustering project entry.

**Direction:** Inventory actual preview/FFM/Vector API use and consumer requirements. Scope flags to modules/tasks that require them, keep test/runtime flags consistent with compile requirements, and remove stale comments/duplicate entries. Validate the clean build and published consumer profiles before changing conventions.

### F-008 — CPU backend implementation is a large concentration point

**Severity:** Medium · **Type:** Cohesion / maintainability candidate

Observed file: `backend/cpu/aljabr-backend-cpu/src/main/java/tech/kayys/aljabr/backend/cpu/CpuBackend.java` (approximately 92 KB in the audited tree).

File size alone does not prove a Single Responsibility violation. Inspect its operation families, storage access, dispatch, parallel execution, fallback, and resource lifecycle responsibilities before splitting. Extract only cohesive implementation units with real independent change drivers and focused tests; do not create a generic utility package or speculative interface hierarchy.


### F-005 follow-up — Downstream ownership and migration disposition

A follow-up comparison against the AQLI repository establishes that these resources should not be moved into a new Aljabr REST/integration module merely to preserve their current packaging.

**Confirmed AQLI overlap**

- AQLI already owns the REST adapter module at `framework/api/aqli-api`, with Quarkus REST, CDI, OpenAPI and REST-assured/JUnit test dependencies.
- It already implements `ChatCompletionsResource` at `framework/api/aqli-api/src/main/java/tech/kayys/aqli/server/api/v1/ChatCompletionsResource.java`, including non-streaming and SSE chat completion paths.
- It already implements `EmbeddingsResource` at `framework/api/aqli-api/src/main/java/tech/kayys/aqli/server/api/v1/EmbeddingsResource.java`.
- It also owns general completion, model, job, health and administrative REST resources in the same API module.
- AQLI's `aqli-api/build.gradle.kts` explicitly excludes both `aqli-safetensor-api` and `aljabr-safetensor-api`, so the old Safetensors REST artifact is not part of the canonical AQLI API dependency graph.
- AQLI's API module has a test source tree. Its `ServerApiTest` exercises the chat completion endpoint (including successful responses, streaming and invalid requests) and embeddings endpoint (including successful response shape and invalid requests) over HTTP; mapper, request-header, model-operation and streaming tests provide additional coverage. The tests are consolidated in the server API integration suite rather than separate resource-named test classes, so do not mistake the absence of those class names for absence of endpoint coverage.

**Current Safetensors API artifact shape**

- `aljabr-safetensor-api` has no `src/test` tree and no `src/main/resources` tree in the audited baseline.
- Its Gradle build directly declares Quarkus REST, Jackson REST, JAX-RS, CDI, MicroProfile Config/OpenAPI and Mutiny dependencies.
- It publishes only to `mavenLocal()` in the inspected build configuration.
- The REST classes contain runtime registries, reflection-based service lookup, endpoint DTOs and workflow/control-plane behavior. They are not format parsing, tensor, model-loading or weight-layout contracts.
- The resource classes expose method-level `@Path` annotations but no class-level `@Path` annotations in the audited sources. Treat actual Quarkus/JAX-RS discovery as unverified until an integration test proves registration; do not assume these endpoints are active merely because the classes compile.

**Disposition by responsibility**

| Existing type | Canonical owner / action |
|---|---|
| `InferenceApiResource` | Retire from Aljabr. AQLI's `ChatCompletionsResource` is already the canonical serving adapter. Keep protocol interoperability documented on the AQLI endpoint, not encoded as a product name in Aljabr types. |
| `EmbeddingsResource` | Retire from Aljabr. AQLI already owns the endpoint and maps requests through its SDK embedding capability. Keep reusable embedding computation/model support in Aljabr. |
| `ModelRouterResource` | Serving/deployment control plane belongs to AQLI. Do not move its REST class into Aljabr. Audit AQLI routing policy and API coverage before deciding whether an endpoint is still needed. |
| `RagResource` and `RagPipeline` | Remove from the Safetensors format boundary. Retrieval, indexing, collection management and RAG workflow belong in a dedicated downstream knowledge/search product or integration. Aljabr may provide embedding/model execution primitives, but should not own the retrieval workflow or HTTP API. |
| `AudioResource` | HTTP transport and model alias registry belong downstream. AQLI already has audio runtime and audio capability SPI modules, but this resource should not be copied until the AQLI SDK/application layer has the required transcription and speech capabilities. The current TTS method is explicitly a placeholder and must not be promoted as a working endpoint. |
| `QuantizationResource` and its DTOs | Keep quantization algorithms and transport-neutral configuration/results in Aljabr where they are reusable compute/model transformation capabilities. Put any HTTP/admin adapter in AQLI or a downstream application. Do not retain JAX-RS DTOs in the Safetensors API artifact. |

**Migration decision**

Do not create `integration/aljabr-safetensor-rest` or another Aljabr REST artifact. It would preserve the wrong ownership boundary and duplicate AQLI's existing transport adapter. The target state is:

1. Aljabr retains Safetensors parsing, tensor/weight mapping, model execution and quantization primitives.
2. AQLI owns inference, embedding, audio and model-management transport adapters where the corresponding SDK/application capabilities exist.
3. Retrieval/RAG HTTP workflows are owned by the downstream knowledge/search runtime, not by a tensor-format module.
4. The old `aljabr-safetensor-api` artifact is deprecated and then removed after a final external consumer and release-compatibility check. Keep the coordinate only for a bounded transition if a real external consumer is found; do not add a permanent empty compatibility module by default.

**Next implementation sequence**

- Add resource-level API conformance tests in AQLI for chat completion and embeddings before deleting the duplicate implementations.
- Verify whether any external consumer outside the inspected Andalus organization depends on `aljabr-safetensor-api`; the organization search found no references, but that is not proof about all published consumers.
- Audit the Aljabr quantization engine/API independently to ensure the useful compute contract is retained when its REST wrapper is retired.
- Then remove the REST/workflow classes and their Quarkus/JAX-RS dependency set from the Safetensors artifact in a focused removal PR. Do not migrate placeholder or untested endpoint behavior as part of that removal.

## 3. Priority order

| Priority | Finding | Next evidence/action |
|---|---|---|
| P0 | F-005 Safetensors API contains HTTP/RAG/control-plane resources | Full source/import/runtime consumer map; propose a downstream transport/workflow extraction |
| P1 | F-002/F-003/F-004 inference SPI mixes serving concerns | Complete AQLI + Aljabr production/test/public-API consumer graph; classify every contract member |
| P1 | F-006 GGUF generic engine and legacy service descriptor | Verify ServiceLoader contract ownership and AQLI construction/consumption |
| P1 | F-007 global build flags | Module-level JDK API usage and Gradle task/profile audit |
| P2 | F-008 CPU backend concentration | Method/responsibility map, call graph and test seam inventory |
| Done via PR | F-001 stale root architecture page | Keep the compatibility entry point aligned with the unified architecture |

## 4. Non-goals and safeguards

- Do not rename or merge GGUF and Safetensors loaders; format-specific parsing and weight layouts remain distinct.
- Do not move generic numerical/tensor/model-computation primitives into AQLI merely because the current AQLI runner uses them.
- Do not move request admission, continuous batching policy, HTTP/SSE endpoints, auth, deployment routing or serving metrics into Aljabr.
- Do not add a new generic inference/runner SPI to conceal existing duplication.
- Do not split classes based only on line count or introduce interfaces with one implementation and no real extension/test boundary.
- Do not remove published types or service descriptors until production consumers, external compatibility and packaging behavior are understood.
- Hardware-specific validation remains separate from portable unit/contract testing; request user hardware testing only when a change actually requires it.

## 5. Completion criteria for this audit phase

- Every high-priority finding has a complete production/test/public API consumer map.
- Each candidate has a canonical owner and explicit migration/compatibility plan.
- Gradle dependency edges and published POM/module metadata support the intended boundaries.
- Tests cover provider discovery, resource lifecycle, streaming termination/cancellation and relevant fallback/error paths.
- Each implementation PR changes one coherent boundary and includes evidence from focused tests or clearly records why hardware validation is deferred.
