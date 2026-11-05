# ADR-0002: Inference Contracts and Model Execution Ownership

- **Status:** Accepted
- **Date:** 2026-11-05
- **Scope:** Aljabr inference/model-execution APIs and AQLI runtime inference APIs

## Context

Aljabr is the reusable Java ML/computational framework. AQLI is the inference runtime that consumes Aljabr and owns serving, orchestration, scheduling, and deployment behavior.

The current repositories contain overlapping inference APIs:

- Aljabr publishes `tech.kayys.aljabr.spi.inference.InferenceEngine`, `InferenceRequest`, `InferenceResponse`, and `StreamingInferenceChunk`.
- AQLI publishes corresponding types under `tech.kayys.aqli.spi.inference`.
- AQLI's `framework/spi/aqli-spi-inference` already declares an API dependency on Aljabr's `aljabr-spi-inference`, while also defining its own request, response, engine, response-interface, and streaming-chunk types.
- The request/response and streaming types are structurally very similar, but their package-local dependencies differ. For example, the AQLI request uses AQLI's `Message` and related types, while Aljabr's request uses Aljabr's equivalents.
- Aljabr also has format-specific execution contracts: `GgufBackend` / `GgufBackendProvider` and `SafetensorEngine`.
- AQLI's `DirectInferenceEngine` implements Aljabr's `SafetensorEngine`, while AQLI's `DirectSafetensorBackend` adapts the AQLI request API to that direct execution path.
- Aljabr's `GgufInferenceEngine` implements Aljabr's inference SPI and manages GGUF backend selection and loaded backends. Its Java backend contains a Java generation implementation, but its provider currently declares no architecture ready for automatic generation selection.

This means the current design has two overlapping general inference SPIs and two format-specific lifecycle/execution approaches. Removing one set of types immediately would cause broad source and binary compatibility risk, and would not by itself establish a correct format-neutral execution boundary.

## Decision

### 1. Aljabr owns reusable model and computation capabilities

Aljabr is the canonical owner of reusable framework-level capabilities, including:

- Tensor, dtype, layout, storage, device, and resource-lifetime contracts.
- Model configuration, architecture metadata, model-family extension points, and reusable model representations.
- Format-specific parsing/loading and weight-access contracts for GGUF, Safetensors, ONNX, LiteRT, and other supported formats.
- Format adapters that translate source-format tensors and metadata into reusable Aljabr model/tensor APIs.
- Reusable transformer/model execution operations and primitives, including forward execution, prefill/decode building blocks, attention and KV-cache mechanisms where they are general reusable computation capabilities.
- Tokenization, sampling algorithms, numerical kernels, and backend dispatch where they are reusable independently of a serving product.
- CPU, Metal, CUDA, and other compute backend implementations and their capability/readiness reporting.
- Low-level execution diagnostics and measurements.

A format is not an execution engine. GGUF and Safetensors remain distinct formats with distinct loaders/adapters. Their model data may be normalized to shared Aljabr representations where semantics and implementation permit; the loaders must not be conflated or renamed as if they were one format.

### 2. AQLI owns inference-runtime orchestration and serving policy

AQLI owns:

- Runtime model registration and user-facing model identity.
- Runtime configuration, provider-selection policy, and fallback policy.
- Request validation/planning, routing, admission, scheduling, batching, and concurrency policy.
- Session lifecycle, request cancellation, client disconnect behavior, and serving-level KV-cache/session policy.
- Runtime generation options and request-to-execution mapping.
- Streaming delivery semantics, API protocols, CLI behavior, and serving endpoints.
- Operational metrics and profiling aggregation: TTFT, request latency, throughput, queueing, concurrency, per-model/session metrics, dashboards, and alerts.
- Deployment composition and selection of which Aljabr providers are installed.

AQLI may orchestrate repeated Aljabr execution calls for prefill and decode. That does not make Aljabr a serving runtime, nor does it transfer reusable model computation out of Aljabr.

### 3. Aljabr format APIs must not become competing serving runtimes

Format-specific Aljabr APIs may own format-specific parsing, model loading, backend-provider discovery, backend instantiation, and resource release when those are necessary to provide reusable framework capabilities.

They must not accumulate AQLI-specific concerns such as HTTP/SSE behavior, CLI lifecycle, request admission, tenant policy, serving queues, deployment configuration, or production metrics aggregation.

The existing `GgufInferenceEngine` and `SafetensorEngine` contracts require a further source audit before changing their public shape. This ADR does not authorize deleting or moving them in this change.

### 4. Do not introduce another generic engine SPI

Do not add a new `JavaInferenceEngine`, `ModelExecutor`, `UnifiedInferenceEngine`, or equivalent abstraction merely to bridge the current duplication.

First determine which existing Aljabr contracts can express reusable model execution and which AQLI contracts are runtime-facing. Any new contract must fill a demonstrated gap, have one canonical owner, and be format-neutral only where the underlying semantics truly are format-neutral.

### 5. Preserve AQLI compatibility during convergence

AQLI's existing `tech.kayys.aqli.spi.inference` API remains in place during the migration. It may continue to provide runtime-facing facades and compatibility adapters.

The duplicated AQLI request/response/streaming types are migration candidates, not immediate deletion candidates. Before consolidating them:

1. Inventory production and test consumers in both repositories.
2. Compare all fields, constructors, builders, serialization annotations, enum values, and behavioral semantics—not only class names.
3. Identify AQLI-only fields and supporting types, including auth, request context, attachments, tools, and runtime metadata.
4. Define explicit conversion/adapter behavior and compatibility guarantees.
5. Migrate consumers incrementally and validate published-artifact compatibility.
6. Deprecate duplicate types only after consumers have moved and compatibility tests pass.

Do not create a dependency cycle: Aljabr must never depend on AQLI.

### 6. Keep format-specific adapters separate

The target design retains separate format adapters such as:

- `GgufModelLoader` / GGUF reader and metadata adapter.
- `SafetensorsModelLoader` / Safetensors reader and shard adapter.

They should expose compatible Aljabr model/weight/tensor capabilities where possible. The shared execution path consumes those common capabilities; it should not require a fake `GgufSafetensorBackend` or `SafetensorGgufBackend` wrapper.

Temporary llama.cpp and LibTorch integrations remain optional compatibility backends until the Java paths are validated for correctness, supported architectures, and performance. They must not be removed merely because a Java implementation exists.

## Current-state findings

| Area | Current observation | Direction |
|---|---|---|
| Aljabr inference SPI | General request/response/engine contracts already exist | Audit and narrow to reusable framework execution responsibilities |
| AQLI inference SPI | Similar contracts exist despite an API dependency on Aljabr inference SPI | Retain compatibility while mapping and migrating duplicated types |
| GGUF | Aljabr owns parser, GGUF backend SPI, Java backend, and GGUF inference facade | Preserve Java execution work; separate reusable execution from runtime policy incrementally |
| Safetensors | Aljabr owns Safetensors loader SPI; AQLI direct engine currently implements its lifecycle contract | Keep format loading in Aljabr; keep AQLI orchestration; audit execution primitives for reusable Aljabr ownership |
| Streaming | Both repositories expose similar streaming chunk DTOs | Separate token/execution events from AQLI transport and serving semantics before consolidation |

## Migration sequence

### Phase 1 — contract inventory (this ADR)

- Record the ownership boundary and current duplicated API surface.
- Inventory source and binary consumers of both inference SPIs.
- Compare model, weight, tokenizer, tensor, generation, and streaming contracts across both formats.
- Identify the smallest reusable execution seam already present in Aljabr.

### Phase 2 — Aljabr execution contract alignment

- Audit existing model-family, tensor, tokenizer, GGUF, and Safetensors APIs.
- Reuse or evolve existing Aljabr contracts rather than creating parallel SPIs.
- Keep GGUF and Safetensors loaders separate.
- Add format-parity contract tests for shared model metadata and tensor access where applicable.
- Keep serving and deployment policy out of Aljabr.

### Phase 3 — AQLI adapter convergence

- Make AQLI runtime adapters consume the aligned Aljabr capabilities.
- Keep AQLI's external/runtime-facing API stable while adapters are introduced.
- Consolidate duplicate request/response/event types only with explicit compatibility coverage.
- Move reusable transformer execution code into Aljabr only when it is demonstrably independent of AQLI runtime policy.

### Phase 4 — validate Java-first execution

- Validate Java GGUF and Safetensors paths independently against reference outputs.
- Record architecture, dtype, quantization, backend, and hardware coverage.
- Benchmark CPU and accelerator backends under equivalent conditions.
- Promote Java providers to automatic selection only for validated model/architecture/backend combinations.
- Retain native fallbacks until Java paths meet agreed correctness, compatibility, and performance gates.

## Acceptance criteria

- Aljabr builds and is usable without an AQLI checkout or dependency.
- AQLI consumes Aljabr's published framework/model-execution APIs rather than maintaining a parallel implementation of those capabilities.
- No duplicate generic engine SPI is introduced.
- GGUF and Safetensors remain distinct format adapters.
- Reusable model execution and compute primitives are owned by Aljabr; serving orchestration and runtime policy are owned by AQLI.
- AQLI's public/runtime compatibility is preserved through the migration.
- Java provider readiness is based on functional validation, not merely implementation presence.
- Native fallback removal is gated by documented correctness and performance evidence.
