# AQLI Inference SPI Consolidation Inventory

## Purpose

This document records the repository-level contract comparison that precedes the AQLI-to-Aljabr inference SPI migration. The target architecture is one canonical inference contract surface in Aljabr. AQLI consumes those contracts and retains runtime orchestration; it must not publish a second set of equivalent inference DTOs or engine interfaces.

This is a migration inventory, not authorization to delete AQLI APIs before their consumers have been migrated.

## Canonical ownership

| Concern | Canonical owner |
|---|---|
| Inference engine and request/response contracts | Aljabr |
| Streaming inference chunk contract | Aljabr |
| Shared inference message, attachment, tool and request-context value types | Aljabr inference SPI |
| Model execution, reusable model operations and compute capabilities | Aljabr |
| Runtime provider selection, request admission/routing, scheduling, sessions and serving policy | AQLI |
| HTTP/API/CLI/deployment lifecycle and operational configuration | AQLI |

AQLI may define runtime-specific orchestration types, but they must not duplicate the canonical model-execution/inference contract types.

## Confirmed duplicate surfaces

The following same-named public types exist in both repositories:

- `InferenceEngine`
- `InferenceRequest`
- `InferenceResponse`
- `InferenceResponseInterface`
- `StreamingInferenceChunk`
- `InferenceObserver`
- `InferencePhase`
- `InferencePipeline`
- `InferenceStage`
- `KVCacheState`
- `LocalInferenceEngine`
- `Priority`
- `StreamingResponse`
- `StreamingSession`
- `ValidationContext`
- `VerificationRequest` and `VerificationResponse`
- `AsyncJobStatus`
- `Attachment`
- inference DTOs `ChatMessage`, `InferenceContext`, `InferenceResult` and `PromptRequest`
- `BatchInferenceRequest`

AQLI additionally has `InferencePhasePlugin`, which is AQLI plugin/runtime integration surface and should be evaluated separately rather than copied into Aljabr by default.

## Contract comparison findings

### Engine interface

The Aljabr and AQLI `InferenceEngine` interfaces have the same overall lifecycle and execution methods: infer, asynchronous and blocking execution, streaming, async job submission, health, initialization, shutdown and statistics.

There is a material package-level divergence in embedding methods: Aljabr returns Aljabr embedding request/response types, while AQLI returns AQLI embedding request/response types. This must be resolved against Aljabr's embedding SPI before AQLI can consume the canonical interface directly.

### Request and response

The two `InferenceRequest` implementations have matching field intent and broadly matching construction/access patterns, but bind their fields to different package-local supporting types:

- AQLI uses `tech.kayys.aqli.spi.Message`, `RequestContext` and `ToolDefinition`.
- Aljabr uses `tech.kayys.aljabr.spi.Message`, `RequestContext` and `ToolDefinition`.

The request implementations also carry low-level/native context state and request/runtime hints. These semantics and serialization behavior must be preserved during migration.

The `InferenceResponse` implementations have matching response fields, finish-reason semantics, tool-call representation and builder shape. They remain distinct Java types and therefore are not interchangeable merely because their serialized forms are similar.

The two `InferenceResponseInterface` declarations are structurally identical. They are a strong candidate for immediate consolidation once all AQLI references are redirected to the Aljabr interface.

### Streaming

The two `StreamingInferenceChunk` records have the same component layout and factory/helper behavior in the inspected sources. Their modality type is already sourced from Aljabr in AQLI, but the record itself remains duplicated under the AQLI package.

### Supporting types

The inspected AQLI and Aljabr versions of `RequestContext` and `ToolDefinition` have matching APIs and behavior. The richer inference `Message` types also have matching multimodal fields and factories; both reference their repository-local `ToolCall` type.

Aljabr also has a separate lightweight tokenizer/model `Message` value type. It is not the inference message and must not be confused with `tech.kayys.aljabr.spi.Message` in the inference SPI.

## Dependency observation

AQLI's `aqli-spi-inference` Gradle module already declares an API dependency on `tech.kayys.aljabr:aljabr-spi-inference`, while continuing to publish its own copies of the inference contracts. This makes the duplication especially avoidable, but the supporting AQLI SPI types and embedding packages must be aligned first.

## Migration sequence

1. **Complete consumer inventory in AQLI.** Identify every production source, test, plugin, public SDK and serialization boundary that imports or exposes the AQLI inference SPI types. Classify each use as runtime implementation, public API, adapter, or test.
2. **Align supporting contracts.** Confirm the canonical Aljabr definitions for messages, tool calls/definitions, attachments, request context, modality and embedding contracts. Preserve any AQLI-only runtime information in AQLI-owned runtime context types rather than forking the inference request.
3. **Migrate AQLI consumers.** Change AQLI production and test code to import Aljabr's canonical inference contracts. Keep explicit adapters at external API/wire boundaries where compatibility requires them.
4. **Remove duplicate AQLI inference SPI types.** Delete only after repository-wide references and public API exposure have been addressed and all affected Gradle modules/tests pass.
5. **Enforce the boundary.** Keep the AQLI SPI module dependent on Aljabr's API and add architecture/build checks preventing new duplicate inference contracts.

## Safety gates

- Do not remove a public AQLI type while production code, tests, SDKs, plugins or serialization boundaries still depend on it.
- Do not convert the AQLI runtime into an Aljabr serving layer. Aljabr owns the canonical contract and reusable execution capabilities; AQLI continues to own runtime orchestration and serving.
- Preserve JSON property names and wire compatibility unless a separately reviewed versioned API change explicitly authorizes a change.
- Keep GGUF and Safetensors format loaders/adapters distinct; this consolidation is about shared inference contracts, not format identity.
- Validate with focused module tests first, then the full AQLI build/CI. Hardware-specific inference validation remains a separate gate.

## Scope of this inventory

This document records confirmed type-level duplication and selected contract comparisons. It does not claim that the complete production/test consumer graph has already been enumerated. The consumer inventory in step 1 is required before the first deletion or package migration.
