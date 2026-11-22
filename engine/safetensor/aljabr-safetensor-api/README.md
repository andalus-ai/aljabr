# Safetensors REST API — Deprecated

> **Deprecated:** Do not add new dependencies on this artifact. Its HTTP resources and workflow contracts are scheduled for removal from Aljabr after the external-consumer and release-compatibility check.

This module is not a Safetensors format API. It currently bundles Quarkus/JAX-RS resources, transport DTOs, serving policy and workflow endpoints. Those responsibilities do not belong in Aljabr's format-specific model/weight modules.

## Where functionality belongs

- **Chat completions and embeddings HTTP endpoints:** use the AQLI REST API in `framework/api/aqli-api`. AQLI already owns these endpoints and maps them through its SDK.
- **Model routing and serving control plane:** AQLI.
- **Audio HTTP serving:** AQLI's audio capability/runtime once the required SDK operations are supported. Do not rely on the placeholder TTS endpoint in this module.
- **RAG, retrieval and collection workflows:** a downstream knowledge/search runtime, not Aljabr's Safetensors format layer.
- **Quantization computation:** retain reusable algorithms and transport-neutral configuration/results in Aljabr. This module's HTTP resource and JSON DTOs are deprecated.

## Compatibility

No HTTP paths or wire behavior are changed by this deprecation marker. The artifact remains temporarily available while external consumers and release compatibility are checked. It will not receive new endpoint features. A removal release will be announced through the repository's normal release process.

## Naming and protocol compatibility

Implementation types should use neutral capability names. Protocol compatibility (for example, an OpenAI-compatible request/response shape) belongs in endpoint descriptions and documentation, not in product-branded Java class names.
