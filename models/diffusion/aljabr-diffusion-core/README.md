# Aljabr Diffusion Core

Family-neutral diffusion orchestration primitives.

The core provides:

- An explicitly supplied registry of optional diffusion-family providers.
- `DiffusionComponentResolver`, which resolves capability requirements against an immutable component manifest with deterministic variant selection.

Resolution is metadata-only. It does not fetch artifacts, parse checkpoints, allocate tensors, select hardware backends or implement a serving lifecycle. Format-specific loaders and storage adapters remain separate capabilities and should reuse Aljabr's existing model and Safetensors APIs.

Applications may construct registries and resolvers explicitly; there is no global registration.
