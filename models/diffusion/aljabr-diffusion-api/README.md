# Aljabr Diffusion API

Family-neutral public contracts for optional diffusion model implementations.

The API defines:

- `DiffusionModelDescriptor` and `DiffusionFamilyProvider` for family discovery.
- `DiffusionComponentArtifact` and `DiffusionComponentManifest` for describing independently loadable model files.
- `DiffusionComponentRequirement` for declaring required/optional component roles, accepted formats and preferred variants.

A component's role is a stable semantic identifier, not a filename or concrete loader class. Its location is opaque to the API so local, repository and object-store resolution can be supplied by separate storage adapters.

A family module can publish a `DiffusionFamilyProvider` implementation through Java `ServiceLoader`, or an application can inject providers directly. This API deliberately does not load files, parse Safetensors, allocate tensors, select devices or execute pipelines.

## Artifact loading and resolution

- `DiffusionComponentLoader` is the format adapter SPI (for example SafeTensors).
- `DiffusionArtifactResolver` is the storage/repository materialization SPI.
- `ResolvedDiffusionArtifact` records the local file, immutable repository
  revision when available, checksum, and cache-hit status.
- `DiffusionArtifactResolutionContext` carries cache and timeout policy.

Keep repository access separate from format parsing and model-family semantics.
Resolvers must verify declared checksums before returning a materialized file.


## Request conditioning inputs

- `DiffusionConditioningInput` describes one caller-supplied input by stable ID,
  modality, opaque source reference, optional frame sequence index, and
  application metadata.
- `DiffusionConditioningSet` preserves input order, rejects duplicate IDs and
  duplicate video-frame indices, and supports modality filtering.

The source reference is intentionally opaque. The API does not fetch, decode,
preprocess, embed, or transfer the input. Video-frame indices describe caller
ordering; temporal sampling and preprocessing remain execution-layer concerns.
