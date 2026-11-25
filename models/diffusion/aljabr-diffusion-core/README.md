# Diffusion Core

Family-neutral orchestration for diffusion model components.

## Component pipeline

`DiffusionPipelineLoader` coordinates the existing capability-selection and format-loading contracts:

1. `DiffusionComponentResolver` selects artifacts for the requested component roles and validates all required roles before I/O.
2. `DiffusionArtifactResolverRegistry` materializes each selected artifact using a URI-scheme resolver (for example, the local-file resolver or optional Hugging Face adapter).
3. `DiffusionComponentLoaderRegistry` selects a format adapter (for example, the SafeTensors loader).
4. `LoadedDiffusionPipeline` owns the loaded component handles and closes them in reverse load order.

The core module does not know about MiniMax-H3, other model families, Hugging Face APIs, tensor formats, or compute backends. Those remain independent providers.

## Ownership

The caller owns the returned `LoadedDiffusionPipeline` and must close it. Closing the pipeline closes every loaded component handle. If a later component fails to resolve or load, handles already opened by the pipeline are closed before the failure is propagated.

Artifact files remain owned by their resolver/cache. Closing a loaded pipeline does not delete cached files.
