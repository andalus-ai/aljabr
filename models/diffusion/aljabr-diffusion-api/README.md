# Aljabr Diffusion API

Family-neutral public contracts for optional diffusion model implementations.

This module intentionally has no dependency on a concrete model family, scheduler implementation, tensor backend, AQLI or Tafkir. It defines model descriptors, provider discovery, component manifests and requirements, plus the format-neutral DiffusionComponentLoader SPI.

A family module declares semantic component roles and requirements. A format integration implements DiffusionComponentLoader<T> to open artifacts of its format and returns an explicitly owned, closeable handle. The caller owns that handle and must close it; tensor views must not outlive its backing resource. The API does not resolve locations, download files, parse checkpoints, or choose compute backends.

A family module can publish a DiffusionFamilyProvider implementation through Java ServiceLoader metadata, or an application can inject providers directly.
