# Aljabr Diffusion API

Family-neutral public contracts for optional diffusion model implementations.

This module intentionally has no dependency on a concrete model family, scheduler
implementation, tensor backend, AQLI or Tafkir. It currently defines a stable
model descriptor and the provider-discovery contract. Model construction and
pipeline execution contracts will be added after model loading, tensor lifetime
and execution-state requirements are established.

A family module can publish a `DiffusionFamilyProvider` implementation through
Java `ServiceLoader` metadata, or an application can inject providers directly.
