# Aljabr Diffusion SafeTensors Adapter

Format integration for diffusion component manifests. This module implements
the family-neutral DiffusionComponentLoader SPI by delegating to the existing
Aljabr SafetensorShardLoader.

- Opens local .safetensors files and directories containing sharded checkpoints.
- Uses the existing lazy shard mapping and tensor views; it does not parse or
  copy checkpoint data itself.
- Returns an owned handle. Tensor views are valid only while that handle remains
  open; close it to release mapped shard resources.
- Accepts local filesystem paths and file: URIs only. Hugging Face or remote
  repository locations must first be materialized by a separate repository/
  artifact resolver.

The adapter has no dependency on a model family or compute backend.
