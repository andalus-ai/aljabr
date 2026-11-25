# Aljabr Diffusion SafeTensors Adapter

This format integration implements the family-neutral `DiffusionComponentLoader`
SPI by delegating checkpoint parsing and shard management to the existing
`SafetensorShardLoader`.

- Opens local SafeTensors files and directories containing sharded checkpoints.
- Preserves lazy shard access and zero-copy tensor views.
- Returns an owned handle; tensor views are valid only while the handle remains open.
- Accepts filesystem paths and `file:` URIs. Remote repository locations must
  first be resolved by a repository/artifact resolver.

The adapter has no dependency on a model family or compute backend.
