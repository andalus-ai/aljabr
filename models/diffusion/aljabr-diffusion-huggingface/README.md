# Aljabr Diffusion — Hugging Face Resolver

Optional repository adapter implementing the family-neutral diffusion artifact resolver SPI.

## Artifact location

Use `hf://owner/repository/path/to/component.safetensors?revision=main`.
The revision defaults to `main`. Mutable branch/tag names are resolved through the Hugging Face model API and every file request is pinned to the returned 40-character commit SHA.

## Cache and integrity

Files are stored below `<cacheDirectory>/huggingface/<owner>/<repository>/<commit>/<file>`.
Downloads use a temporary file and are moved into place only after transfer and checksum validation. Declared artifact SHA-256 values are always checked. Hugging Face LFS SHA-256 metadata is also checked when exposed by the response headers. Ordinary Git blob IDs are not treated as SHA-256.

Set `HF_TOKEN` for gated/private repositories. The token is sent only as a bearer authorization header and is not written to cache paths or exception messages.

## Scope

This resolver materializes one repository file per artifact. Sharded SafeTensors indexes and their shard files should be represented as separate manifest artifacts (or resolved by a higher-level manifest-aware coordinator). It does not introduce a second checkpoint loader, downloader framework, or credential store.
