# Aljabr Diffusion Tokenizer Integration

Optional prompt-conditioning adapter backed by Aljabr's tokenizer SPI.

## Responsibilities

- Resolve an opaque prompt input through an injected `PromptTextResolver`.
- Encode the resolved text with an injected `tech.kayys.aljabr.spi.tokenizer.Tokenizer`.
- Return immutable, caller-owned token IDs as `PromptTokenizationData`.
- Configure BOS/EOS insertion through `PromptTokenizationOptions`.

The source resolver is deliberately separate from tokenization. Applications
can resolve inline text, prompt-store identifiers, or other source schemes
without adding storage assumptions to the adapter.

This module produces token IDs only. It does not execute a text encoder,
generate embeddings, choose a tokenizer vocabulary, allocate tensors, or
select a device. Those remain separate capabilities.

## Dependency boundary

This module is optional and depends on:

- `aljabr-diffusion-api` for the generic conditioning SPI.
- `aljabr-spi-tokenizer` for the shared tokenizer contract.

The generic diffusion API does not depend on this integration. Applications
opt into it by adding this module and registering
`TokenizerPromptConditioningPreprocessor` with the conditioning registry.
