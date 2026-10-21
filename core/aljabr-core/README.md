aljabr-core

Purpose
- Aggregator module exposing stable public APIs from core aljabr modules for external consumers (e.g., aljabr)

Contents
- Re-exports: tensor, tokenizer-core, model-runner, model-repository, SPIs
- Compile-only references to quantizers so consumers can opt-in to implementations

Migration notes
- To migrate a aljabr consumer, add a dependency on project(":core:aljabr-core") and refactor uses of internal aljabr modules to the public APIs.
- Keep only minimal runtime interfaces in aljabr to avoid circular deps.

Next steps
1. Update aljabr consumer build files to depend on aljabr-core.
2. Extract any shared classes that cause cycles into small SPI modules if needed.
