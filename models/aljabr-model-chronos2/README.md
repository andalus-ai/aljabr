# Aljabr Chronos-2 model family

This optional model-family module recognizes the Hugging Face checkpoint
`autogluon/chronos-2` (architecture `Chronos2Model`) and validates its
forecasting configuration.

Chronos-2 is an encoder-only time-series foundation model. It is not a causal
language model and must not be routed through text generation, tokenization, or
the GGUF decoder path.

## Checkpoint characteristics

- Safetensors checkpoint; upstream config declares `model_type=t5` and
  `architectures=[Chronos2Model]`.
- 120M parameters, float32 checkpoint.
- 8,192 context steps and up to 1,024 forecast steps.
- 16-step input/output patches.
- Quantile output head.
- Univariate, multivariate, cross-series, and covariate-informed forecasting.

## Current implementation status

This integration currently provides model-family discovery, a dedicated
forecasting capability, strict parsing/validation of the upstream
`config.json` contract, Chronos-2 encoder input assembly, and a checkpoint
name/shape layout validator. The mapper validates Safetensors header metadata
without owning or copying tensor payloads. This is **not** yet executable model
inference or direct Safetensors runner support.

The input assembler uses the shared Time Series API for fitted normalization
and patch extraction, then applies Chronos-2's three-channel time/value/mask
feature layout. The checkpoint mapper declares the upstream tensor names and
shapes for the embedding blocks, encoder attention/FFN layers, and quantile
head. Float32 reference execution now covers the encoder stack and the residual input/output projections: time self-attention with RoPE, same-group attention across series, position-wise ReLU feed-forward residuals, final normalization, input patch embedding, optional regression-token insertion, and quantile-major output patch projection. The residual projection math is extracted into `aljabr-model-common` so other model families can reuse it. These components consume explicit checkpoint-supplied weights and remain correctness references rather than optimized backend kernels. Actual Safetensors tensor binding, full batched covariate/group assembly, normalization inversion in the forecast orchestration path, and numerical parity against upstream reference outputs remain before the family can advertise inference readiness.

## Upstream

- https://huggingface.co/autogluon/chronos-2
- https://huggingface.co/amazon/chronos-2
- https://github.com/amazon-science/chronos-forecasting
