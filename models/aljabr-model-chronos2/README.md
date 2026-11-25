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

This first integration slice provides model-family discovery, a dedicated
forecasting capability, and strict parsing/validation of the upstream
`config.json` contract. It does **not** yet provide executable model inference
or claim direct Safetensors support.

The next implementation slice must add the Chronos-2 checkpoint tensor mapper
and forward path: robust scaling/arcsinh preprocessing, patch/meta-feature
encoding, rotary time attention, group attention, feed-forward blocks, and the
quantile head. It must be validated against upstream reference outputs before
the family advertises direct inference readiness.

## Upstream

- https://huggingface.co/autogluon/chronos-2
- https://huggingface.co/amazon/chronos-2
- https://github.com/amazon-science/chronos-forecasting
