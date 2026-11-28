# Diffusion model integration

## Conditional 2D UNet adapter

`UNet2DConditionalInvoker` adapts the existing Aljabr
`UNet2DConditional` model to the generic diffusion execution contract.
It delegates checkpoint-specific conditioning extraction and sigma-to-time
conversion to `UNet2DConditioningMapper`.

The mapper receives the borrowed latent sample, request-owned prepared
conditioning, and current sigma-step context. It returns owned
`UNet2DModelInputs`: timestep embeddings and optional encoder hidden states.
The invoker closes these temporary tensors after the model call. The model
and prepared conditioning remain owned by their enclosing model/request
lifecycles.

This deliberately does not hard-code CLIP, T5, a tokenizer, a scheduler
parameterization, or a checkpoint's sigma-to-timestep policy in the generic
adapter. Those belong in checkpoint/family mapping implementations.

**Compatibility note:** the current `UNet2DConditional` implementation is
documented in its source as a simplified scaffold, not a complete
checkpoint-compatible Stable Diffusion UNet. This adapter establishes the
reusable invocation seam; it does not claim end-to-end checkpoint support.
