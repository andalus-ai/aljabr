# Aljabr Diffusion Execution Loop

Family-neutral orchestration for iterative sigma-space denoising.

This module composes:
- a caller-provided denoiser,
- the family-provided prediction-to-derivative converter,
- an existing `SigmaStepExecutor`,
- immutable `SigmaStepContext` values.

It does not select a model family, scheduler, device, or backend. Callers build the ordered step contexts from their scheduler's schedule.

## Ownership

- The initial sample is borrowed and remains open.
- Each denoiser prediction must be a newly owned tensor. The loop closes it after the step.
- Each step executor must return a newly owned tensor. The loop owns intermediate results and closes them after the next result is produced.
- The final tensor is returned to the caller, who must close it.
- A denoiser or executor must never return either borrowed input as an owned result.

Cancellation and progress callbacks are caller supplied. Cancellation is checked before each model evaluation. If cancellation, inference, or a callback fails, all loop-owned tensors are closed.

## Scope

The first implementation supports sigma-space solvers through `SigmaStepExecutor`. Discrete alpha/beta schedulers remain separate and should use a matching execution contract rather than being coerced into sigma semantics.


## Classifier-free guidance

`ClassifierFreeGuidance` combines unconditional and conditional predictions
using:

```text
guided = unconditional + guidanceScale * (conditional - unconditional)
```

A `GuidancePredictionProvider` owns model invocation and conditioning. It may
advertise batched prediction support to evaluate both branches in one model
call; otherwise the engine invokes the unconditional and conditional branches
sequentially. The engine never guesses backend batching support.

`GuidanceTensorOperations` owns tensor algebra and optional standard-deviation
matching/blending for guidance rescale. This keeps allocation and
backend-specific kernels outside the orchestration layer.

Provider predictions and intermediate tensors are newly owned. The guidance
engine closes predictions and intermediate guided tensors. The returned
prediction is newly owned by the caller. The input sample is borrowed.
