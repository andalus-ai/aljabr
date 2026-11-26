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
