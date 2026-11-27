# Aljabr Diffusion Execution Loop

Family-neutral orchestration for iterative sigma-space denoising.

This module composes a caller-provided denoiser, family-provided prediction
converter, SigmaStepExecutor, and immutable SigmaStepContext values. It does
not select a model family, scheduler, device, or backend.

## Model-family invocation

DiffusionModelInvoker is the boundary between generic execution and an
architecture-specific diffusion model. It receives the current latent sample,
a request-owned PreparedDiffusionConditioningSet, and SigmaStepContext. The
family adapter maps prepared conditioning to the architecture's model inputs
and returns a newly owned prediction tensor.

ConditionedDiffusionDenoiser binds one prepared conditioning set to an invoker.
For classifier-free guidance, ConditionedGuidancePredictionProvider binds
separate unconditional and conditional sets to the same invoker and uses
sequential evaluation. Families/backends that support concatenated branch
evaluation can implement GuidancePredictionProvider directly and opt into
batched prediction.

The request scope owns and closes prepared conditioning after generation.
Denoisers and providers only borrow those resources; they do not close them.
The execution loop owns and closes each model prediction and intermediate
latent tensor according to its existing ownership contract.

## Ownership

- The initial sample is borrowed and remains open.
- Each denoiser prediction must be newly owned; the loop closes it after the step.
- Each step executor returns a newly owned tensor; the loop owns intermediate results.
- The final tensor is returned to the caller, who must close it.
- Cancellation and failures close loop-owned tensors.

## Classifier-free guidance

ClassifierFreeGuidance combines unconditional and conditional predictions:

`guided = unconditional + guidanceScale * (conditional - unconditional)`

GuidanceTensorOperations owns tensor algebra and optional guidance rescale.
The guidance engine closes provider predictions and intermediate tensors;
the returned prediction is newly owned by the caller. The input sample is
borrowed.

Use GuidedDiffusionDenoiser as the DiffusionDenoiser passed to a
scheduler-specific runner. The provider captures prepared conditioning and
performs family-specific model evaluations. No scheduler or pipeline needs a
CFG-specific branch.
