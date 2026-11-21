# Aljabr Diffusion Execution API

Backend-neutral execution contracts shared by sigma-space diffusion schedulers.

## Contracts

- `SigmaStepContext` is an immutable, per-step snapshot of the sigma transition.
- `SigmaPredictionConverter` explicitly converts a model output into the derivative expected by a sigma-space solver.
- `SigmaStepExecutor` applies a scheduler-specific tensor update and returns a newly owned sample tensor.

The contracts deliberately do not infer the meaning of a model's `VELOCITY`
output. Velocity parameterizations vary by model family and training objective;
a family must supply an explicit converter. The same rule applies to any
non-standard epsilon/sample parameterization.

Inputs to execution are borrowed. A converter must return a newly owned
derivative tensor. An executor owns and must close that derivative after use,
and returns a newly owned output sample. This makes tensor lifetime explicit
without exposing backend storage or imposing in-place mutation.

This first contract is stateless. Multistep solver history and random-noise
sources will be explicit additions rather than hidden mutable executor state.
