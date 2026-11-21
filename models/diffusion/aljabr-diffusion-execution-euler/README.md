# Aljabr Diffusion Euler Sigma Execution

A backend-neutral Euler solver step over the shared sigma-space execution API.

For a model-family supplied derivative converter, the executor applies:

`x(next) = x(current) + derivative * (sigma(next) - sigma(current))`

The executor does not infer prediction semantics. The supplied converter must
declare the same `PredictionType` as the step context and return a newly owned
derivative tensor with the same shape as the current sample.

The current sample and model output are borrowed. The executor closes the
converter-owned derivative, closes its scaled intermediate, and returns a
newly owned output sample. Backend selection remains outside this module.

No universal VELOCITY conversion is provided. Epsilon, sample, flow, and
velocity conversions depend on the model family's parameterization and
training objective; explicit converters are the appropriate extension point.
