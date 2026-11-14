# Aljabr Autograd NN Adapter

This optional module bridges the eager autograd API and existing Aljabr NN
modules without introducing an autograd dependency into either
`aljabr-nn` or `aljabr-tensor`.

## Differentiable Linear

`DifferentiableLinear` wraps an existing `Linear` module and records
`input × weightᵀ + bias` using an explicit `AutogradContext`. It exposes
differentiable weight and bias variables, their registered `Parameter` handles,
and validated gradient tensors after backward. The adapter does not apply
updates or own optimizer state.

The adapter currently supports eager F32, rank-two batched input, rank-two
weights, and optional rank-one bias. The context and its borrowed parameter
tensors must remain alive through backward. The adapter checks parameter
identity on each forward/gradient access and fails if a parameter has been
replaced. Create a new adapter and context after replacing or moving parameters.
Returned gradients are borrowed from the context and must not be released
independently.

Ordinary `Linear.forward(Tensor)` remains unchanged and has no autograd
dependency. This adapter is opt-in and does not claim general NNModule coverage.
