# Aljabr Autograd NN Adapter

This optional module bridges the eager autograd API and existing Aljabr NN
modules without introducing an autograd dependency into either
`aljabr-nn` or `aljabr-tensor`.

## Differentiable Linear

`DifferentiableLinear` wraps an existing `Linear` module and records
`input × weightᵀ + bias` using an explicit `AutogradContext`. It exposes
differentiable weight and bias variables so callers can inspect their
gradients after backward.

The adapter currently supports eager F32, rank-two batched input, rank-two
weights, and optional rank-one bias. The context and its borrowed parameter
tensors must remain alive through backward. Do not replace or move the wrapped
module's parameters while the adapter's context is active.

Ordinary `Linear.forward(Tensor)` remains unchanged and has no autograd
dependency. This adapter is opt-in and does not claim general NNModule coverage.
