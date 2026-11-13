# Aljabr Autograd NN Adapter

This optional integration module connects the eager autograd API to the existing
neural-network module without making either `aljabr-nn` or `aljabr-tensor`
depend on autograd.

The initial adapter supports floating-point, rank-two `Linear` execution with
an optional rank-one bias. It records the input, weight, and bias in an
`AutogradContext` and exposes parameter gradients by their stable
`Parameter` handles.

```java
try (AutogradContext context = new AutogradContext()) {
    Variable trackedInput = context.variable(input);
    LinearAutograd.ForwardPass pass =
            LinearAutograd.forward(context, linear, trackedInput);

    pass.output().sum().backward();
    Map<Parameter, Tensor> gradients = pass.parameterGradients();
    // Consume gradients before closing the context.
}
```

Gradient tensors returned by `parameterGradients()` are borrowed from the
autograd context. They remain valid only until that context is closed. This
adapter does not mutate `Tensor.grad()` or take ownership of module parameters.

Quantized weights, non-F32 tensors, batched matmul beyond rank two, and
non-Linear modules remain unsupported by this adapter. Normal
`NNModule.forward(Tensor)` inference and training behavior is unchanged.
