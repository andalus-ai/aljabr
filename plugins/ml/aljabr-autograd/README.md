# Aljabr Autograd

This module is an experimental, eager reverse-mode differentiation layer over
the public Aljabr `Tensor` API. It is intentionally separate from
`aljabr-tensor`: tensor operations do not depend on this module, and ordinary
tensor consumers do not pay an autograd dependency cost.

## Initial API

Create an `AutogradContext`, wrap caller-owned leaf tensors as `Variable`
instances, compose supported operations, and call `backward()` on a scalar
result or supply an upstream gradient for a non-scalar result:

```java
try (AutogradContext context = new AutogradContext()) {
    Variable x = context.variable(input);
    Variable loss = x.mul(x).mean();
    loss.backward();

    // For a non-scalar output, pass an F32 seed with the same shape/backend:
    // output.backward(upstreamGradient);

    Tensor gradient = x.grad();
    // Read or copy the gradient before closing the context.
}
```

## Current prototype contract

- Eager F32 tensors only.
- Caller owns leaf tensors and must keep them alive through backward.
- The context owns intermediate forward values and gradients; close it to
  release those resources.
- A variable and all operands must belong to the same context and use the same
  backend instance.
- Elementwise binary operations require identical shapes. Broadcasting and its
  backward reduction are not implemented.
- Matrix multiplication currently supports rank-two inputs.
- Implicit `backward()` is supported only for scalar outputs. Non-scalar outputs
  accept an explicit F32 upstream gradient with matching shape and backend; the
  context copies and owns the supplied gradient.
- A context runs a successful backward pass at most once. Invalid output-gradient
  metadata and no-gradient no-op calls do not consume the context.
- Gradients returned by `Variable.grad()` are borrowed from the context and
  must not be released independently.
- The context is single-use and not thread-safe. Do not share it across threads.
- Only the implemented operations have derivative rules. This is not a
  drop-in replacement for `Tensor.backward()`; the tensor module remains
  independent and continues to reject unsupported implicit differentiation.

The current operation set is add/subtract/multiply/divide, scalar add/multiply/
divide, rank-two matrix multiplication, sum, and mean. This list is deliberately
smaller than the forward Tensor API.

## Validation status

The module includes analytical CPU tests and a finite-difference check in its
test source. These tests have not yet been executed in this development step.
The module must not be described as validated until its Gradle test task and
the relevant CI checks pass.
