# Aljabr Optimizers

This optional module provides parameter-update algorithms independently of
autograd and training orchestration.

## Dependency boundary

- Depends on `aljabr-nn` for stable `Parameter` handles and tensor operations.
- Does not depend on `aljabr-autograd`; gradients may come from eager autograd,
  another differentiation implementation, or an externally supplied gradient map.
- Does not own model parameters. Updates are published through
  `Parameter.updateTensor`.
- Does not own training loops, datasets, metrics, or experiment lifecycle; those
  belong in Tafkir.

## Initial algorithms

- `SGD`: stateless gradient descent.
- `AdamW`: adaptive moments with decoupled weight decay.

Both consume stable, fully qualified parameter names and gradient maps. Missing
gradients and frozen parameters are skipped. Gradient shape, dtype, backend,
and device are validated before parameter replacement. Optimizer state is
explicitly released by `reset()`; `AdamW` is also `AutoCloseable`.

This is an initial F32-oriented API built on the currently supported tensor
arithmetic. Mixed precision, gradient scaling, sparse gradients, parameter
groups, schedulers, checkpoint serialization, and distributed optimizer
semantics are not claimed.
