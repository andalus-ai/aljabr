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
- `MomentumSGD`: SGD with per-parameter momentum buffers and optional Nesterov lookahead.
- `RMSProp`: adaptive learning rates from an exponentially decaying average of squared gradients.
- `Adagrad`: adaptive learning rates from a cumulative sum of squared gradients.
- `AdamW`: adaptive moments with decoupled weight decay.
- `Nadam`: Adam-style adaptive moments with Dozat-formulation Nesterov lookahead.\n- `Adadelta`: adaptive updates using accumulated squared gradients and parameter updates.

All optimizers consume stable, fully qualified parameter names and gradient
maps. Missing gradients and frozen parameters are skipped. Gradient shape,
dtype, backend, and device are validated before parameter replacement. Stateful
optimizer state is explicitly released by `reset()`; `MomentumSGD`, `RMSProp`,
`Adagrad`, `Nadam`, `Adadelta`, and `AdamW` are also `AutoCloseable`.

This is an initial F32-oriented API built on the currently supported tensor
arithmetic. RMSProp currently provides the uncentered, no-momentum form; centered statistics,
optimizer momentum, and weight decay are not included. Adagrad currently omits
learning-rate decay and weight decay. Nadam currently implements the Dozat
lookahead formulation without weight decay or configurable schedules. Mixed
precision, gradient scaling, sparse gradients, parameter groups, schedulers,
checkpoint serialization, and distributed optimizer semantics are not claimed.
