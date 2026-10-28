# Java research quickstart: tensors and matrix operations

This example demonstrates the current Aljabr tensor API from ordinary Java. It is intentionally small: create input/weight tensors, run a matrix multiplication, reduce the result to a scalar, inspect values, and release the native/off-heap-backed resources.

> **Scope:** this is a tensor-computation example, not a complete neural-network training example. It does not claim that optimizer updates, end-to-end autograd, model checkpointing, or every backend are production-ready.

## Example

Save as `TensorResearchQuickstart.java` in a Java project that depends on the Aljabr tensor module:

```java
import java.util.Arrays;

import tech.kayys.aljabr.core.tensor.Tensor;

public final class TensorResearchQuickstart {
    private TensorResearchQuickstart() {}

    public static void main(String[] args) {
        Tensor inputs = Tensor.of(new float[] {
            1.0f, 2.0f,
            3.0f, 4.0f
        }, 2, 2);

        Tensor weights = Tensor.of(new float[] {
            0.5f, 1.0f,
            1.5f, 2.0f
        }, 2, 2);

        Tensor predictions = null;
        Tensor mean = null;
        try {
            // [2, 2] x [2, 2] -> [2, 2]
            predictions = inputs.matmul(weights);
            mean = predictions.mean();

            System.out.println("shape = " + predictions.shape());
            System.out.println("values = " + Arrays.toString(predictions.toFloatArray()));
            System.out.println("mean = " + mean.item());
        } finally {
            if (mean != null) mean.release();
            if (predictions != null) predictions.release();
            weights.release();
            inputs.release();
        }
    }
}
```

For the current reference CPU tensor semantics, the expected matrix values are `[3.5, 5.0, 7.5, 11.0]`, and their mean is `6.75`. Confirm backend support and numeric parity for the selected runtime before using a different device or dtype.

## Try a research variation

Change the input values or weights, then compare the output. For a small CPU reference operation, `TensorOps.einsum("ij,jk->ik", a, b)` also expresses matrix multiplication; its current implementation is documented as a correctness baseline, not an optimized large-tensor kernel.

When experimenting, record:
- Aljabr commit/version and JDK version.
- Tensor shapes, dtype, device and selected backend.
- Seeds and exact input data for reproducibility.
- Numerical tolerance and expected reference values.
- Whether setup, allocation, synchronization and copies are included in timings.

## Build and dependency note

Use JDK 25 and the Gradle module coordinates published for the version you select. In a checkout, inspect the actual module name and its dependencies with:

```bash
./gradlew projects
./gradlew :core:aljabr-tensor:test
```

The second command runs the repository's tensor-module tests; it does not compile this documentation snippet as a dedicated example target. Treat the snippet as an API illustration until it is added to an executable example/test source set and verified in CI.

## What comes next for a PyTorch-like workflow

A complete research workflow should build on this low-level surface with validated, composable layers:

1. Tensor operations with explicit shape/dtype/device semantics and resource ownership.
2. Differentiable operation coverage and gradient-parity tests.
3. `Module`/parameter abstractions and model composition.
4. Loss functions, optimizers and a tested training loop.
5. Dataset, batching, deterministic random generators and experiment metrics.
6. Checkpoint save/load and reproducible model export.
7. CPU reference tests plus per-backend correctness and performance suites.

These are capability milestones, not claims that every item is already complete. New examples should be executable in CI, not documentation-only pseudocode.
