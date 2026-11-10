# Aljabr Numerics

The dependency-light numerical foundation shared by Aljabr algorithms.

## Initial API

- `NumericalPrecision`: common FLOAT32/FLOAT64 precision profiles and comparison tolerances.
- `Tolerance`: combined absolute/relative floating-point comparison.
- `NumericalChecks`: finite and positive-finite boundary validation.
- `NonFinitePolicy`: explicit NaN/infinity acceptance policy.
- `RandomSource` and `SplitMix64`: deterministic, seedable random streams for reproducible algorithms.

## Example

```java
import tech.kayys.aljabr.numerics.*;

RandomSource random = new SplitMix64(2026L);
double sample = random.nextDouble();
boolean converged = NumericalPrecision.FLOAT64.defaultTolerance()
        .isClose(1.0, 1.0 + 1.0e-11);
```

This module deliberately has no dependency on tensors, neural networks, model formats,
inference runtimes, or external math libraries. It establishes shared contracts; it is
not yet a linear algebra or statistical algorithm implementation. The generator is
not suitable for cryptographic purposes.
