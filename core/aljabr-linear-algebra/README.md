# Aljabr Linear Algebra

Initial CPU reference implementation for dense double-precision linear algebra.

## Included API

- `Vector`: immutable dense vector, arithmetic, dot product and Euclidean norm.
- `Matrix`: immutable dense row-major matrix, arithmetic, transpose, matrix/vector multiplication and infinity norm.
- `LUDecomposition`: partial-pivoting LU factorization, determinant and vector/matrix right-hand-side solves.
- `SingularMatrixException`: explicit failure when a pivot is below the configured numerical threshold.

## Example

```java
Matrix a = Matrix.of(new double[][]{{0, 2}, {1, 1}});
Vector b = Vector.of(4, 3);
Vector x = LUDecomposition.factor(a).solve(b);
```

The module depends only on `:core:aljabr-numerics`. It is a transparent CPU reference
path, not an optimized BLAS implementation. QR, Cholesky, SVD, eigenvalue and sparse
algorithms are subsequent milestones.
