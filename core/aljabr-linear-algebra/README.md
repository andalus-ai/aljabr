# Aljabr Linear Algebra

Dense double-precision CPU reference implementations for foundational linear algebra.

## Included API

- `Vector`: immutable dense vector arithmetic, dot product and Euclidean norm.
- `Matrix`: immutable dense row-major matrix arithmetic, transpose, multiplication and infinity norm.
- `LUDecomposition`: partial-pivoting LU factorization, determinant and vector/matrix right-hand-side solves.
- `QRDecomposition`: Householder QR for tall and square matrices, thin Q and full-column-rank least-squares solve.
- `CholeskyDecomposition`: factorization and solve for symmetric positive-definite matrices.
- `SingularValueDecomposition`: one-sided Jacobi thin SVD for tall, square and wide matrices, numerical rank and pseudoinverse solve.
- `SymmetricEigenDecomposition`: Jacobi eigendecomposition for real symmetric matrices.
- `ConditionNumber`: spectral 2-norm condition number, reciprocal condition number and numerical-rank diagnostics derived from SVD.
- `SingularMatrixException`: explicit failure for singular LU pivots.

## Examples

```java
Matrix a = Matrix.of(new double[][]{{0, 2}, {1, 1}});
Vector b = Vector.of(4, 3);
Vector x = LUDecomposition.factor(a).solve(b);

ConditionNumber diagnostics = ConditionNumber.of(a);
double condition = diagnostics.twoNorm();
```

Condition number is positive infinity when the matrix is numerically rank deficient under the supplied tolerance. Rectangular matrices are supported; full rank means rank equals `min(rows, columns)`.

The module depends only on `:core:aljabr-numerics`. These are transparent CPU reference algorithms, not optimized BLAS/LAPACK replacements. General non-symmetric eigenvalue algorithms, sparse matrix storage/solvers, pivoted QR and iterative Krylov methods remain future milestones.
