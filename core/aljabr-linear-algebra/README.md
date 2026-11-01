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
- `HessenbergReduction`: orthogonal Householder similarity reduction of general real square matrices to upper Hessenberg form.
- `RealSchurDecomposition`: shifted-QR real Schur factorization with orthogonal Q and quasi-upper-triangular T.
- `ConditionNumber`: spectral 2-norm condition number, reciprocal condition number and numerical-rank diagnostics derived from SVD.
- `SingularMatrixException`: explicit failure for singular LU pivots.

## Examples

```java
Matrix a = Matrix.of(new double[][]{{0, 2}, {1, 1}});
Vector b = Vector.of(4, 3);
Vector x = LUDecomposition.factor(a).solve(b);

ConditionNumber diagnostics = ConditionNumber.of(a);
double condition = diagnostics.twoNorm();

HessenbergReduction reduction = HessenbergReduction.reduce(a);
Matrix h = reduction.hessenberg();
```

Hessenberg reduction returns orthogonal Q and H satisfying `H = Q^T * A * Q`; reconstruction is `A = Q * H * Q^T`. It is the preprocessing stage for general real Schur/eigenvalue algorithms, not itself a general eigensolver. For symmetric input, the result is tridiagonal up to round-off.

Condition number is positive infinity when the matrix is numerically rank deficient under the supplied tolerance. Rectangular matrices are supported; full rank means rank equals `min(rows, columns)`.

The module depends only on `:core:aljabr-numerics`. These are transparent CPU reference algorithms, not optimized BLAS/LAPACK replacements. General real Schur iteration, sparse matrix storage/solvers, pivoted QR and iterative Krylov methods remain future milestones.
