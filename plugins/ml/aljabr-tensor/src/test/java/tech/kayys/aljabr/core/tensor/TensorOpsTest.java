package tech.kayys.aljabr.core.tensor;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.internal.TensorOperationValidator;
import tech.kayys.aljabr.diagnostics.Diagnostic;

import org.junit.jupiter.api.Test;

class TensorOpsTest {

    private final ComputeBackend backend = new CpuBackend();

    private Tensor tensor(float[] data, long... shape) {
        return TensorFactory.of(data, backend, shape);
    }


    @Test
    void catConcatenatesAlongFirstDimension() {
        Tensor a = tensor(new float[] { 1, 2, 3, 4 }, 2, 2);
        Tensor b = tensor(new float[] { 5, 6 }, 1, 2);
        try {
            Tensor result = TensorOps.cat(List.of(a, b), 0);
            try {
                assertThat(result.shape().dims()).containsExactly(3, 2);
                assertThat(result.backend()).isSameAs(backend);
                assertThat(result.toFloatArray()).containsExactly(1, 2, 3, 4, 5, 6);
            } finally {
                result.release();
            }
        } finally {
            a.release();
            b.release();
        }
    }

    @Test
    void catConcatenatesAlongInnerDimensionAndSupportsNegativeAxis() {
        Tensor a = tensor(new float[] { 1, 2, 3, 4 }, 2, 2);
        Tensor b = tensor(new float[] { 5, 6, 7, 8 }, 2, 2);
        try {
            Tensor result = TensorOps.cat(List.of(a, b), -1);
            try {
                assertThat(result.shape().dims()).containsExactly(2, 4);
                assertThat(result.backend()).isSameAs(backend);
                assertThat(result.toFloatArray()).containsExactly(1, 2, 5, 6, 3, 4, 7, 8);
            } finally {
                result.release();
            }
        } finally {
            a.release();
            b.release();
        }
    }

    @Test
    void catRejectsEmptyListInvalidAxisAndIncompatibleShapes() {
        assertThatThrownBy(() -> TensorOps.cat(List.of(), 0))
                .isInstanceOf(IllegalArgumentException.class);
        Tensor a = tensor(new float[] { 1, 2 }, 1, 2);
        Tensor wrongShape = tensor(new float[] { 3, 4, 5 }, 1, 3);
        try {
            assertThatThrownBy(() -> TensorOps.cat(List.of(a), 2))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("out of range");
            assertThatThrownBy(() -> TensorOps.cat(List.of(a, wrongShape), 0))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("shape mismatch");
        } finally {
            a.release();
            wrongShape.release();
        }
    }

    @Test
    void einsumComputesMatrixMultiplication() {
        Tensor a = tensor(new float[] { 1, 2, 3, 4 }, 2, 2);
        Tensor b = tensor(new float[] { 5, 6, 7, 8 }, 2, 2);
        try {
            Tensor result = TensorOps.einsum("ij,jk->ik", a, b);
            try {
                assertThat(result.shape().dims()).containsExactly(2, 2);
                assertThat(result.backend()).isSameAs(backend);
                assertThat(result.toFloatArray()).containsExactly(19, 22, 43, 50);
            } finally {
                result.release();
            }
        } finally {
            a.release();
            b.release();
        }
    }

    @Test
    void einsumComputesDotProductAndScalarReduction() {
        Tensor a = tensor(new float[] { 1, 2, 3 }, 3);
        Tensor b = tensor(new float[] { 4, 5, 6 }, 3);
        try {
            Tensor dot = TensorOps.einsum("i,i->", a, b);
            try {
                assertThat(dot.shape().dims()).isEmpty();
                assertThat(dot.item()).isEqualTo(32.0f);
            } finally {
                dot.release();
            }

            Tensor sum = TensorOps.einsum("i->", a);
            try {
                assertThat(sum.item()).isEqualTo(6.0f);
            } finally {
                sum.release();
            }
        } finally {
            a.release();
            b.release();
        }
    }

    @Test
    void einsumSupportsTransposeAndOuterProduct() {
        Tensor matrix = tensor(new float[] { 1, 2, 3, 4 }, 2, 2);
        Tensor left = tensor(new float[] { 1, 2 }, 2);
        Tensor right = tensor(new float[] { 3, 4, 5 }, 3);
        try {
            Tensor transpose = TensorOps.einsum("ij->ji", matrix);
            try {
                assertThat(transpose.shape().dims()).containsExactly(2, 2);
                assertThat(transpose.toFloatArray()).containsExactly(1, 3, 2, 4);
            } finally {
                transpose.release();
            }

            Tensor outer = TensorOps.einsum("i,j->ij", left, right);
            try {
                assertThat(outer.shape().dims()).containsExactly(2, 3);
                assertThat(outer.toFloatArray()).containsExactly(3, 4, 5, 6, 8, 10);
            } finally {
                outer.release();
            }
        } finally {
            matrix.release();
            left.release();
            right.release();
        }
    }


    @Test
    void catRejectsOperandsFromDifferentBackendInstances() {
        Tensor a = tensor(new float[] { 1, 2 }, 1, 2);
        Tensor b = TensorFactory.of(new float[] { 3, 4 }, new CpuBackend(), 1, 2);
        try {
            assertThatThrownBy(() -> TensorOps.cat(List.of(a, b), 0))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("compute backend mismatch");
        } finally {
            a.release();
            b.release();
        }
    }

    @Test
    void einsumRejectsOperandsFromDifferentBackendInstances() {
        Tensor a = tensor(new float[] { 1, 2 }, 2);
        Tensor b = TensorFactory.of(new float[] { 3, 4 }, new CpuBackend(), 2);
        try {
            assertThatThrownBy(() -> TensorOps.einsum("i,i->", a, b))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("same compute backend instance");
        } finally {
            a.release();
            b.release();
        }
    }

    @Test
    void einsumRejectsMalformedEquationsAndIncompatibleDimensions() {
        Tensor a = tensor(new float[] { 1, 2, 3, 4 }, 2, 2);
        Tensor aVector = tensor(new float[] { 1, 2, 3, 4 }, 4);
        Tensor wrongShape = tensor(new float[] { 1, 2, 3 }, 3);
        try {
            assertThatThrownBy(() -> TensorOps.einsum("ij,jk", a, a))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("explicit");
            assertThatThrownBy(() -> TensorOps.einsum("ii->i", a))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("repeated label");
            assertThatThrownBy(() -> TensorOps.einsum("i,i->i", aVector, wrongShape))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("dimension mismatch");
            assertThatThrownBy(() -> TensorOps.einsum("i->i", a))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("rank");
        } finally {
            a.release();
            wrongShape.release();
        }
    }

    @Test
    void catReportsShapeMismatchWithStructuredContext() {
        Tensor a = tensor(new float[] {1, 2}, 1, 2);
        Tensor b = tensor(new float[] {3, 4, 5}, 1, 3);
        List<Diagnostic> diagnostics = new ArrayList<>();
        try {
            assertThatThrownBy(() -> TensorOps.cat(List.of(a, b), 0, diagnostics::add))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("shape mismatch");
            assertThat(diagnostics).hasSize(1);
            Diagnostic diagnostic = diagnostics.getFirst();
            assertThat(diagnostic.code()).isEqualTo("tensor.operation.shape-mismatch");
            assertThat(diagnostic.category()).isEqualTo("tensor.operation.validation");
            assertThat(diagnostic.context())
                    .containsEntry("operation", "cat")
                    .containsEntry("operandIndex", "1")
                    .containsEntry("axis", "1")
                    .containsEntry("expectedDimension", "2")
                    .containsEntry("actualDimension", "3");
        } finally {
            a.release();
            b.release();
        }
    }

    @Test
    void einsumReportsDimensionMismatchWithStructuredContext() {
        Tensor a = tensor(new float[] {1, 2, 3, 4}, 2, 2);
        Tensor b = tensor(new float[] {5, 6, 7}, 3);
        List<Diagnostic> diagnostics = new ArrayList<>();
        try {
            assertThatThrownBy(() -> TensorOps.einsum("ij,j->i", diagnostics::add, a, b))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("dimension mismatch");
            assertThat(diagnostics).hasSize(1);
            Diagnostic diagnostic = diagnostics.getFirst();
            assertThat(diagnostic.code()).isEqualTo("tensor.operation.dimension-mismatch");
            assertThat(diagnostic.context())
                    .containsEntry("operation", "einsum")
                    .containsEntry("operandIndex", "1")
                    .containsEntry("axis", "0")
                    .containsEntry("label", "j")
                    .containsEntry("expectedDimension", "2")
                    .containsEntry("actualDimension", "3");
        } finally {
            a.release();
            b.release();
        }
    }

    @Test
    void successfulOperationsDoNotInvokeDiagnosticSink() {
        Tensor a = tensor(new float[] {1, 2}, 1, 2);
        Tensor b = tensor(new float[] {3, 4}, 1, 2);
        AtomicInteger calls = new AtomicInteger();
        try {
            Tensor result = TensorOps.cat(List.of(a, b), 0, diagnostic -> calls.incrementAndGet());
            try {
                assertThat(calls).hasValue(0);
            } finally {
                result.release();
            }
        } finally {
            a.release();
            b.release();
        }
    }


    @Test
    void defaultTensorCatUsesSharedValidationAndReportsContractFailure() {
        Tensor a = tensor(new float[] {1, 2}, 1, 2);
        Tensor b = tensor(new float[] {3, 4, 5}, 1, 3);
        List<Diagnostic> diagnostics = new ArrayList<>();
        try {
            assertThatThrownBy(() -> DefaultTensor.cat(List.of(a, b), 0, diagnostics::add))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("shape mismatch");
            assertThat(diagnostics).hasSize(1);
            assertThat(diagnostics.getFirst().code()).isEqualTo("tensor.operation.shape-mismatch");
            assertThat(diagnostics.getFirst().context())
                    .containsEntry("operation", "cat")
                    .containsEntry("operandIndex", "1")
                    .containsEntry("axis", "1");
        } finally {
            a.release();
            b.release();
        }
    }

    @Test
    void sharedMatmulShapeValidationBuildsBroadcastedBatchOutputShape() {
        TensorOperationValidator.MatmulPlan plan =
                TensorOperationValidator.validateMatmul(new Shape(2, 1, 3, 4), new Shape(1, 5, 4, 6));
        assertThat(plan.batchDimensions()).containsExactly(2, 5);
        assertThat(plan.outputDimensions()).containsExactly(2, 5, 3, 6);
        assertThat(plan.rows()).isEqualTo(3);
        assertThat(plan.inner()).isEqualTo(4);
        assertThat(plan.columns()).isEqualTo(6);
    }

    @Test
    void sharedMatmulShapeValidationRejectsRankAndContractionMismatches() {
        assertThatThrownBy(() -> TensorOperationValidator.validateMatmul(new Shape(4), new Shape(4, 2)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("at least 2D");
        assertThatThrownBy(() -> TensorOperationValidator.validateMatmul(new Shape(2, 3), new Shape(4, 5)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Dimension mismatch");
    }

    @Test
    void sharedMatmulShapeValidationRejectsIncompatibleBatchDimensions() {
        assertThatThrownBy(() -> TensorOperationValidator.validateMatmul(
                new Shape(2, 3, 4), new Shape(5, 4, 6)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("batch dimensions are not broadcast-compatible");
    }

    @Test
    void cpuMatmulUsesSharedBroadcastShapeAndPreservesBatchedResults() {
        Tensor a = tensor(new float[] {1, 0, 0, 1, 2, 0, 0, 2}, 2, 1, 2, 2);
        Tensor b = tensor(new float[] {1, 2, 3, 4, 5, 6}, 1, 3, 2, 1);
        try {
            Tensor result = backend.matmul(a, b);
            try {
                assertThat(result.shape().dims()).containsExactly(2, 3, 2, 1);
                assertThat(result.toFloatArray()).containsExactly(1, 2, 3, 4, 5, 6, 2, 4, 6, 8, 10, 12);
            } finally {
                result.release();
            }
        } finally {
            a.release();
            b.release();
        }
    }

    @Test
    void sharedBroadcastValidationBuildsTrailingAxisOutputShape() {
        TensorOperationValidator.BroadcastPlan plan =
                TensorOperationValidator.validateBroadcast(new Shape(2, 1, 3), new Shape(1, 4, 1));
        assertThat(plan.outputDimensions()).containsExactly(2, 4, 3);
        assertThat(plan.outputElements()).isEqualTo(24);
    }

    @Test
    void sharedBroadcastValidationSupportsScalarsAndZeroDimensions() {
        TensorOperationValidator.BroadcastPlan scalar =
                TensorOperationValidator.validateBroadcast(new Shape(), new Shape(2, 3));
        assertThat(scalar.outputDimensions()).containsExactly(2, 3);
        assertThat(scalar.outputElements()).isEqualTo(6);

        TensorOperationValidator.BroadcastPlan empty =
                TensorOperationValidator.validateBroadcast(new Shape(0, 3), new Shape(1, 3));
        assertThat(empty.outputDimensions()).containsExactly(0, 3);
        assertThat(empty.outputElements()).isZero();
    }

    @Test
    void sharedBroadcastValidationRejectsIncompatibleDimensionsAndOverflow() {
        assertThatThrownBy(() -> TensorOperationValidator.validateBroadcast(
                new Shape(2, 3), new Shape(4, 3)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not broadcast-compatible");
        assertThatThrownBy(() -> TensorOperationValidator.validateBroadcast(
                new Shape(Long.MAX_VALUE, 2), new Shape(1, 2)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("overflows");
    }

    @Test
    void cpuElementwiseOperationsUseSharedBroadcastPlan() {
        Tensor matrix = tensor(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor row = tensor(new float[] {10, 20, 30}, 1, 3);
        try {
            Tensor sum = backend.add(matrix, row);
            Tensor difference = backend.sub(matrix, row);
            Tensor product = backend.mul(matrix, row);
            Tensor quotient = backend.div(matrix, row);
            try {
                assertThat(sum.shape().dims()).containsExactly(2, 3);
                assertThat(sum.toFloatArray()).containsExactly(11, 22, 33, 14, 25, 36);
                assertThat(difference.toFloatArray()).containsExactly(-9, -18, -27, -6, -15, -24);
                assertThat(product.toFloatArray()).containsExactly(10, 40, 90, 40, 100, 180);
                assertThat(quotient.toFloatArray()).containsExactly(0.1f, 0.1f, 0.1f, 0.4f, 0.25f, 0.2f);
            } finally {
                sum.release();
                difference.release();
                product.release();
                quotient.release();
            }
        } finally {
            matrix.release();
            row.release();
        }
    }

}
