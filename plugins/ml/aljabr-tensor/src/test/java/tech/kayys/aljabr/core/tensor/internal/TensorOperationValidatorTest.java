package tech.kayys.aljabr.core.tensor.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.diagnostics.Diagnostic;

class TensorOperationValidatorTest {


    @Test
    void reshapeRequiresAnExplicitNonNegativeShapeWithTheSameElementCount() {
        Shape output = TensorOperationValidator.validateReshape(new Shape(2, 3, 4), 4, 6);
        assertThat(output.dims()).containsExactly(4, 6);

        Shape scalar = TensorOperationValidator.validateReshape(new Shape(1), new long[0]);
        assertThat(scalar.rank()).isZero();
        Shape empty = TensorOperationValidator.validateReshape(new Shape(2, 0, 3), 0, 6);
        assertThat(empty.numel()).isZero();

        assertThatThrownBy(() -> TensorOperationValidator.validateReshape(new Shape(2, 3), 4, 2))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("reshape must preserve the number of elements: input=6, output=8");
        assertThatThrownBy(() -> TensorOperationValidator.validateReshape(new Shape(2, 3), 3, -1))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Tensor dimensions must be non-negative");
        assertThatThrownBy(() -> TensorOperationValidator.validateReshape(new Shape(2, 3), 2, 2, 2))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("reshape must preserve the number of elements");
        assertThatThrownBy(() -> TensorOperationValidator.validateReshape(new Shape(2, 3), Long.MAX_VALUE, 2))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("element count overflows long");
    }

    @Test
    void unsqueezeNormalizesNegativeAxesAndBuildsExpectedOutputShape() {
        assertThat(TensorOperationValidator.validateUnsqueeze(new Shape(2, 3, 4), 0).outputShape().dims())
                .containsExactly(1, 2, 3, 4);
        assertThat(TensorOperationValidator.validateUnsqueeze(new Shape(2, 3, 4), -4).axis())
                .isZero();
        assertThat(TensorOperationValidator.validateUnsqueeze(new Shape(2, 3, 4), -1).outputShape().dims())
                .containsExactly(2, 3, 4, 1);
        assertThat(TensorOperationValidator.validateUnsqueeze(new Shape(2, 3, 4), 3).outputShape().dims())
                .containsExactly(2, 3, 4, 1);
        assertThat(TensorOperationValidator.validateUnsqueeze(new Shape(), -1).outputShape().dims())
                .containsExactly(1);
    }

    @Test
    void unsqueezeRejectsAxesOutsideRankPlusOneRange() {
        assertThatThrownBy(() -> TensorOperationValidator.validateUnsqueeze(new Shape(2, 3), -4))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("unsqueeze axis -4 is out of range for tensor rank 2");
        assertThatThrownBy(() -> TensorOperationValidator.validateUnsqueeze(new Shape(2, 3), 3))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("unsqueeze axis 3 is out of range for tensor rank 2");
    }

    @Test
    void squeezeRemovesAllSingletonDimensionsAndPreservesLegacyFallback() {
        assertThat(TensorOperationValidator.validateSqueeze(new Shape(1, 2, 1, 3)).dims())
                .containsExactly(2, 3);
        assertThat(TensorOperationValidator.validateSqueeze(new Shape(2, 0, 1, 3)).dims())
                .containsExactly(2, 0, 3);
        assertThat(TensorOperationValidator.validateSqueeze(new Shape(2, 3)).dims())
                .containsExactly(2, 3);
        assertThat(TensorOperationValidator.validateSqueeze(new Shape(1, 1)).dims())
                .containsExactly(1);
        assertThat(TensorOperationValidator.validateSqueeze(new Shape()).dims())
                .containsExactly(1);
    }

    @Test
    void sliceValidatesRankBoundsAndReturnsRequestedShape() {
        Shape output = TensorOperationValidator.validateSlice(
                new Shape(4, 5, 6), new long[] { 1, 0, 2 }, new long[] { 3, 5, 4 });
        assertThat(output.dims()).containsExactly(3, 5, 4);

        Shape empty = TensorOperationValidator.validateSlice(
                new Shape(4, 5), new long[] { 4, 1 }, new long[] { 0, 2 });
        assertThat(empty.dims()).containsExactly(0, 2);

        assertThatThrownBy(() -> TensorOperationValidator.validateSlice(
                new Shape(), new long[0], new long[0]))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("Cannot slice a scalar tensor");
        assertThatThrownBy(() -> TensorOperationValidator.validateSlice(
                new Shape(2, 3), new long[] { 0 }, new long[] { 1 }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("Slice offsets and sizes must both have one entry per tensor dimension");
        assertThatThrownBy(() -> TensorOperationValidator.validateSlice(
                new Shape(2, 3), new long[] { -1, 0 }, new long[] { 1, 1 }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("offset=-1");
        assertThatThrownBy(() -> TensorOperationValidator.validateSlice(
                new Shape(2, 3), new long[] { 0, 2 }, new long[] { 1, 2 }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("dimension 1");
        assertThatThrownBy(() -> TensorOperationValidator.validateSlice(
                new Shape(2, 3), new long[] { 3, 0 }, new long[] { 0, 1 }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("offset=3");
        assertThatThrownBy(() -> TensorOperationValidator.validateSlice(
                new Shape(2, 3), new long[] { 0, 0 }, new long[] { -1, 1 }))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("dimensions must be non-negative");
    }

    @Test
    void reductionNormalizesNegativeAxisAndPreservesReductionSize() {
        TensorOperationValidator.ReductionPlan plan = TensorOperationValidator.validateReduction(
                new Shape(2, 3, 4), -1, false);

        assertThat(plan.axis()).isEqualTo(2);
        assertThat(plan.reductionSize()).isEqualTo(4);
    }

    @Test
    void reductionRejectsOutOfRangeAndScalarAxes() {
        assertThatThrownBy(() -> TensorOperationValidator.validateReduction(
                new Shape(2, 3), 2, false))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Reduction axis 2 is out of range");
        assertThatThrownBy(() -> TensorOperationValidator.validateReduction(
                new Shape(), 0, false))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("tensor rank 0");
    }

    @Test
    void meanRejectsEmptyReductionDimensionButSumAllowsIt() {
        assertThatThrownBy(() -> TensorOperationValidator.validateReduction(
                new Shape(2, 0, 3), 1, true))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("Mean is undefined for an empty reduction dimension");

        TensorOperationValidator.ReductionPlan sum = TensorOperationValidator.validateReduction(
                new Shape(2, 0, 3), 1, false);
        assertThat(sum.axis()).isEqualTo(1);
        assertThat(sum.reductionSize()).isZero();
    }

    @Test
    void matmulReportsContractedDimensionMismatch() {
        List<Diagnostic> diagnostics = new ArrayList<>();

        assertThatThrownBy(() -> TensorOperationValidator.validateMatmul(
                new Shape(2, 3), new Shape(4, 5), diagnostics::add))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Dimension mismatch");

        assertThat(diagnostics).hasSize(1);
        Diagnostic diagnostic = diagnostics.getFirst();
        assertThat(diagnostic.code()).isEqualTo("tensor.operation.dimension-mismatch");
        assertThat(diagnostic.category()).isEqualTo("tensor.operation.validation");
        assertThat(diagnostic.context())
                .containsEntry("operation", "matmul")
                .containsEntry("axis", "contracted")
                .containsEntry("expectedDimension", "3")
                .containsEntry("actualDimension", "4");
    }

    @Test
    void matmulReportsIncompatibleBatchDimensions() {
        List<Diagnostic> diagnostics = new ArrayList<>();

        assertThatThrownBy(() -> TensorOperationValidator.validateMatmul(
                new Shape(2, 3, 4), new Shape(5, 4, 6), diagnostics::add))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not broadcast-compatible");

        assertThat(diagnostics).hasSize(1);
        assertThat(diagnostics.getFirst().code()).isEqualTo("tensor.operation.broadcast-mismatch");
        assertThat(diagnostics.getFirst().context())
                .containsEntry("operation", "matmul")
                .containsEntry("axis", "0")
                .containsEntry("leftDimension", "2")
                .containsEntry("rightDimension", "5");
    }

    @Test
    void broadcastReportsIncompatibleTrailingDimension() {
        List<Diagnostic> diagnostics = new ArrayList<>();

        assertThatThrownBy(() -> TensorOperationValidator.validateBroadcast(
                new Shape(2, 3), new Shape(2, 4), diagnostics::add))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not broadcast-compatible");

        assertThat(diagnostics).hasSize(1);
        assertThat(diagnostics.getFirst().code()).isEqualTo("tensor.operation.broadcast-mismatch");
        assertThat(diagnostics.getFirst().context())
                .containsEntry("operation", "broadcast")
                .containsEntry("axis", "1")
                .containsEntry("leftDimension", "3")
                .containsEntry("rightDimension", "4");
    }

    @Test
    void successfulMatmulAndBroadcastValidationDoNotInvokeDiagnosticSink() {
        AtomicInteger calls = new AtomicInteger();

        TensorOperationValidator.MatmulPlan matmul = TensorOperationValidator.validateMatmul(
                new Shape(2, 3, 4), new Shape(1, 4, 5), diagnostic -> calls.incrementAndGet());
        TensorOperationValidator.BroadcastPlan broadcast = TensorOperationValidator.validateBroadcast(
                new Shape(2, 1, 4), new Shape(1, 3, 4), diagnostic -> calls.incrementAndGet());

        assertThat(matmul.outputDimensions()).containsExactly(2, 3, 5);
        assertThat(broadcast.outputDimensions()).containsExactly(2, 3, 4);
        assertThat(calls).hasValue(0);
    }

    @Test
    void legacyValidationOverloadsRemainSilent() {
        assertThatThrownBy(() -> TensorOperationValidator.validateMatmul(
                new Shape(2, 3), new Shape(4, 5)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Dimension mismatch");

        assertThatThrownBy(() -> TensorOperationValidator.validateBroadcast(
                new Shape(2, 3), new Shape(2, 4)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("not broadcast-compatible");
    }
}
