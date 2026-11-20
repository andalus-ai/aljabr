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
