package tech.kayys.aljabr.models;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class UNetSkipConnectionsTest {

    @Test
    void providesLifoBorrowedFeatureTrackingWithoutTakingTensorOwnership() {
        CpuBackend backend = new CpuBackend();
        Tensor first = TensorFactory.of(new float[4], backend, 1, 1, 2, 2);
        Tensor second = TensorFactory.of(new float[4], backend, 1, 1, 2, 2);
        try {
            UNetSkipConnections skips = new UNetSkipConnections();
            skips.push(first);
            skips.push(second);

            assertThat(skips.size()).isEqualTo(2);
            assertThat(skips.pop()).isSameAs(second);
            assertThat(skips.pop()).isSameAs(first);
            assertThat(skips).matches(UNetSkipConnections::isEmpty);
            skips.requireEmpty();

            // The stack only tracks references; tensors remain usable/owned by caller.
            assertThat(first.isReleased()).isFalse();
            assertThat(second.isReleased()).isFalse();
        } finally {
            first.release();
            second.release();
        }
    }

    @Test
    void rejectsUnderflowAndUnconsumedSkips() {
        UNetSkipConnections skips = new UNetSkipConnections();
        assertThatThrownBy(skips::pop)
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("No encoder skip");
        assertThatThrownBy(skips::requireEmpty)
                .doesNotThrowAnyException();
    }
}
