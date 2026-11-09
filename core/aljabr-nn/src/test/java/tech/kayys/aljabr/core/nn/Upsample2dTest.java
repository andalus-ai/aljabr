package tech.kayys.aljabr.core.nn;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class Upsample2dTest {

    @Test
    void nearestNeighborUpsamplingRepeatsSpatialSamples() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 1, 1, 2, 2);
        try {
            Tensor output = new Upsample2d(backend, 2.0f).forward(input);
            try {
                assertThat(output.shape().dims()).containsExactly(1, 1, 4, 4);
                assertThat(output.toFloatArray()).containsExactly(
                        1, 1, 2, 2,
                        1, 1, 2, 2,
                        3, 3, 4, 4,
                        3, 3, 4, 4);
                assertThat(output.backend()).isSameAs(backend);
            } finally {
                output.release();
            }
        } finally {
            input.release();
        }
    }

    @Test
    void fractionalScaleUsesFloorForOutputDimensions() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 1, 1, 2, 2);
        try {
            Tensor output = new Upsample2d(backend, 1.5f).forward(input);
            try {
                assertThat(output.shape().dims()).containsExactly(1, 1, 3, 3);
                assertThat(output.toFloatArray()).containsExactly(1, 1, 2, 1, 1, 2, 3, 3, 4);
            } finally {
                output.release();
            }
        } finally {
            input.release();
        }
    }

    @Test
    void rejectsNonNchwInputs() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[] {1, 2, 3, 4}, backend, 1, 4);
        try {
            assertThatThrownBy(() -> new Upsample2d(backend, 2.0f).forward(input))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("NCHW");
        } finally {
            input.release();
        }
    }
}
