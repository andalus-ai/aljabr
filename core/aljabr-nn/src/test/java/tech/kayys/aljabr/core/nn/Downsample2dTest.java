package tech.kayys.aljabr.core.nn;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class Downsample2dTest {

    @Test
    void reducesSpatialDimensionsByApproximatelyHalf() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[1 * 32 * 8 * 8], backend, 1, 32, 8, 8);
        try {
            Downsample2d layer = new Downsample2d(backend, 32);
            Tensor output = layer.forward(input);
            try {
                assertThat(output.shape().dims()).containsExactly(1, 32, 4, 4);
                assertThat(output.backend()).isSameAs(backend);
                assertThat(layer.convolution().stride()).isEqualTo(2);
                assertThat(layer.convolution().padding()).isEqualTo(1);
            } finally {
                output.release();
            }
        } finally {
            input.release();
        }
    }

    @Test
    void supportsChannelProjection() {
        CpuBackend backend = new CpuBackend();
        Tensor input = TensorFactory.of(new float[1 * 32 * 8 * 8], backend, 1, 32, 8, 8);
        try {
            Tensor output = new Downsample2d(backend, 32, 64).forward(input);
            try {
                assertThat(output.shape().dims()).containsExactly(1, 64, 4, 4);
            } finally {
                output.release();
            }
        } finally {
            input.release();
        }
    }

    @Test
    void rejectsInvalidInputShapeAndBackend() {
        CpuBackend backend = new CpuBackend();
        Tensor invalid = TensorFactory.of(new float[16], backend, 1, 4, 2, 2);
        try {
            assertThatThrownBy(() -> new Downsample2d(backend, 8).forward(invalid))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("8 channels");
        } finally {
            invalid.release();
        }

        CpuBackend other = new CpuBackend();
        Tensor input = TensorFactory.of(new float[1 * 8 * 4 * 4], other, 1, 8, 4, 4);
        try {
            assertThatThrownBy(() -> new Downsample2d(backend, 8).forward(input))
                    .isInstanceOf(IllegalArgumentException.class);
        } finally {
            input.release();
        }
    }
}
