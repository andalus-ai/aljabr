package tech.kayys.aljabr.core.nn;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TransformerEncoderLayerTest {

    @Test
    void composesAttentionResidualFeedForwardAndLayerNorm() {
        CpuBackend backend = new CpuBackend();
        TransformerEncoderLayer layer = new TransformerEncoderLayer(backend, 4, 2, 8, 0.0f, 1.0e-12f);
        Tensor input = TensorFactory.of(new float[] {
                1, 0, 0, 1,
                0, 1, 1, 0
        }, backend, 1, 2, 4);
        Tensor mask = TensorFactory.of(new float[] {
                0, Float.NEGATIVE_INFINITY,
                0, 0
        }, backend, 2, 2);
        Tensor output = null;
        try {
            output = layer.forward(input, mask);
            assertEquals(3, output.shape().rank());
            assertEquals(1, output.shape().dim(0));
            assertEquals(2, output.shape().dim(1));
            assertEquals(4, output.shape().dim(2));
            assertEquals(backend, output.backend());
            for (float value : output.toFloatArray()) {
                assertTrue(Float.isFinite(value));
            }
        } finally {
            if (output != null) output.release();
            mask.release();
            input.release();
            layer.close();
        }
    }

    @Test
    void rejectsInvalidLayerDimensionsAndEpsilon() {
        CpuBackend backend = new CpuBackend();
        assertThrows(IllegalArgumentException.class,
                () -> new TransformerEncoderLayer(backend, 5, 2, 8, 0.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new TransformerEncoderLayer(backend, 4, 2, 8, 0.0f, 0.0f));
    }
}
