package tech.kayys.aljabr.core.nn;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MultiheadAttentionTest {

    @Test
    void selfAttentionProjectsAndPreservesBatchSequenceEmbeddingShape() {
        CpuBackend backend = new CpuBackend();
        MultiheadAttention attention = new MultiheadAttention(backend, 4, 2);
        Tensor input = TensorFactory.of(new float[24], backend, 2, 3, 4);
        Tensor output = null;
        try {
            output = attention.forward(input);
            assertEquals(3, output.shape().rank());
            assertEquals(2, output.shape().dim(0));
            assertEquals(3, output.shape().dim(1));
            assertEquals(4, output.shape().dim(2));
            assertEquals(backend, output.backend());
            assertArrayEquals(new float[24], output.toFloatArray(), 1e-6f);
        } finally {
            if (output != null) output.release();
            input.release();
            attention.close();
        }
    }

    @Test
    void crossAttentionSupportsDifferentQueryAndContextSequenceLengths() {
        CpuBackend backend = new CpuBackend();
        MultiheadAttention attention = new MultiheadAttention(backend, 4, 2);
        Tensor query = TensorFactory.of(new float[8], backend, 1, 2, 4);
        Tensor key = TensorFactory.of(new float[12], backend, 1, 3, 4);
        Tensor value = TensorFactory.of(new float[12], backend, 1, 3, 4);
        Tensor output = null;
        try {
            output = attention.forward(query, key, value, null);
            assertEquals(1, output.shape().dim(0));
            assertEquals(2, output.shape().dim(1));
            assertEquals(4, output.shape().dim(2));
        } finally {
            if (output != null) output.release();
            value.release();
            key.release();
            query.release();
            attention.close();
        }
    }

    @Test
    void acceptsAdditiveMaskAndCausalAttention() {
        CpuBackend backend = new CpuBackend();
        MultiheadAttention attention = new MultiheadAttention(backend, 4, 2);
        Tensor input = TensorFactory.of(new float[8], backend, 1, 2, 4);
        Tensor mask = TensorFactory.of(new float[] {0f, Float.NEGATIVE_INFINITY, 0f, 0f}, backend, 2, 2);
        Tensor masked = null;
        Tensor causal = null;
        try {
            masked = attention.forward(input, input, input, mask);
            causal = attention.forward(input, input, input, null, true);
            assertEquals(3, masked.shape().rank());
            assertEquals(3, causal.shape().rank());
            assertEquals(backend, masked.backend());
            assertEquals(backend, causal.backend());
        } finally {
            if (causal != null) causal.release();
            if (masked != null) masked.release();
            mask.release();
            input.release();
            attention.close();
        }
    }
}
