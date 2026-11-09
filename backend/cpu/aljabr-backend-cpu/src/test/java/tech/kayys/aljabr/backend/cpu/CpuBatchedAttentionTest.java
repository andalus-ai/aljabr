package tech.kayys.aljabr.backend.cpu;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CpuBatchedAttentionTest {

    @Test
    void computesIndependentAttentionForEachBatchAndHead() {
        CpuBackend backend = new CpuBackend();
        Tensor query = TensorFactory.of(new float[8], backend, 1, 2, 2, 2);
        Tensor key = TensorFactory.of(new float[12], backend, 1, 2, 3, 2);
        Tensor value = TensorFactory.of(new float[] {
                1, 3, 3, 5, 5, 7,
                2, 4, 4, 6, 6, 8
        }, backend, 1, 2, 3, 2);

        Tensor output = backend.attention(query, key, value);
        assertArrayEquals(new float[] {
                3, 5, 3, 5,
                4, 6, 4, 6
        }, output.toFloatArray(), 1e-5f);

        output.release();
        value.release();
        key.release();
        query.release();
    }

    @Test
    void rejectsIncompatibleBatchHeadAndFeatureDimensions() {
        CpuBackend backend = new CpuBackend();
        Tensor query = TensorFactory.of(new float[8], backend, 1, 2, 2, 2);
        Tensor key = TensorFactory.of(new float[12], backend, 1, 1, 3, 2);
        Tensor value = TensorFactory.of(new float[12], backend, 1, 2, 3, 2);

        assertThrows(IllegalArgumentException.class, () -> backend.attention(query, key, value));

        value.release();
        key.release();
        query.release();
    }

    @Test
    void additiveMaskSuppressesMaskedKeyAndCausalModeRestrictsFutureKeys() {
        CpuBackend backend = new CpuBackend();
        Tensor query = TensorFactory.of(new float[2], backend, 1, 1, 2, 1);
        Tensor key = TensorFactory.of(new float[2], backend, 1, 1, 2, 1);
        Tensor value = TensorFactory.of(new float[] {2f, 8f}, backend, 1, 1, 2, 1);
        Tensor mask = TensorFactory.of(new float[] {0f, Float.NEGATIVE_INFINITY}, backend, 2, 1);
        Tensor masked = null;
        Tensor causal = null;
        try {
            masked = backend.attention(query, key, value, mask, false);
            assertArrayEquals(new float[] {2f, 2f}, masked.toFloatArray(), 1e-6f);
            causal = backend.attention(query, key, value, null, true);
            assertArrayEquals(new float[] {2f, 5f}, causal.toFloatArray(), 1e-6f);
        } finally {
            if (causal != null) causal.release();
            if (masked != null) masked.release();
            mask.release(); value.release(); key.release(); query.release();
        }
    }

    @Test
    void returnsZeroWhenEveryKeyIsMasked() {
        CpuBackend backend = new CpuBackend();
        Tensor query = TensorFactory.of(new float[1], backend, 1, 1, 1, 1);
        Tensor key = TensorFactory.of(new float[2], backend, 1, 1, 2, 1);
        Tensor value = TensorFactory.of(new float[] {2f, 8f}, backend, 1, 1, 2, 1);
        Tensor mask = TensorFactory.of(new float[] {Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY}, backend, 1, 2);
        Tensor output = null;
        try {
            output = backend.attention(query, key, value, mask, false);
            assertArrayEquals(new float[] {0f}, output.toFloatArray(), 1e-6f);
        } finally {
            if (output != null) output.release();
            mask.release(); value.release(); key.release(); query.release();
        }
    }
}
