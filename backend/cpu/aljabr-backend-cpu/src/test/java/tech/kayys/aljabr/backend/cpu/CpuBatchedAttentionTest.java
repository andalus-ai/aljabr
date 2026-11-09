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
        Tensor key = TensorFactory.of(new float[6], backend, 1, 1, 3, 2);
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
        Tensor mask = TensorFactory.of(new float[] {0f, Float.NEGATIVE_INFINITY, 0f, Float.NEGATIVE_INFINITY}, backend, 2, 2);
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

    @Test
    void singleExpOnlineSoftmaxHandlesRepeatedRunningMaximumUpdates() {
        CpuBackend backend = new CpuBackend();
        float[] logits = {0f, 2f, 1f, 3f, -1f};
        float[] values = {1f, 2f, 3f, 4f, 5f};
        Tensor query = TensorFactory.of(new float[] {1f}, backend, 1, 1, 1, 1);
        Tensor key = TensorFactory.of(logits, backend, 1, 1, logits.length, 1);
        Tensor value = TensorFactory.of(values, backend, 1, 1, values.length, 1);
        Tensor output = null;
        try {
            output = backend.attention(query, key, value);
            double maximum = Double.NEGATIVE_INFINITY;
            for (float logit : logits) maximum = Math.max(maximum, logit);
            double denominator = 0d;
            double weighted = 0d;
            for (int i = 0; i < logits.length; i++) {
                double weight = Math.exp(logits[i] - maximum);
                denominator += weight;
                weighted += weight * values[i];
            }
            assertArrayEquals(new float[] {(float) (weighted / denominator)},
                    output.toFloatArray(), 1e-6f);
        } finally {
            if (output != null) output.release();
            value.release();
            key.release();
            query.release();
        }
    }

    @Test
    void onlineSoftmaxMatchesStableReferenceForLongerKeySequence() {
        CpuBackend backend = new CpuBackend();
        float[] qValues = {0.7f, -0.3f};
        float[] kValues = {
                0.2f, 0.8f,
                -0.4f, 0.1f,
                0.9f, -0.2f,
                0.3f, 0.5f,
                -0.7f, -0.6f
        };
        float[] vValues = {
                1f, 2f,
                3f, -1f,
                -2f, 4f,
                0.5f, 0.25f,
                5f, -3f
        };
        Tensor query = TensorFactory.of(qValues, backend, 1, 1, 1, 2);
        Tensor key = TensorFactory.of(kValues, backend, 1, 1, 5, 2);
        Tensor value = TensorFactory.of(vValues, backend, 1, 1, 5, 2);
        Tensor output = null;
        try {
            output = backend.attention(query, key, value);
            double[] logits = new double[5];
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < 5; i++) {
                logits[i] = (qValues[0] * kValues[i * 2] + qValues[1] * kValues[i * 2 + 1])
                        / Math.sqrt(2d);
                max = Math.max(max, logits[i]);
            }
            double denominator = 0d;
            double expected0 = 0d;
            double expected1 = 0d;
            for (int i = 0; i < 5; i++) {
                double probability = Math.exp(logits[i] - max);
                denominator += probability;
                expected0 += probability * vValues[i * 2];
                expected1 += probability * vValues[i * 2 + 1];
            }
            assertArrayEquals(new float[] {
                    (float) (expected0 / denominator), (float) (expected1 / denominator)
            }, output.toFloatArray(), 1e-5f);
        } finally {
            if (output != null) output.release();
            value.release();
            key.release();
            query.release();
        }
    }
}
