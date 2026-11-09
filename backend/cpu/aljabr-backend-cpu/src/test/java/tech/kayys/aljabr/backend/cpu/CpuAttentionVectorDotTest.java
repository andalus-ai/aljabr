package tech.kayys.aljabr.backend.cpu;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CpuAttentionVectorDotTest {
    @Test
    void vectorDotHandlesLaneAlignedAndScalarTailDimensions() {
        CpuBackend backend = new CpuBackend();
        int dimension = 7;
        float[] queryValues = {0.25f, -0.5f, 0.75f, 1.0f, -1.25f, 0.5f, 0.125f};
        float[] keyValues = {
                0.5f, 0.25f, -0.25f, 0.75f, 0.5f, -0.5f, 1.0f,
                -0.5f, 0.75f, 0.25f, -1.0f, 0.5f, 0.25f, -0.75f
        };
        float[] valueValues = {
                1f, 2f, 3f, 4f, 5f, 6f, 7f,
                -2f, 1f, 0.5f, 3f, -1f, 4f, 2f
        };
        Tensor query = TensorFactory.of(queryValues, backend, 1, 1, 1, dimension);
        Tensor key = TensorFactory.of(keyValues, backend, 1, 1, 2, dimension);
        Tensor value = TensorFactory.of(valueValues, backend, 1, 1, 2, dimension);
        Tensor output = null;
        try {
            output = backend.attention(query, key, value);
            double[] logits = new double[2];
            double max = Double.NEGATIVE_INFINITY;
            for (int keyIndex = 0; keyIndex < 2; keyIndex++) {
                double dot = 0d;
                for (int d = 0; d < dimension; d++) {
                    dot += (double) queryValues[d] * keyValues[keyIndex * dimension + d];
                }
                logits[keyIndex] = dot / Math.sqrt(dimension);
                max = Math.max(max, logits[keyIndex]);
            }
            double p0 = Math.exp(logits[0] - max);
            double p1 = Math.exp(logits[1] - max);
            double denominator = p0 + p1;
            float[] expected = new float[dimension];
            for (int d = 0; d < dimension; d++) {
                expected[d] = (float) ((p0 * valueValues[d] + p1 * valueValues[dimension + d])
                        / denominator);
            }
            assertArrayEquals(expected, output.toFloatArray(), 1e-5f);
        } finally {
            if (output != null) output.release();
            value.release();
            key.release();
            query.release();
        }
    }
}
