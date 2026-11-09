package tech.kayys.aljabr.backend.cpu;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class FlashAttentionCpuTest {

    @Test
    void rankTwoAttentionComputesOutputAcrossQueryBlocks() {
        CpuBackend backend = new CpuBackend();
        Tensor query = TensorFactory.of(new float[130], backend, 65, 2);
        Tensor key = TensorFactory.of(new float[130], backend, 65, 2);
        float[] values = new float[130];
        for (int i = 0; i < values.length; i++) {
            values[i] = i;
        }
        Tensor value = TensorFactory.of(values, backend, 65, 2);
        Tensor output = null;
        try {
            output = backend.attention(query, key, value);
            float[] expected = new float[130];
            double sum0 = 0d;
            double sum1 = 0d;
            for (int i = 0; i < 65; i++) {
                sum0 += values[i * 2];
                sum1 += values[i * 2 + 1];
            }
            for (int i = 0; i < 65; i++) {
                expected[i * 2] = (float) (sum0 / 65d);
                expected[i * 2 + 1] = (float) (sum1 / 65d);
            }
            assertArrayEquals(expected, output.toFloatArray(), 1e-4f);
        } finally {
            if (output != null) output.release();
            value.release();
            key.release();
            query.release();
            backend.close();
        }
    }
}
