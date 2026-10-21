package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Generic GGUF runtime test helpers shared across quantization families.
 */
final class GgufFx {
    private GgufFx() {
    }

    static void restoreProperty(String name, String value) {
        if (value == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, value);
        }
        if (name.startsWith("aljabr.gguf.parallel_")) {
            GgufParallelConfig.resetParallelConfig();
        }
    }

    static void assertSingleRowMatVecMatchesDot(
            GGUFModel model,
            GGUFTensorInfo tensor,
            float[] vector) {
        float[] output = new float[1];

        GgufTensorOps.matVecRows(model, tensor, vector, output, 1, true);

        assertEquals(GgufTensorOps.dotRow(model, tensor, 0, vector), output[0], 0.0f);
    }

    static float[] dequantizedRow(GGUFModel model, GGUFTensorInfo tensor) {
        float[] row = new float[(int) tensor.shape()[0]];
        GgufTensorOps.dequantizeRow(model, tensor, 0, row);
        return row;
    }

    static float[] ones(int length) {
        float[] values = new float[length];
        for (int i = 0; i < values.length; i++) {
            values[i] = 1.0f;
        }
        return values;
    }

    static float[] ramp(int length) {
        float[] values = new float[length];
        for (int i = 0; i < values.length; i++) {
            values[i] = i + 1.0f;
        }
        return values;
    }

    static float sum(float[] values) {
        float total = 0.0f;
        for (float value : values) {
            total += value;
        }
        return total;
    }

    static float dot(float[] left, float[] right) {
        float total = 0.0f;
        for (int index = 0; index < left.length; index++) {
            total += left[index] * right[index];
        }
        return total;
    }
}
