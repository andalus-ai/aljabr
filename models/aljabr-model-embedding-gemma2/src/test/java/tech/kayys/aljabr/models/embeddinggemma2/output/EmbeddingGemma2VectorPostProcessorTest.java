package tech.kayys.aljabr.models.embeddinggemma2.output;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmbeddingGemma2VectorPostProcessorTest {
    @Test
    void truncatesBeforeNormalizingAndDoesNotMutateInput() {
        float[] input = new float[768];
        input[0] = 3.0f;
        input[1] = 4.0f;
        input[512] = 100.0f;
        float[] original = input.clone();

        float[] result = EmbeddingGemma2VectorPostProcessor.process(input, 512);

        assertEquals(512, result.length);
        assertEquals(0.6f, result[0], 1.0e-6f);
        assertEquals(0.8f, result[1], 1.0e-6f);
        assertEquals(1.0d, norm(result), 1.0e-6d);
        assertArrayEquals(original, input);
    }

    @Test
    void supportsEveryDocumentedMatryoshkaDimension() {
        float[] input = new float[768];
        input[0] = 1.0f;
        for (int dimension : new int[]{768, 512, 256, 128}) {
            float[] result = EmbeddingGemma2VectorPostProcessor.process(input, dimension);
            assertEquals(dimension, result.length);
            assertEquals(1.0f, result[0], 1.0e-6f);
            assertEquals(1.0d, norm(result), 1.0e-6d);
            assertTrue(EmbeddingGemma2VectorPostProcessor.isSupportedDimension(dimension));
        }
        assertFalse(EmbeddingGemma2VectorPostProcessor.isSupportedDimension(384));
    }

    @Test
    void rejectsWrongNativeDimensionAndUnsupportedOutputDimension() {
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2VectorPostProcessor.process(new float[512], 512));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2VectorPostProcessor.process(new float[768], 384));
    }

    @Test
    void rejectsNonFiniteAndZeroVectors() {
        float[] nan = new float[768];
        nan[7] = Float.NaN;
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2VectorPostProcessor.process(nan, 768));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2VectorPostProcessor.process(new float[768], 768));
    }

    private static double norm(float[] values) {
        double sum = 0.0d;
        for (float value : values) sum += (double) value * value;
        return Math.sqrt(sum);
    }
}
