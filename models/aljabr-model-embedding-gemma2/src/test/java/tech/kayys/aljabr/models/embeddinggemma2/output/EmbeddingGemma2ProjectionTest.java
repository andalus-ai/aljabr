package tech.kayys.aljabr.models.embeddinggemma2.output;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class EmbeddingGemma2ProjectionTest {

    @Test
    void appliesCheckpointWeightsAndBias() {
        float[] input = new float[EmbeddingGemma2Projection.INPUT_DIMENSION];
        input[0] = 2f;
        input[1] = 3f;
        float[][] weights = new float[EmbeddingGemma2Projection.OUTPUT_DIMENSION]
                [EmbeddingGemma2Projection.INPUT_DIMENSION];
        float[] bias = new float[EmbeddingGemma2Projection.OUTPUT_DIMENSION];
        weights[0][0] = 4f;
        weights[0][1] = 5f;
        bias[0] = 7f;

        float[] result = EmbeddingGemma2Projection.project(input, weights, bias);
        assertEquals(30f, result[0]);
        assertEquals(0f, result[1]);
        assertEquals(EmbeddingGemma2Projection.OUTPUT_DIMENSION, result.length);
    }

    @Test
    void supportsBiasFreeCheckpointHead() {
        float[][] weights = new float[EmbeddingGemma2Projection.OUTPUT_DIMENSION]
                [EmbeddingGemma2Projection.INPUT_DIMENSION];
        weights[3][9] = 2f;
        float[] input = new float[EmbeddingGemma2Projection.INPUT_DIMENSION];
        input[9] = 6f;
        assertEquals(12f, EmbeddingGemma2Projection.project(input, weights, null)[3]);
    }

    @Test
    void rejectsMalformedAndNonFiniteCheckpointData() {
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2Projection.project(new float[4], new float[768][512], null));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2Projection.project(new float[512], new float[767][512], null));
        float[][] weights = new float[768][512];
        weights[10][11] = Float.NaN;
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2Projection.project(new float[512], weights, null));
    }

    @Test
    void doesNotMutateInputsOrWeights() {
        float[] input = new float[512];
        input[0] = 3f;
        float[][] weights = new float[768][512];
        weights[0][0] = 2f;
        EmbeddingGemma2Projection.project(input, weights, null);
        assertEquals(3f, input[0]);
        assertEquals(2f, weights[0][0]);
    }
}
