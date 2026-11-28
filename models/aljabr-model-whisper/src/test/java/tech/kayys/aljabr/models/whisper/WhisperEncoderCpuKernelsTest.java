package tech.kayys.aljabr.models.whisper;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class WhisperEncoderCpuKernelsTest {

    @Test
    void conv1dUsesChannelFirstLayoutStrideAndPadding() {
        float[][][] input = {{{1, 2, 3, 4}}};
        float[][][] weights = {{{1, 1, 1}}};

        float[][][] output = WhisperEncoderCpuKernels.conv1d(
                input, weights, new float[] {0}, 2, 1);

        assertEquals(2, output[0][0].length);
        assertEquals(3.0f, output[0][0][0], 1e-6f);
        assertEquals(9.0f, output[0][0][1], 1e-6f);
    }

    @Test
    void linearProjectsTheLastDimensionWithoutMutatingInput() {
        float[][][] input = {{{1, 2}, {3, 4}}};
        float[][] outputWeights = {{2, 0}, {0, 3}};
        float[][][] output = WhisperEncoderCpuKernels.linear(
                input, outputWeights, new float[] {1, -1});

        assertEquals(3.0f, output[0][0][0], 1e-6f);
        assertEquals(5.0f, output[0][0][1], 1e-6f);
        assertEquals(7.0f, output[0][1][0], 1e-6f);
        assertEquals(11.0f, output[0][1][1], 1e-6f);
        assertEquals(1.0f, input[0][0][0], 1e-6f);
    }

    @Test
    void layerNormNormalizesEachTokenIndependently() {
        float[][][] output = WhisperEncoderCpuKernels.layerNorm(
                new float[][][] {{{1, 3}, {2, 2}}},
                new float[] {1, 1}, new float[] {0, 0}, 1e-5f);

        assertEquals(-1.0f, output[0][0][0], 1e-3f);
        assertEquals(1.0f, output[0][0][1], 1e-3f);
        assertEquals(0.0f, output[0][1][0], 1e-6f);
        assertEquals(0.0f, output[0][1][1], 1e-6f);
    }

    @Test
    void geluMatchesKnownReferenceValues() {
        float[][][] output = WhisperEncoderCpuKernels.gelu(
                new float[][][] {{{0.0f, 1.0f, -1.0f}}});

        assertEquals(0.0f, output[0][0][0], 1e-6f);
        assertEquals(0.8413447f, output[0][0][1], 2e-5f);
        assertEquals(-0.1586553f, output[0][0][2], 2e-5f);
    }

    @Test
    void rejectsInvalidConvolutionStride() {
        assertThrows(IllegalArgumentException.class, () ->
                WhisperEncoderCpuKernels.conv1d(
                        new float[][][] {{{1, 2, 3}}},
                        new float[][][] {{{1}}},
                        new float[] {0}, 0, 0));
    }
}
