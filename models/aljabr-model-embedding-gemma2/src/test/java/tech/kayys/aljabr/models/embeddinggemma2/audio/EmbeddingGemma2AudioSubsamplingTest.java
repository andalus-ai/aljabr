package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingGemma2AudioSubsamplingTest {
    private static EmbeddingGemma2AudioConfig config() {
        return new EmbeddingGemma2AudioConfig(
                "gemma4_audio", 3, 1, 1, 8, 5, List.of(2, 2),
                12, 13, 0, 1e-6, 0.5, "silu", "float32", false,
                16_000, 4, 8, 4, 2, 0.0, 8_000.0, 0.001,
                0.0, 1.0, 0.0, 280, 40, 1, 2, 3, 1.0e10);
    }

    private static EmbeddingGemma2AudioSubsampling.Weights zeroWeights() {
        var layer0 = new EmbeddingGemma2AudioSubsampling.ConvLayerWeights(
                new float[2][1][3][3], new float[]{1f, 1f});
        var layer1 = new EmbeddingGemma2AudioSubsampling.ConvLayerWeights(
                new float[2][2][3][3], new float[]{1f, 1f});
        return new EmbeddingGemma2AudioSubsampling.Weights(
                layer0, layer1, new float[3][2]);
    }

    @Test
    void downsamplesTimeAndFrequencyByFourAndProjectsToHiddenSize() {
        float[][][] features = new float[1][5][4];
        boolean[][] mask = {{true, false, true, false, true}};

        var output = new EmbeddingGemma2AudioSubsampling(config(), zeroWeights())
                .compute(features, mask);

        assertEquals(1, output.hiddenStates().length);
        assertEquals(2, output.hiddenStates()[0].length);
        assertEquals(3, output.hiddenStates()[0][0].length);
        assertEquals(1, output.frequencySize());
        assertArrayEquals(new boolean[]{true, true}, output.mask()[0]);
    }

    @Test
    void appliesInputMaskBeforeConvolutionAndReturnsDefensiveCopies() {
        float[][][] features = {{{1f, 1f, 1f, 1f}, {100f, 100f, 100f, 100f},
                {2f, 2f, 2f, 2f}, {200f, 200f, 200f, 200f}}};
        boolean[][] mask = {{true, false, true, false}};

        var output = new EmbeddingGemma2AudioSubsampling(config(), zeroWeights())
                .compute(features, mask);
        float[][][] exposed = output.hiddenStates();
        exposed[0][0][0] = 99f;
        assertEquals(0f, output.hiddenStates()[0][0][0]);
        boolean[][] exposedMask = output.mask();
        exposedMask[0][0] = false;
        assertTrue(output.mask()[0][0]);
    }

    @Test
    void rejectsMalformedCheckpointKernelShape() {
        var malformed = new EmbeddingGemma2AudioSubsampling.Weights(
                new EmbeddingGemma2AudioSubsampling.ConvLayerWeights(
                        new float[2][1][5][5], new float[]{1f, 1f}),
                zeroWeights().layer1(), new float[3][2]);
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2AudioSubsampling(config(), malformed));
    }

    @Test
    void rejectsMismatchedMaskShape() {
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2AudioSubsampling(config(), zeroWeights())
                        .compute(new float[1][4][4], new boolean[1][3]));
    }
}
