package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingGemma2AudioConformerLayerTest {
    private static EmbeddingGemma2AudioConfig config() {
        return new EmbeddingGemma2AudioConfig(
                "gemma4_audio", 2, 1, 1, 8, 1, List.of(2, 2),
                2, 3, 0, 1e-6, 0.5, "silu", "float32", false,
                16_000, 4, 8, 4, 2, 0.0, 8_000.0, 0.001,
                0.0, 1.0, 0.0, 280, 40, 1, 2, 3, 1.0e10, 50.0, -1.0e9);
    }

    private static EmbeddingGemma2AudioFeedForward.Weights feedForwardWeights() {
        return new EmbeddingGemma2AudioFeedForward.Weights(
                ffLinear(new float[8][2]),
                ffLinear(new float[2][8]),
                new float[]{1f, 1f},
                new float[]{1f, 1f});
    }

    private static EmbeddingGemma2AudioFeedForward.LinearWeights ffLinear(float[][] kernel) {
        return new EmbeddingGemma2AudioFeedForward.LinearWeights(
                kernel, EmbeddingGemma2AudioFeedForward.ClipRange.unbounded());
    }

    private static EmbeddingGemma2AudioAttention.LinearWeights attentionLinear(float[][] kernel) {
        return new EmbeddingGemma2AudioAttention.LinearWeights(
                kernel, EmbeddingGemma2AudioAttention.ClipRange.unbounded());
    }

    private static EmbeddingGemma2AudioAttention.Weights attentionWeights() {
        float[][] identity = {{1f, 0f}, {0f, 1f}};
        float[][] zero = new float[2][2];
        return new EmbeddingGemma2AudioAttention.Weights(
                attentionLinear(zero),
                attentionLinear(zero),
                attentionLinear(identity),
                attentionLinear(identity),
                zero,
                new float[]{0f, 0f});
    }

    private static EmbeddingGemma2AudioLightConv1d.Weights lightConvWeights() {
        return new EmbeddingGemma2AudioLightConv1d.Weights(
                new EmbeddingGemma2AudioLightConv1d.LinearWeights(
                        new float[4][2], EmbeddingGemma2AudioLightConv1d.ClipRange.unbounded()),
                new EmbeddingGemma2AudioLightConv1d.LinearWeights(
                        new float[2][2], EmbeddingGemma2AudioLightConv1d.ClipRange.unbounded()),
                new float[2][1],
                new float[]{1f, 1f},
                new float[]{1f, 1f});
    }

    private static EmbeddingGemma2AudioConformerLayer layer() {
        return new EmbeddingGemma2AudioConformerLayer(
                config(),
                new EmbeddingGemma2AudioConformerLayer.Weights(
                        feedForwardWeights(),
                        new float[]{1f, 1f},
                        attentionWeights(),
                        new float[]{1f, 1f},
                        lightConvWeights(),
                        feedForwardWeights(),
                        new float[]{1f, 1f}));
    }

    @Test
    void composesFeedForwardAttentionResidualConvolutionAndFinalNormInReferenceOrder() {
        float[][][] input = {{{1f, 0f}, {0f, 2f}}};

        float[][][] output = layer().compute(input, null);

        assertEquals(1, output.length);
        assertEquals(2, output[0].length);
        assertArrayEquals(new float[]{(float) Math.sqrt(2.0), 0f}, output[0][0], 1e-5f);

        // FFN-1 and the zero-weight light-convolution/FFN-2 fixtures are
        // residual identities. Attention averages the two normalized frames
        // for the second query; the layer then adds its saved residual and
        // applies the final RMSNorm.
        float secondResidual = 3f;
        float inverseRms = (float) (1.0 / Math.sqrt((1f + secondResidual * secondResidual) / 2.0 + 1e-6));
        assertArrayEquals(new float[]{inverseRms, secondResidual * inverseRms}, output[0][1], 1e-5f);
        assertArrayEquals(new float[]{1f, 0f}, input[0][0], 0f);
        assertArrayEquals(new float[]{0f, 2f}, input[0][1], 0f);
    }

    @Test
    void forwardsFrameMaskToAttentionWithoutMutatingInput() {
        float[][][] input = {{{1f, 0f}, {0f, 2f}}};
        boolean[][] mask = {{true, false}};

        float[][][] output = layer().compute(input, mask);

        assertArrayEquals(new float[]{(float) Math.sqrt(2.0), 0f}, output[0][0], 1e-5f);
        assertArrayEquals(new float[]{0f, (float) Math.sqrt(2.0)}, output[0][1], 1e-5f);
        assertArrayEquals(new float[]{0f, 2f}, input[0][1], 0f);
    }

    @Test
    void rejectsIncorrectLayerNormScaleGeometry() {
        var weights = new EmbeddingGemma2AudioConformerLayer.Weights(
                feedForwardWeights(),
                new float[]{1f},
                attentionWeights(),
                new float[]{1f, 1f},
                lightConvWeights(),
                feedForwardWeights(),
                new float[]{1f, 1f});

        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2AudioConformerLayer(config(), weights));
    }

    @Test
    void rejectsMaskWithDifferentBatchOrSequenceGeometry() {
        assertThrows(IllegalArgumentException.class,
                () -> layer().compute(new float[][][] {{{1f, 0f}, {0f, 1f}}}, new boolean[2][2]));
        assertThrows(IllegalArgumentException.class,
                () -> layer().compute(new float[][][] {{{1f, 0f}, {0f, 1f}}}, new boolean[1][1]));
    }
}
