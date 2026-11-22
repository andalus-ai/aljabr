package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingGemma2AudioEncoderTest {
    private static EmbeddingGemma2AudioConfig config() {
        return new EmbeddingGemma2AudioConfig(
                "gemma4_audio", 2, 2, 1, 3, 1, List.of(2, 2),
                2, 3, 0, 1e-6, 0.5, "silu", "float32", false,
                16_000, 4, 8, 4, 2, 0.0, 8_000.0, 0.001,
                0.0, 1.0, 0.0, 280, 40, 1, 2, 3, 1.0e10, 50.0, -1.0e9);
    }

    private static EmbeddingGemma2AudioSubsampling.Weights subsamplingWeights() {
        var layer0 = new EmbeddingGemma2AudioSubsampling.ConvLayerWeights(
                new float[2][1][3][3], new float[]{1f, 1f});
        var layer1 = new EmbeddingGemma2AudioSubsampling.ConvLayerWeights(
                new float[2][2][3][3], new float[]{1f, 1f});
        // Two frequency bins after the two stride-2 convolutions, each with
        // two channels: 2 * 2 = 4 projection inputs.
        return new EmbeddingGemma2AudioSubsampling.Weights(
                layer0, layer1, new float[2][4]);
    }

    private static EmbeddingGemma2AudioFeedForward.Weights feedForwardWeights() {
        return new EmbeddingGemma2AudioFeedForward.Weights(
                new EmbeddingGemma2AudioFeedForward.LinearWeights(
                        new float[8][2], EmbeddingGemma2AudioFeedForward.ClipRange.unbounded()),
                new EmbeddingGemma2AudioFeedForward.LinearWeights(
                        new float[2][8], EmbeddingGemma2AudioFeedForward.ClipRange.unbounded()),
                new float[]{1f, 1f},
                new float[]{1f, 1f});
    }

    private static EmbeddingGemma2AudioAttention.Weights attentionWeights() {
        var zero = new float[2][2];
        var identity = new float[][]{{1f, 0f}, {0f, 1f}};
        return new EmbeddingGemma2AudioAttention.Weights(
                attentionLinear(zero), attentionLinear(zero),
                attentionLinear(identity), attentionLinear(identity),
                zero, new float[]{0f, 0f});
    }

    private static EmbeddingGemma2AudioAttention.LinearWeights attentionLinear(float[][] kernel) {
        return new EmbeddingGemma2AudioAttention.LinearWeights(
                kernel, EmbeddingGemma2AudioAttention.ClipRange.unbounded());
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

    private static EmbeddingGemma2AudioConformerLayer.Weights layerWeights() {
        return new EmbeddingGemma2AudioConformerLayer.Weights(
                feedForwardWeights(),
                new float[]{1f, 1f},
                attentionWeights(),
                new float[]{1f, 1f},
                lightConvWeights(),
                feedForwardWeights(),
                new float[]{1f, 1f});
    }

    private static EmbeddingGemma2AudioEncoder encoder() {
        return new EmbeddingGemma2AudioEncoder(
                config(),
                new EmbeddingGemma2AudioEncoder.Weights(
                        subsamplingWeights(),
                        List.of(layerWeights(), layerWeights()),
                        new float[3][2],
                        new float[]{0.25f, -0.5f, 1.0f}));
    }

    @Test
    void runsSubsamplingEveryConfiguredConformerLayerAndBiasedOutputProjection() {
        float[][][] features = new float[1][4][4];
        boolean[][] inputMask = {{true, false, true, false}};

        EmbeddingGemma2AudioEncoder.Output output = encoder().compute(features, inputMask);

        assertEquals(1, output.lastHiddenState().length);
        assertEquals(1, output.lastHiddenState()[0].length);
        assertEquals(3, output.lastHiddenState()[0][0].length);
        assertArrayEquals(new float[]{0.25f, -0.5f, 1.0f},
                output.lastHiddenState()[0][0], 1e-6f);
        assertArrayEquals(new boolean[]{true}, output.attentionMask()[0]);
    }

    @Test
    void supportsAbsentInputMaskAndPreservesItAsAbsent() {
        var output = encoder().compute(new float[1][4][4], null);
        assertNull(output.attentionMask());
        assertArrayEquals(new float[]{0.25f, -0.5f, 1.0f},
                output.lastHiddenState()[0][0], 1e-6f);
    }

    @Test
    void rejectsLayerCountThatDoesNotMatchAudioConfig() {
        var weights = new EmbeddingGemma2AudioEncoder.Weights(
                subsamplingWeights(), List.of(layerWeights()),
                new float[3][2], new float[3]);
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2AudioEncoder(config(), weights));
    }

    @Test
    void validatesOutputProjectionGeometryAndBias() {
        var badKernel = new EmbeddingGemma2AudioEncoder.Weights(
                subsamplingWeights(), List.of(layerWeights(), layerWeights()),
                new float[3][3], new float[3]);
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2AudioEncoder(config(), badKernel));

        var badBias = new EmbeddingGemma2AudioEncoder.Weights(
                subsamplingWeights(), List.of(layerWeights(), layerWeights()),
                new float[3][2], new float[2]);
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2AudioEncoder(config(), badBias));
    }

    @Test
    void outputArraysAreDefensiveCopies() {
        var output = encoder().compute(new float[1][4][4], null);
        float[][][] exposed = output.lastHiddenState();
        exposed[0][0][0] = 99f;
        assertEquals(0.25f, output.lastHiddenState()[0][0][0], 0f);
    }
}
