package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingGemma2AudioFeedForwardTest {
    private static EmbeddingGemma2AudioConfig config() {
        return new EmbeddingGemma2AudioConfig(
                "gemma4_audio", 8, 1, 1, 8, 5, List.of(2, 2),
                12, 13, 0, 1e-6, 0.5, "silu", "float32", true,
                16_000, 4, 8, 4, 2, 0.0, 8_000.0, 0.001,
                0.0, 1.0, 0.0, 280, 40, 1, 2, 3, 1.0e10);
    }

    private static EmbeddingGemma2AudioFeedForward.Weights weights() {
        float[][] expand = new float[32][8];
        float[][] contract = new float[8][32];
        for (int i = 0; i < 8; i++) {
            expand[i][i] = 1f;
            contract[i][i] = 1f;
        }
        return new EmbeddingGemma2AudioFeedForward.Weights(
                new EmbeddingGemma2AudioFeedForward.LinearWeights(
                        expand, EmbeddingGemma2AudioFeedForward.ClipRange.unbounded()),
                new EmbeddingGemma2AudioFeedForward.LinearWeights(
                        contract, EmbeddingGemma2AudioFeedForward.ClipRange.unbounded()),
                new float[]{1,1,1,1,1,1,1,1},
                new float[]{1,1,1,1,1,1,1,1});
    }

    @Test
    void preservesShapeAndAddsScaledResidual() {
        float[][][] input = {{{1,2,3,4,5,6,7,8}, {-1,-2,-3,-4,-5,-6,-7,-8}}};
        var output = new EmbeddingGemma2AudioFeedForward(config(), weights()).compute(input);
        assertEquals(1, output.length);
        assertEquals(2, output[0].length);
        assertEquals(8, output[0][0].length);
        assertNotEquals(input[0][0][0], output[0][0][0]);
    }

    @Test
    void appliesPerLinearCheckpointClippingBounds() {
        var base = weights();
        var boundedExpand = new EmbeddingGemma2AudioFeedForward.LinearWeights(
                base.expand().kernel(), new EmbeddingGemma2AudioFeedForward.ClipRange(0, 0, 0, 0));
        var bounded = new EmbeddingGemma2AudioFeedForward.Weights(
                boundedExpand, base.contract(), base.preNormScale(), base.postNormScale());
        float[][][] output = new EmbeddingGemma2AudioFeedForward(config(), bounded)
                .compute(new float[][][] {{{1,1,1,1,1,1,1,1}}});
        assertArrayEquals(new float[]{1,1,1,1,1,1,1,1}, output[0][0], 1e-6f);
    }

    @Test
    void rejectsIncorrectCheckpointDimensions() {
        var malformed = new EmbeddingGemma2AudioFeedForward.Weights(
                new EmbeddingGemma2AudioFeedForward.LinearWeights(
                        new float[31][8], EmbeddingGemma2AudioFeedForward.ClipRange.unbounded()),
                weights().contract(), weights().preNormScale(), weights().postNormScale());
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2AudioFeedForward(config(), malformed));
    }

    @Test
    void doesNotMutateCallerInput() {
        float[][][] input = {{{1,2,3,4,5,6,7,8}}};
        float original = input[0][0][0];
        new EmbeddingGemma2AudioFeedForward(config(), weights()).compute(input);
        assertEquals(original, input[0][0][0]);
    }
}
