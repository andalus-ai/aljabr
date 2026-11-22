package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingGemma2AudioAttentionTest {
    private static EmbeddingGemma2AudioConfig config() {
        return new EmbeddingGemma2AudioConfig(
                "gemma4_audio", 2, 1, 1, 8, 3, List.of(2, 2),
                2, 3, 0, 1e-6, 0.5, "silu", "float32", false,
                16_000, 4, 8, 4, 2, 0.0, 8_000.0, 0.001,
                0.0, 1.0, 0.0, 280, 40, 1, 2, 3, 1.0e10, 50.0, -1.0e9);
    }

    private static EmbeddingGemma2AudioAttention.LinearWeights linear(float[][] kernel) {
        return new EmbeddingGemma2AudioAttention.LinearWeights(
                kernel, EmbeddingGemma2AudioAttention.ClipRange.unbounded());
    }

    private static EmbeddingGemma2AudioAttention.Weights weights(boolean relativeProjection) {
        float[][] identity = {{1f, 0f}, {0f, 1f}};
        float[][] zero = new float[2][2];
        return new EmbeddingGemma2AudioAttention.Weights(
                linear(relativeProjection ? identity : zero),
                linear(zero),
                linear(identity),
                linear(identity),
                relativeProjection ? identity : zero,
                new float[]{0f, 0f});
    }

    @Test
    void computesChunkedLocalAttentionAcrossChunkBoundaries() {
        var attention = new EmbeddingGemma2AudioAttention(config(), weights(false));
        float[][][] input = {{{1f, 0f}, {0f, 1f}, {2f, 2f}}};

        var output = attention.compute(input, null);

        assertArrayEquals(new float[]{1f, 0f}, output.hiddenStates()[0][0], 1e-6f);
        assertArrayEquals(new float[]{0.5f, 0.5f}, output.hiddenStates()[0][1], 1e-6f);
        assertArrayEquals(new float[]{1f, 1.5f}, output.hiddenStates()[0][2], 1e-6f);
        assertEquals(0.5f, output.attentionWeights()[0][0][1][2], 1e-6f);
        assertEquals(0.5f, output.attentionWeights()[0][0][1][3], 1e-6f);
        assertEquals(0.5f, output.attentionWeights()[0][0][2][1], 1e-6f);
        assertEquals(0.5f, output.attentionWeights()[0][0][2][2], 1e-6f);
    }

    @Test
    void appliesValidMaskToQueriesAndKeys() {
        var attention = new EmbeddingGemma2AudioAttention(config(), weights(false));
        float[][][] input = {{{1f, 0f}, {100f, 100f}, {2f, 2f}}};
        boolean[][] mask = {{true, false, true}};

        var output = attention.compute(input, mask);

        assertArrayEquals(new float[]{1f, 0f}, output.hiddenStates()[0][0], 1e-6f);
        assertArrayEquals(new float[]{0f, 0f}, output.hiddenStates()[0][1], 1e-6f);
        assertArrayEquals(new float[]{2f, 2f}, output.hiddenStates()[0][2], 1e-6f);
        assertEquals(0f, output.attentionWeights()[0][0][2][1], 1e-7f);
        assertEquals(1f, output.attentionWeights()[0][0][2][2], 1e-6f);
    }

    @Test
    void addsRelativeKeyScoresBeforeSoftmax() {
        var attention = new EmbeddingGemma2AudioAttention(config(), weights(true));
        float[][][] input = {{{0f, 0f}, {1f, 0f}}};

        var output = attention.compute(input, null);
        float[][][][] probabilities = output.attentionWeights();

        // For query 1, context index 2 is key 0 (relative position +1),
        // while context index 3 is the current key (relative position 0).
        assertTrue(probabilities[0][0][1][2] > probabilities[0][0][1][3]);
        assertEquals(1f, probabilities[0][0][1][2] + probabilities[0][0][1][3], 1e-6f);
    }

    @Test
    void returnsZeroForQueriesWithNoValidKeys() {
        var attention = new EmbeddingGemma2AudioAttention(config(), weights(false));
        float[][][] input = {{{1f, 2f}}};
        boolean[][] mask = {{false}};

        var output = attention.compute(input, mask);

        assertArrayEquals(new float[]{0f, 0f}, output.hiddenStates()[0][0], 1e-7f);
        for (float probability : output.attentionWeights()[0][0][0]) {
            assertEquals(0f, probability, 1e-7f);
        }
    }

    @Test
    void rejectsInvalidMaskGeometry() {
        var attention = new EmbeddingGemma2AudioAttention(config(), weights(false));
        assertThrows(IllegalArgumentException.class,
                () -> attention.compute(new float[1][3][2], new boolean[1][2]));
    }

    @Test
    void rejectsMalformedProjectionWeights() {
        var base = weights(false);
        var malformed = new EmbeddingGemma2AudioAttention.Weights(
                linear(new float[1][2]), base.key(), base.value(), base.output(),
                base.relativeKeyProjection(), base.perDimensionScale());

        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2AudioAttention(config(), malformed));
    }

    @Test
    void doesNotMutateInputOrExposeCheckpointArrays() {
        var base = weights(false);
        var attention = new EmbeddingGemma2AudioAttention(config(), base);
        float[][][] input = {{{1f, 2f}, {3f, 4f}}};
        float original = input[0][0][0];
        float[][] exposed = base.relativeKeyProjection();
        exposed[0][0] = 100f;

        var output = attention.compute(input, null);
        float[][][] returned = output.hiddenStates();
        returned[0][0][0] = 500f;

        assertEquals(original, input[0][0][0]);
        assertEquals(0f, base.relativeKeyProjection()[0][0]);
        assertNotEquals(500f, output.hiddenStates()[0][0][0]);
    }
}
