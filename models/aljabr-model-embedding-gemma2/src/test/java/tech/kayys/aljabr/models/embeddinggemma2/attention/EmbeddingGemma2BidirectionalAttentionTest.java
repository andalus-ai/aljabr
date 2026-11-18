package tech.kayys.aljabr.models.embeddinggemma2.attention;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class EmbeddingGemma2BidirectionalAttentionTest {

    @Test
    void attendsToTokensOnBothSidesWithoutCausalMask() {
        float[][][] query = {{{1f}}, {{1f}}};
        float[][][] key = {{{0f}}, {{0f}}};
        float[][][] value = {{{2f}}, {{6f}}};

        var result = EmbeddingGemma2BidirectionalAttention.compute(
                query, key, value, new int[]{0, 1}, null, null, 1f);

        assertEquals(4f, result.output()[0][0][0], 1e-6f);
        assertEquals(4f, result.output()[1][0][0], 1e-6f);
        assertEquals(0.5f, result.probabilities()[0][0][1], 1e-6f);
    }

    @Test
    void mapsQueryHeadsToGroupedKeyValueHeads() {
        float[][][] query = {{{0f}, {0f}, {0f}, {0f}}};
        float[][][] key = {{{0f}, {0f}}};
        float[][][] value = {{{3f}, {9f}}};

        var result = EmbeddingGemma2BidirectionalAttention.compute(
                query, key, value, new int[]{0}, null, null, 1f);

        assertEquals(3f, result.output()[0][0][0]);
        assertEquals(3f, result.output()[0][1][0]);
        assertEquals(9f, result.output()[0][2][0]);
        assertEquals(9f, result.output()[0][3][0]);
    }

    @Test
    void appliesSymmetricInclusiveSlidingWindow() {
        float[][][] query = {{{0f}}, {{0f}}, {{0f}}};
        float[][][] key = {{{0f}}, {{0f}}, {{0f}}};
        float[][][] value = {{{2f}}, {{4f}}, {{8f}}};

        var result = EmbeddingGemma2BidirectionalAttention.compute(
                query, key, value, new int[]{10, 11, 12}, null, 1, 1f);

        assertEquals(3f, result.output()[0][0][0], 1e-6f);
        assertEquals(14f / 3f, result.output()[1][0][0], 1e-6f);
        assertEquals(6f, result.output()[2][0][0], 1e-6f);
        assertEquals(0f, result.probabilities()[0][0][2]);
        assertEquals(0f, result.probabilities()[0][2][0]);
    }

    @Test
    void excludesPaddingKeysButDoesNotApplyCausalMask() {
        float[][][] query = {{{0f}}, {{0f}}};
        float[][][] key = {{{0f}}, {{0f}}};
        float[][][] value = {{{2f}}, {{10f}}};

        var result = EmbeddingGemma2BidirectionalAttention.compute(
                query, key, value, new int[]{0, 1}, new boolean[]{true, false}, null, 1f);

        assertEquals(2f, result.output()[0][0][0]);
        assertEquals(2f, result.output()[1][0][0]);
        assertEquals(0f, result.probabilities()[0][0][1]);
    }

    @Test
    void usesStableSoftmaxForLargeScores() {
        float[][][] query = {{{1000f}}};
        float[][][] key = {{{1f}}, {{0f}}};
        float[][][] value = {{{7f}}, {{1f}}};

        var result = EmbeddingGemma2BidirectionalAttention.compute(
                query, key, value, new int[]{0, 1}, null, null, 1f);

        assertEquals(7f, result.output()[0][0][0], 1e-6f);
        assertTrue(Float.isFinite(result.probabilities()[0][0][0]));
    }

    @Test
    void rejectsIncompatibleHeadsAndQueriesWithNoVisibleKeys() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2BidirectionalAttention.compute(
                        new float[][][]{{{1f}, {1f}, {1f}}},
                        new float[][][]{{{1f}, {1f}}},
                        new float[][][]{{{1f}, {1f}}},
                        new int[]{0}, null, null, 1f));

        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2BidirectionalAttention.compute(
                        new float[][][]{{{1f}}},
                        new float[][][]{{{1f}}},
                        new float[][][]{{{1f}}},
                        new int[]{0}, new boolean[]{false}, null, 1f));
    }
}
