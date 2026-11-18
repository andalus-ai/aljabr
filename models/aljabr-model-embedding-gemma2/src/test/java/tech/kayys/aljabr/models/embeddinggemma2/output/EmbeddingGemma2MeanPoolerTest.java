package tech.kayys.aljabr.models.embeddinggemma2.output;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class EmbeddingGemma2MeanPoolerTest {

    @Test
    void poolsOnlyUnmaskedTokens() {
        float[][] states = {{1f, 3f}, {100f, 100f}, {3f, 7f}};
        float[] pooled = EmbeddingGemma2MeanPooler.meanPool(states, new boolean[] {true, false, true});
        assertArrayEquals(new float[] {2f, 5f}, pooled);
    }

    @Test
    void ignoresInvalidValuesInMaskedPadding() {
        float[][] states = {{2f, 4f}, {Float.NaN, Float.POSITIVE_INFINITY}};
        assertArrayEquals(new float[] {2f, 4f},
                EmbeddingGemma2MeanPooler.meanPool(states, new boolean[] {true, false}));
    }

    @Test
    void rejectsAllPaddingAndMismatchedShapes() {
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2MeanPooler.meanPool(new float[][] {{1f}}, new boolean[] {false}));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2MeanPooler.meanPool(new float[][] {{1f}, {2f}}, new boolean[] {true}));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2MeanPooler.meanPool(new float[][] {{1f}, {2f, 3f}},
                        new boolean[] {true, true}));
    }

    @Test
    void doesNotMutateTokenStates() {
        float[][] states = {{1f, 2f}, {3f, 4f}};
        float[][] before = {{1f, 2f}, {3f, 4f}};
        EmbeddingGemma2MeanPooler.meanPool(states, new boolean[] {true, true});
        assertArrayEquals(before[0], states[0]);
        assertArrayEquals(before[1], states[1]);
    }
}
