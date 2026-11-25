package tech.kayys.aljabr.models.embedding.common;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MaskedMeanPoolerTest {
    @Test
    void poolsOnlyUnmaskedTokens() {
        float[][] states = {{1f, 3f}, {100f, 100f}, {3f, 7f}};
        assertArrayEquals(new float[] {2f, 5f},
                MaskedMeanPooler.meanPool(states, new boolean[] {true, false, true}));
    }

    @Test
    void ignoresMaskedPaddingValuesAndDoesNotMutateInputs() {
        float[][] states = {{2f, 4f}, {Float.NaN, Float.POSITIVE_INFINITY}};
        float[][] original = {{2f, 4f}, {Float.NaN, Float.POSITIVE_INFINITY}};
        assertArrayEquals(new float[] {2f, 4f},
                MaskedMeanPooler.meanPool(states, new boolean[] {true, false}));
        assertArrayEquals(original[0], states[0]);
        assertArrayEquals(original[1], states[1]);
    }

    @Test
    void rejectsEmptyAllMaskedAndInconsistentUnmaskedStates() {
        assertThrows(IllegalArgumentException.class,
                () -> MaskedMeanPooler.meanPool(new float[0][], new boolean[0]));
        assertThrows(IllegalArgumentException.class,
                () -> MaskedMeanPooler.meanPool(new float[][] {{1f}}, new boolean[] {false}));
        assertThrows(IllegalArgumentException.class,
                () -> MaskedMeanPooler.meanPool(new float[][] {{1f}, {2f, 3f}},
                        new boolean[] {true, true}));
    }

    @Test
    void rejectsNonFiniteUnmaskedStates() {
        assertThrows(IllegalArgumentException.class,
                () -> MaskedMeanPooler.meanPool(new float[][] {{Float.NaN}}, new boolean[] {true}));
    }
}
