package tech.kayys.aljabr.models.common.transformer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class RotaryPositionEmbeddingReferenceTest {
    @Test
    void leavesPositionZeroUnchangedAndRotatesAtNonZeroPositions() {
        float[][][] input = {{{1f, 0f, 0f, 0f}}};
        float[][][] zero = RotaryPositionEmbeddingReference.apply(input, new int[] {0}, 10000d, null);
        assertEquals(1f, zero[0][0][0], 1e-6f);
        assertEquals(0f, zero[0][0][2], 1e-6f);

        float[][][] rotated = RotaryPositionEmbeddingReference.apply(input, new int[] {1}, 10000d, null);
        assertEquals(Math.cos(1d), rotated[0][0][0], 1e-6);
        assertEquals(Math.sin(1d), rotated[0][0][2], 1e-6);
    }

    @Test
    void preservesUnrotatedTailForPartialRope() {
        float[][][] input = {{{1f, 0f, 7f}}};
        float[][][] output = RotaryPositionEmbeddingReference.apply(input, new int[] {1}, 10000d, 2d / 3d);
        assertEquals(7f, output[0][0][2], 0f);
    }

    @Test
    void rejectsInvalidPositionGeometry() {
        float[][][] input = {{{1f, 2f}}};
        assertThrows(IllegalArgumentException.class,
                () -> RotaryPositionEmbeddingReference.apply(input, new int[] {0, 1}, 10000d, null));
        assertThrows(IllegalArgumentException.class,
                () -> RotaryPositionEmbeddingReference.apply(input, new int[] {0}, 0d, null));
    }
}
