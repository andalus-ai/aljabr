package tech.kayys.aljabr.models.common.transformer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class RmsNormReferenceTest {
    @Test
    void normalizesEachHeadWithoutSubtractingMeanAndAppliesScale() {
        float[][][] input = {{{3f, 4f}}, {{0f, 2f}}};
        float[][][] output = RmsNormReference.apply(input, new float[] {2f, 0.5f}, 1e-6f);
        assertEquals(1.697056f, output[0][0][0], 1e-5f);
        assertEquals(0.565685f, output[0][0][1], 1e-5f);
        assertEquals(0f, output[1][0][0], 1e-6f);
        assertEquals(0.707106f, output[1][0][1], 1e-5f);
    }

    @Test
    void rejectsIncompatibleScaleAndInvalidEpsilon() {
        float[][][] input = {{{1f, 2f}}};
        assertThrows(IllegalArgumentException.class,
                () -> RmsNormReference.apply(input, new float[] {1f}, 1e-6f));
        assertThrows(IllegalArgumentException.class,
                () -> RmsNormReference.apply(input, null, 0f));
    }
}
