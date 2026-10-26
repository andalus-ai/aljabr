package tech.kayys.aljabr.gguf.loader.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GGUFVectorOpsRopeTest {

    @Test
    void legacyRopeOverloadMatchesExplicitFullHeadRotation() {
        float[] legacy = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        float[] explicit = legacy.clone();

        GGUFVectorOps.rope(legacy, 2, 8, 3, 10_000.0f, null);
        GGUFVectorOps.rope(explicit, 2, 8, 8, 3, 10_000.0f, null);

        assertArrayEquals(explicit, legacy, 0.0f);
    }

    @Test
    void partialRopeLeavesDimensionsBeyondRotaryWindowUnchanged() {
        float[] values = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] original = values.clone();

        GGUFVectorOps.rope(values, 1, 8, 4, 1, 10_000.0f, null);

        float cos0 = (float) Math.cos(1.0);
        float sin0 = (float) Math.sin(1.0);
        float theta1 = (float) Math.pow(10_000.0, -2.0 / 4.0);
        float cos1 = (float) Math.cos(theta1);
        float sin1 = (float) Math.sin(theta1);

        assertEquals(original[0] * cos0 - original[2] * sin0, values[0], 1e-6f);
        assertEquals(original[0] * sin0 + original[2] * cos0, values[2], 1e-6f);
        assertEquals(original[1] * cos1 - original[3] * sin1, values[1], 1e-6f);
        assertEquals(original[1] * sin1 + original[3] * cos1, values[3], 1e-6f);
        for (int i = 4; i < original.length; i++) {
            assertEquals(original[i], values[i], 0.0f, "non-rotary dimension " + i + " must be unchanged");
        }
    }

    @Test
    void rejectsOddRotaryDimension() {
        assertThrows(IllegalArgumentException.class,
                () -> GGUFVectorOps.rope(new float[8], 1, 8, 5, 1, 10_000.0f, null));
    }
}
