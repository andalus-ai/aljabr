package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufLogitsProcessorSoftcapTest {

    @Test
    void zeroCapLeavesLogitsUnchanged() {
        float[] logits = {-100.0f, -1.0f, 0.0f, 1.0f, 100.0f};
        float[] expected = logits.clone();

        GgufLogitsProcessor.applySoftcap(logits, 0.0f);

        assertArrayEquals(expected, logits);
    }

    @Test
    void positiveCapAppliesScaledHyperbolicTangent() {
        float[] logits = {-30.0f, 0.0f, 30.0f};

        GgufLogitsProcessor.applySoftcap(logits, 30.0f);

        assertEquals(-30.0f * (float) Math.tanh(1.0), logits[0], 1.0e-6f);
        assertEquals(0.0f, logits[1]);
        assertEquals(30.0f * (float) Math.tanh(1.0), logits[2], 1.0e-6f);
    }

    @Test
    void softcapBoundsLargePositiveAndNegativeLogits() {
        float[] logits = {-10000.0f, 10000.0f};

        GgufLogitsProcessor.applySoftcap(logits, 12.0f);

        assertEquals(-12.0f, logits[0], 1.0e-5f);
        assertEquals(12.0f, logits[1], 1.0e-5f);
    }

    @Test
    void rejectsNegativeAndNonFiniteCaps() {
        float[] logits = {1.0f};
        assertThrows(IllegalArgumentException.class,
                () -> GgufLogitsProcessor.applySoftcap(logits, -1.0f));
        assertThrows(IllegalArgumentException.class,
                () -> GgufLogitsProcessor.applySoftcap(logits, Float.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> GgufLogitsProcessor.applySoftcap(logits, Float.POSITIVE_INFINITY));
    }

    @Test
    void rejectsNullLogits() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufLogitsProcessor.applySoftcap(null, 1.0f));
    }
}
