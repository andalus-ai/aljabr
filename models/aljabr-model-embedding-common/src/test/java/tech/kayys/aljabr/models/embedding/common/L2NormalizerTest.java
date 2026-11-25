package tech.kayys.aljabr.models.embedding.common;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class L2NormalizerTest {
    @Test
    void normalizesCompleteVectorWithoutMutatingInput() {
        float[] input = {3f, 4f};
        float[] result = L2Normalizer.normalize(input);
        assertArrayEquals(new float[] {0.6f, 0.8f}, result, 1.0e-6f);
        assertArrayEquals(new float[] {3f, 4f}, input);
    }

    @Test
    void truncatesBeforeNormalization() {
        float[] result = L2Normalizer.normalizePrefix(new float[] {3f, 4f, 100f}, 2);
        assertArrayEquals(new float[] {0.6f, 0.8f}, result, 1.0e-6f);
    }

    @Test
    void rejectsInvalidDimensionsNonFiniteAndZeroVectors() {
        assertThrows(IllegalArgumentException.class, () -> L2Normalizer.normalize(new float[0]));
        assertThrows(IllegalArgumentException.class,
                () -> L2Normalizer.normalizePrefix(new float[] {1f}, 2));
        assertThrows(IllegalArgumentException.class,
                () -> L2Normalizer.normalizePrefix(new float[] {1f}, 0));
        assertThrows(IllegalArgumentException.class,
                () -> L2Normalizer.normalize(new float[] {Float.NaN}));
        assertThrows(IllegalArgumentException.class,
                () -> L2Normalizer.normalize(new float[] {0f, 0f}));
    }
}
