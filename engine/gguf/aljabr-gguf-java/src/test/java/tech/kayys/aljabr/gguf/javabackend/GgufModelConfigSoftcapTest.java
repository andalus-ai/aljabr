package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufModelConfigSoftcapTest {

    @Test
    void zeroDisablesFinalLogitSoftcap() {
        assertEquals(0.0f, GgufModelConfig.validateFinalLogitSoftcap(0.0f));
    }

    @Test
    void acceptsFinitePositiveFinalLogitSoftcap() {
        assertEquals(30.0f, GgufModelConfig.validateFinalLogitSoftcap(30.0f));
    }

    @Test
    void rejectsNegativeFinalLogitSoftcap() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufModelConfig.validateFinalLogitSoftcap(-1.0f));
    }

    @Test
    void rejectsNanFinalLogitSoftcap() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufModelConfig.validateFinalLogitSoftcap(Float.NaN));
    }

    @Test
    void rejectsInfiniteFinalLogitSoftcap() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufModelConfig.validateFinalLogitSoftcap(Float.POSITIVE_INFINITY));
        assertThrows(IllegalArgumentException.class,
                () -> GgufModelConfig.validateFinalLogitSoftcap(Float.NEGATIVE_INFINITY));
    }
}
