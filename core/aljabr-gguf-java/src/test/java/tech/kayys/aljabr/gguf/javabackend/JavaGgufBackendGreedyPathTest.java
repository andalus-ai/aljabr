package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JavaGgufBackendGreedyPathTest {

    @Test
    void usesMetalGreedyWhenPenaltyIsDisabled() {
        assertTrue(JavaGgufBackend.canUseMetalGreedy(true, 1.0f, true));
    }

    @Test
    void keepsRepetitionPenaltyOnRegularSamplerPath() {
        assertFalse(JavaGgufBackend.canUseMetalGreedy(true, 1.1f, true));
    }

    @Test
    void doesNotEnableMetalGreedyForNonGreedySampling() {
        assertFalse(JavaGgufBackend.canUseMetalGreedy(false, 1.0f, true));
    }

    @Test
    void doesNotEnableMetalGreedyWhenNativeDecodeIsUnavailable() {
        assertFalse(JavaGgufBackend.canUseMetalGreedy(true, 1.0f, false));
    }
}
