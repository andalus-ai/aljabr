package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.api.GgufBackendSelection;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufBackendSelectionTest {

    @Test
    void defaultsToAutoWhenNoBackendRequested() {
        GgufBackendSelection selection = GgufBackendSelection.resolve(Map.of());

        assertEquals("auto", selection.normalizedValue());
        assertFalse(selection.explicit());
    }

    @Test
    void allowsExplicitJavaBackendByAlias() {
        GgufBackendSelection selection = GgufBackendSelection.resolve(Map.of("gguf.backend", "java-native"));

        assertEquals("java-native", selection.normalizedValue());
        assertTrue(selection.explicit());
    }

    @Test
    void allowsExplicitLlamaCppBackendByAlias() {
        GgufBackendSelection selection = GgufBackendSelection.resolve(Map.of("gguf.backend", "llama.cpp"));

        assertEquals("llama.cpp", selection.normalizedValue());
        assertTrue(selection.explicit());
    }

    @Test
    void unknownBackendTokenBecomesExplicitRequest() {
        GgufBackendSelection selection = GgufBackendSelection.resolve(Map.of("gguf.backend", "fastest-please"));

        assertEquals("fastest-please", selection.normalizedValue());
        assertEquals("fastest-please", selection.requestedValue());
        assertTrue(selection.explicit());
    }

    @Test
    void autoValueNormalizesToAutoToken() {
        GgufBackendSelection selection = GgufBackendSelection.resolve(Map.of("gguf.backend", "auto"));

        assertEquals("auto", selection.normalizedValue());
        assertFalse(selection.explicit());
    }
}
