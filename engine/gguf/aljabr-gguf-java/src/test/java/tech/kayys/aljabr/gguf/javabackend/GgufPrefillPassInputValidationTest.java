package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufPrefillPassInputValidationTest {

    @Test
    void acceptsNonEmptyPromptWithinVocabularyAndContext() {
        assertDoesNotThrow(() ->
                GgufPrefillPass.validateTokenIds(new int[] {0, 4, 9}, 10, 8));
    }

    @Test
    void rejectsNullAndEmptyPrompts() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufPrefillPass.validateTokenIds(null, 10, 8));
        assertThrows(IllegalArgumentException.class,
                () -> GgufPrefillPass.validateTokenIds(new int[0], 10, 8));
    }

    @Test
    void rejectsPromptLongerThanModelContext() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufPrefillPass.validateTokenIds(new int[] {1, 2, 3}, 10, 2));
    }

    @Test
    void rejectsNegativeAndOutOfVocabularyTokenIds() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufPrefillPass.validateTokenIds(new int[] {1, -1}, 10, 8));
        assertThrows(IllegalArgumentException.class,
                () -> GgufPrefillPass.validateTokenIds(new int[] {1, 10}, 10, 8));
    }
}
