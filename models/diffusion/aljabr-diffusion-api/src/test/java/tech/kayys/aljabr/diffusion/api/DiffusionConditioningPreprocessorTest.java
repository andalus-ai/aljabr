package tech.kayys.aljabr.diffusion.api;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class DiffusionConditioningPreprocessorTest {

    @Test
    void selectsInputsByDeclaredModalityWithoutPreparingThem() {
        var preprocessor = new TestPreprocessor();
        var prompt = new DiffusionConditioningInput(
                "prompt", DiffusionConditioningInput.Modality.PROMPT, "text:hello");
        var image = new DiffusionConditioningInput(
                "image", DiffusionConditioningInput.Modality.IMAGE, "file:///image.png");

        assertTrue(preprocessor.supports(prompt));
        assertFalse(preprocessor.supports(image));
    }

    @Test
    void preparationReturnsOwnedDataWithInputProvenance() throws Exception {
        var preprocessor = new TestPreprocessor();
        var input = new DiffusionConditioningInput(
                "prompt", DiffusionConditioningInput.Modality.PROMPT, "text:hello");

        try (var prepared = preprocessor.prepare(input, DiffusionConditioningPreparationContext.defaults())) {
            assertEquals("prompt", prepared.inputId());
            assertEquals(DiffusionConditioningInput.Modality.PROMPT, prepared.modality());
        }
        assertTrue(preprocessor.lastData.closed);
    }

    @Test
    void contextDefensivelyCopiesOptions() {
        var options = new java.util.HashMap<String, String>();
        options.put("prompt.max-length", "128");
        var context = new DiffusionConditioningPreparationContext(options);
        options.put("prompt.max-length", "999");

        assertEquals("128", context.options().get("prompt.max-length"));
        assertThrows(UnsupportedOperationException.class,
                () -> context.options().put("other", "value"));
    }

    private static final class TestPreprocessor implements DiffusionConditioningPreprocessor {
        private TestData lastData;

        @Override
        public Set<DiffusionConditioningInput.Modality> modalities() {
            return Set.of(DiffusionConditioningInput.Modality.PROMPT);
        }

        @Override
        public DiffusionConditioningData prepare(
                DiffusionConditioningInput input,
                DiffusionConditioningPreparationContext context) {
            if (!supports(input)) {
                throw new DiffusionConditioningPreparationException(input, "Unsupported input");
            }
            lastData = new TestData(input.id(), input.modality());
            return lastData;
        }
    }

    private static final class TestData implements DiffusionConditioningData {
        private final String inputId;
        private final DiffusionConditioningInput.Modality modality;
        private boolean closed;

        private TestData(String inputId, DiffusionConditioningInput.Modality modality) {
            this.inputId = inputId;
            this.modality = modality;
        }

        @Override
        public String inputId() {
            return inputId;
        }

        @Override
        public DiffusionConditioningInput.Modality modality() {
            return modality;
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
