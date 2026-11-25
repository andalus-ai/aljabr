package tech.kayys.aljabr.diffusion.api;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class DiffusionConditioningPreparationOrchestratorTest {

    private static final var PROMPT = DiffusionConditioningInput.Modality.PROMPT;
    private static final var IMAGE = DiffusionConditioningInput.Modality.IMAGE;

    @Test
    void preparesInInputOrderAndClosesInReverseOrder() throws Exception {
        var events = new ArrayList<String>();
        var prompt = input("prompt", PROMPT);
        var image = input("image", IMAGE);
        var orchestrator = orchestrator(
                new RecordingPreprocessor("prompt-adapter", PROMPT, events),
                new RecordingPreprocessor("image-adapter", IMAGE, events));

        var result = orchestrator.prepare(
                new DiffusionConditioningSet(List.of(prompt, image)),
                List.of(requirement(PROMPT, 1), requirement(IMAGE, 1)),
                DiffusionConditioningPreparationContext.defaults());

        assertEquals(List.of("prepare:prompt", "prepare:image"), events);
        assertEquals(List.of("prompt", "image"),
                result.data().stream().map(DiffusionConditioningData::inputId).toList());
        result.close();
        result.close();
        assertEquals(List.of("prepare:prompt", "prepare:image", "close:image", "close:prompt"), events);
        assertTrue(result.isClosed());
    }

    @Test
    void rejectsUnsupportedAndAmbiguousInputs() {
        var prompt = input("prompt", PROMPT);
        var unsupported = new DiffusionConditioningPreparationOrchestrator(
                new DiffusionConditioningPreprocessorRegistry(List.of()));
        var unsupportedFailure = assertThrows(DiffusionConditioningPreparationException.class,
                () -> unsupported.prepare(new DiffusionConditioningSet(List.of(prompt)),
                        List.of(requirement(PROMPT, 1)), DiffusionConditioningPreparationContext.defaults()));
        assertEquals("prompt", unsupportedFailure.inputId());

        var ambiguous = orchestrator(
                new RecordingPreprocessor("one", PROMPT, new ArrayList<>()),
                new RecordingPreprocessor("two", PROMPT, new ArrayList<>()));
        var ambiguousFailure = assertThrows(DiffusionConditioningPreparationException.class,
                () -> ambiguous.prepare(new DiffusionConditioningSet(List.of(prompt)),
                        List.of(requirement(PROMPT, 1)), DiffusionConditioningPreparationContext.defaults()));
        assertTrue(ambiguousFailure.getMessage().contains("Ambiguous"));
    }

    @Test
    void validatesRequirementsBeforeInvokingAnyPreprocessor() {
        var events = new ArrayList<String>();
        var orchestrator = orchestrator(new RecordingPreprocessor("prompt", PROMPT, events));

        assertThrows(IllegalArgumentException.class,
                () -> orchestrator.prepare(
                        new DiffusionConditioningSet(List.of(input("prompt", PROMPT))),
                        List.of(requirement(PROMPT, 2)),
                        DiffusionConditioningPreparationContext.defaults()));
        assertTrue(events.isEmpty());
    }

    @Test
    void cleansUpPartialResultsInReverseOrderAndSuppressesCleanupFailures() {
        var events = new ArrayList<String>();
        var first = new RecordingPreprocessor("prompt", PROMPT, events);
        var second = new FailingPreprocessor("image", IMAGE, events);
        var orchestrator = orchestrator(first, second);

        var failure = assertThrows(IllegalStateException.class,
                () -> orchestrator.prepare(
                        new DiffusionConditioningSet(List.of(
                                input("prompt", PROMPT), input("image", IMAGE))),
                        List.of(requirement(PROMPT, 1), requirement(IMAGE, 1)),
                        DiffusionConditioningPreparationContext.defaults()));

        assertEquals(List.of("prepare:prompt", "prepare:image", "close:prompt"), events);
        assertEquals(1, failure.getSuppressed().length);
        assertEquals("cleanup:prompt", failure.getSuppressed()[0].getMessage());
    }

    @Test
    void rejectsDuplicatePreprocessorIdentifiers() {
        var one = new RecordingPreprocessor("same", PROMPT, new ArrayList<>());
        var two = new RecordingPreprocessor("same", IMAGE, new ArrayList<>());
        assertThrows(IllegalArgumentException.class,
                () -> new DiffusionConditioningPreprocessorRegistry(List.of(one, two)));
    }

    private static DiffusionConditioningPreparationOrchestrator orchestrator(
            DiffusionConditioningPreprocessor... preprocessors) {
        return new DiffusionConditioningPreparationOrchestrator(
                new DiffusionConditioningPreprocessorRegistry(List.of(preprocessors)));
    }

    private static DiffusionConditioningInput input(
            String id, DiffusionConditioningInput.Modality modality) {
        return new DiffusionConditioningInput(id, modality, "test:" + id);
    }

    private static DiffusionConditioningRequirement requirement(
            DiffusionConditioningInput.Modality modality, int count) {
        return new DiffusionConditioningRequirement(modality, count, count);
    }

    private static class RecordingPreprocessor implements DiffusionConditioningPreprocessor {
        private final String id;
        private final DiffusionConditioningInput.Modality modality;
        protected final List<String> events;

        private RecordingPreprocessor(
                String id, DiffusionConditioningInput.Modality modality, List<String> events) {
            this.id = id;
            this.modality = modality;
            this.events = events;
        }

        @Override public String id() { return id; }
        @Override public Set<DiffusionConditioningInput.Modality> modalities() { return Set.of(modality); }

        @Override
        public DiffusionConditioningData prepare(
                DiffusionConditioningInput input, DiffusionConditioningPreparationContext context) {
            events.add("prepare:" + input.id());
            return new TestData(input.id(), input.modality(), events);
        }
    }

    private static final class FailingPreprocessor extends RecordingPreprocessor {
        private FailingPreprocessor(String id, DiffusionConditioningInput.Modality modality,
                                    List<String> events) {
            super(id, modality, events);
        }

        @Override
        public DiffusionConditioningData prepare(
                DiffusionConditioningInput input, DiffusionConditioningPreparationContext context) {
            events.add("prepare:" + input.id());
            throw new IllegalStateException("prepare:" + input.id());
        }
    }

    private record TestData(String inputId, DiffusionConditioningInput.Modality modality,
                            List<String> events) implements DiffusionConditioningData {
        @Override
        public void close() {
            events.add("close:" + inputId);
            if ("prompt".equals(inputId)) {
                throw new IllegalStateException("cleanup:" + inputId);
            }
        }
    }
}
