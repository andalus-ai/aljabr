package tech.kayys.aljabr.gguf.api;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.diagnostics.ExecutionEvent;
import tech.kayys.aljabr.diagnostics.ProviderDecisionEvent;
import tech.kayys.aljabr.spi.exception.InferenceException;
import tech.kayys.aljabr.spi.inference.InferenceRequest;
import tech.kayys.aljabr.spi.inference.InferenceResponse;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

class GgufInferenceEngineProviderSelectionTest {

    @Test
    void autoSkipsJavaWhenNotReadyAndSelectsReadyLlamaCppFallback() throws Exception {
        Path model = Files.createTempFile("selection", ".gguf");
        AtomicBoolean javaCreated = new AtomicBoolean();
        AtomicBoolean llamaCreated = new AtomicBoolean();
        GgufBackendProvider java = provider("java", 10, false, javaCreated);
        GgufBackendProvider llama = provider("llamacpp", -10, true, llamaCreated);

        try (GgufInferenceEngine engine = new GgufInferenceEngine(List.of(llama, java))) {
            GgufBackend selected = engine.loadModel("model", model, Map.of());
            assertEquals("llamacpp", selected.name());
            assertFalse(javaCreated.get());
            assertTrue(llamaCreated.get());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void autoFailsClearlyWhenNoProviderReportsGenerationReady() throws Exception {
        Path model = Files.createTempFile("selection", ".gguf");
        AtomicBoolean javaCreated = new AtomicBoolean();
        GgufBackendProvider java = provider("java", 10, false, javaCreated);

        try (GgufInferenceEngine engine = new GgufInferenceEngine(List.of(java))) {
            InferenceException error = assertThrows(
                    InferenceException.class,
                    () -> engine.loadModel("model", model, Map.of()));
            assertTrue(error.getMessage().contains("No GGUF backend reports generation readiness"));
            assertTrue(error.getMessage().contains("java: generation is not ready"));
            assertFalse(javaCreated.get());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void explicitJavaSelectionRemainsAvailableForDeliberateTesting() throws Exception {
        Path model = Files.createTempFile("selection", ".gguf");
        AtomicBoolean javaCreated = new AtomicBoolean();
        GgufBackendProvider java = provider("java", 10, false, javaCreated);

        try (GgufInferenceEngine engine = new GgufInferenceEngine(List.of(java))) {
            GgufBackend selected = engine.loadModel("model", model, Map.of("gguf.backend", "java"));
            assertEquals("java", selected.name());
            assertTrue(javaCreated.get());
        } finally {
            Files.deleteIfExists(model);
        }
    }


    @Test
    void reportsNotReadyProviderAndSelectedFallback() throws Exception {
        Path model = Files.createTempFile("selection", ".gguf");
        List<ProviderDecisionEvent> events = new ArrayList<>();
        GgufBackendProvider java = provider("java", 10, false, new AtomicBoolean());
        GgufBackendProvider llama = provider("llamacpp", -10, true, new AtomicBoolean());

        try (GgufInferenceEngine engine = new GgufInferenceEngine(List.of(llama, java), events::add)) {
            GgufBackend selected = engine.loadModel("model", model, Map.of());

            assertEquals("llamacpp", selected.name());
            assertEquals(2, events.size());
            assertEquals("java", events.get(0).providerId());
            assertEquals(ProviderDecisionEvent.Decision.NOT_READY, events.get(0).decision());
            assertEquals("llamacpp", events.get(1).providerId());
            assertEquals(ProviderDecisionEvent.Decision.SELECTED, events.get(1).decision());
            assertEquals(ProviderDecisionEvent.SelectionMode.AUTO, events.get(1).selectionMode());
            assertTrue(events.get(1).reason().contains("skipped"));
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void providerObserverFailureDoesNotPreventBackendSelection() throws Exception {
        Path model = Files.createTempFile("selection", ".gguf");
        GgufBackendProvider ready = provider("ready", 1, true, new AtomicBoolean());

        try (GgufInferenceEngine engine = new GgufInferenceEngine(
                List.of(ready), event -> { throw new IllegalStateException("observer failed"); })) {
            GgufBackend selected = engine.loadModel("model", model, Map.of());

            assertEquals("ready", selected.name());
            assertEquals(1, engine.providerObserverFailureCount());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void inferenceExecutionEmitsLifecycleFailureWithoutRequestPayload() throws Exception {
        Path model = Files.createTempFile("selection", ".gguf");
        List<ExecutionEvent> events = new ArrayList<>();
        GgufBackendProvider ready = provider("java", 1, true, new AtomicBoolean());

        try (GgufInferenceEngine engine = new GgufInferenceEngine(
                List.of(ready), event -> { }, events::add)) {
            engine.loadModel("model", model, Map.of());
            InferenceRequest request = InferenceRequest.builder()
                    .model("model")
                    .prompt("private prompt must not be captured")
                    .build();

            assertThrows(InferenceException.class, () -> engine.execute("model", request));
            assertEquals(2, events.size());
            assertEquals(ExecutionEvent.Type.STARTED, events.get(0).type());
            assertEquals(ExecutionEvent.Type.FAILED, events.get(1).type());
            assertEquals("gguf.infer", events.get(1).context().operation());
            assertEquals("java", events.get(1).context().provider());
            assertEquals("java.lang.UnsupportedOperationException", events.get(1).failureType());
            assertFalse(events.get(1).context().attributes().values().stream()
                    .anyMatch(value -> value.contains("private prompt")));
            assertTrue(events.get(1).elapsedNanos() >= 0);
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void explicitUnavailableProviderIsReportedAsUnavailable() throws Exception {
        Path model = Files.createTempFile("selection", ".gguf");
        List<ProviderDecisionEvent> events = new ArrayList<>();
        GgufBackendProvider unavailable = new GgufBackendProvider() {
            @Override public String id() { return "offline"; }
            @Override public boolean isAvailable() { return false; }
            @Override public GgufBackendCapability probe(Path path) { throw new UnsupportedOperationException(); }
            @Override public GgufBackend create(Path path, Map<String, Object> config) {
                throw new UnsupportedOperationException();
            }
        };

        try (GgufInferenceEngine engine = new GgufInferenceEngine(List.of(unavailable), events::add)) {
            assertThrows(InferenceException.class,
                    () -> engine.loadModel("model", model, Map.of("gguf.backend", "offline")));
            assertEquals(1, events.size());
            assertEquals(ProviderDecisionEvent.Decision.UNAVAILABLE, events.get(0).decision());
            assertEquals(ProviderDecisionEvent.SelectionMode.EXPLICIT, events.get(0).selectionMode());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    private static GgufBackendProvider provider(
            String id, int priority, boolean ready, AtomicBoolean created) {
        return new GgufBackendProvider() {
            @Override public String id() { return id; }
            @Override public boolean isAvailable() { return true; }
            @Override public GgufBackendCapability probe(Path path) {
                return ready
                        ? GgufBackendCapability.ready("test", Set.of(), Map.of())
                        : GgufBackendCapability.notReady("test", Set.of(), "generation is not ready", Map.of());
            }
            @Override public GgufBackend create(Path path, Map<String, Object> config) {
                created.set(true);
                return new GgufBackend() {
                    @Override public String name() { return id; }
                    @Override public InferenceResponse execute(InferenceRequest request) {
                        throw new UnsupportedOperationException();
                    }
                    @Override public void close() {}
                };
            }
            @Override public int priority() { return priority; }
        };
    }
}
