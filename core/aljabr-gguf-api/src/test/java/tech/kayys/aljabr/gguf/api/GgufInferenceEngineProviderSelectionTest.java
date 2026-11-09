package tech.kayys.aljabr.gguf.api;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.exception.InferenceException;
import tech.kayys.aljabr.spi.inference.InferenceRequest;
import tech.kayys.aljabr.spi.inference.InferenceResponse;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
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
