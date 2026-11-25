package tech.kayys.aljabr.gguf.api;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.exception.InferenceException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

class GgufCapabilityAwareProviderSelectionTest {

    private static final GgufCapabilitySet AUDIO_INPUT = new GgufCapabilitySet(
            Set.of(GgufModality.TEXT, GgufModality.AUDIO),
            Set.of(GgufModality.TEXT),
            Set.of(GgufTask.TEXT_GENERATION, GgufTask.AUDIO_PERCEPTION));

    @Test
    void autoSelectionSkipsGenerationReadyProviderMissingRequiredAudioCapability() throws Exception {
        Path model = Files.createTempFile("gguf-capabilities", ".gguf");
        AtomicBoolean textCreated = new AtomicBoolean();
        AtomicBoolean omniCreated = new AtomicBoolean();
        GgufBackendProvider text = provider("text", 20,
                GgufCapabilitySet.textGeneration(), textCreated);
        GgufBackendProvider omni = provider("omni", 10, AUDIO_INPUT, omniCreated);

        try (GgufInferenceEngine engine = new GgufInferenceEngine(List.of(text, omni))) {
            GgufBackend backend = engine.loadModel("model", model, Map.of(), AUDIO_INPUT);
            assertEquals("omni", backend.name());
            assertFalse(textCreated.get());
            assertTrue(omniCreated.get());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void autoSelectionFailsWithActionableCapabilityDiagnostic() throws Exception {
        Path model = Files.createTempFile("gguf-capabilities", ".gguf");
        GgufBackendProvider text = provider(
                "text", 1, GgufCapabilitySet.textGeneration(), new AtomicBoolean());

        try (GgufInferenceEngine engine = new GgufInferenceEngine(List.of(text))) {
            InferenceException failure = assertThrows(InferenceException.class,
                    () -> engine.loadModel("model", model, Map.of(), AUDIO_INPUT));
            assertTrue(failure.getMessage().contains("missing GGUF capabilities"));
            assertTrue(failure.getMessage().contains("AUDIO"));
            assertTrue(failure.getMessage().contains("AUDIO_PERCEPTION"));
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void explicitProviderSelectionAlsoEnforcesDeclaredRequirements() throws Exception {
        Path model = Files.createTempFile("gguf-capabilities", ".gguf");
        AtomicBoolean created = new AtomicBoolean();
        GgufBackendProvider text = provider(
                "text", 1, GgufCapabilitySet.textGeneration(), created);

        try (GgufInferenceEngine engine = new GgufInferenceEngine(List.of(text))) {
            InferenceException failure = assertThrows(InferenceException.class,
                    () -> engine.loadModel(
                            "model", model, Map.of("gguf.backend", "text"), AUDIO_INPUT));
            assertTrue(failure.getMessage().contains("does not satisfy required capabilities"));
            assertFalse(created.get());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void legacyLoadModelKeepsGenerationOnlySelectionBehavior() throws Exception {
        Path model = Files.createTempFile("gguf-capabilities", ".gguf");
        AtomicBoolean created = new AtomicBoolean();
        GgufBackendProvider text = provider(
                "text", 1, GgufCapabilitySet.textGeneration(), created);

        try (GgufInferenceEngine engine = new GgufInferenceEngine(List.of(text))) {
            assertEquals("text", engine.loadModel("model", model, Map.of()).name());
            assertTrue(created.get());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    private static GgufBackendProvider provider(
            String id, int priority, GgufCapabilitySet capabilities, AtomicBoolean created) {
        return new GgufBackendProvider() {
            @Override public String id() { return id; }
            @Override public boolean isAvailable() { return true; }
            @Override public int priority() { return priority; }
            @Override public GgufBackendCapability probe(Path path) {
                return GgufBackendCapability.ready("test", Set.of(), Map.of(), capabilities);
            }
            @Override public GgufBackend create(Path path, Map<String, Object> config) {
                created.set(true);
                return new GgufBackend() {
                    @Override public String name() { return id; }
                    @Override public tech.kayys.aljabr.spi.inference.InferenceResponse execute(
                            tech.kayys.aljabr.spi.inference.InferenceRequest request) {
                        throw new UnsupportedOperationException("test backend");
                    }
                    @Override public void close() { }
                };
            }
        };
    }
}
