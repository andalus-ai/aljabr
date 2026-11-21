package tech.kayys.aljabr.safetensor.loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.lang.reflect.Proxy;
import java.util.Map;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2VisionConfig;
import tech.kayys.aljabr.models.embeddinggemma2.vision.EmbeddingGemma2VisionImagePipeline;
import tech.kayys.aljabr.models.embeddinggemma2.vision.EmbeddingGemma2VisionImageProcessor;
import tech.kayys.aljabr.safetensor.embeddinggemma2.EmbeddingGemma2VisionSafetensorLoader;

/**
 * Opt-in compatibility tests for the published google/embeddinggemma-2 checkpoint.
 *
 * <p>Set {@code ALJABR_EMBEDDING_GEMMA2_CHECKPOINT} to the local
 * {@code model.safetensors} path. The matching {@code config.json} and
 * {@code processor_config.json} must be in the same directory. The normal
 * compatibility test maps tensors and constructs the complete Java image
 * pipeline. The full forward test additionally requires
 * {@code ALJABR_EMBEDDING_GEMMA2_RUN_FORWARD=true}; it is intentionally
 * separate because the official image processor expands inputs to its
 * configured soft-token budget and the 16-layer vision encoder is expensive.</p>
 */
class EmbeddingGemma2OfficialCheckpointCompatibilityTest {
    private static final String CHECKPOINT_ENV = "ALJABR_EMBEDDING_GEMMA2_CHECKPOINT";
    private static final String FORWARD_ENV = "ALJABR_EMBEDDING_GEMMA2_RUN_FORWARD";

    @Test
    void mapsPublishedCheckpointAndBuildsImagePipeline() throws Exception {
        Path checkpoint = configuredCheckpoint();
        assertNotNull(loadPipeline(checkpoint));
    }

    @Test
    void executesOfficialCheckpointThroughRealImagePipeline() throws Exception {
        Assumptions.assumeTrue("true".equalsIgnoreCase(System.getenv(FORWARD_ENV)),
                () -> "Set " + FORWARD_ENV + "=true to run the full official-checkpoint vision forward pass");

        EmbeddingGemma2VisionImagePipeline pipeline = loadPipeline(configuredCheckpoint());
        EmbeddingGemma2VisionImageProcessor.RgbImage image = gradientImage(16, 16);

        int expectedSoftTokens = pipeline.preprocess(List.of(image)).numSoftTokensPerImage()[0];
        float[][] features = pipeline.encode(image);

        assertEquals(expectedSoftTokens, features.length,
                "vision output must preserve the processor's actual per-image soft-token count");
        assertTrue(features.length > 0, "vision output must contain image tokens");
        assertTrue(features[0].length > 0, "vision output must use the configured text hidden size");

        boolean hasNonZeroFeature = false;
        for (float[] token : features) {
            assertEquals(features[0].length, token.length, "all image tokens must have the same text width");
            for (float value : token) {
                assertTrue(Float.isFinite(value), "official checkpoint output must be finite");
                hasNonZeroFeature |= Math.abs(value) > 1e-7f;
            }
        }
        assertTrue(hasNonZeroFeature, "official checkpoint must produce non-zero image features");
    }

    private static Path configuredCheckpoint() {
        String configuredPath = System.getenv(CHECKPOINT_ENV);
        Assumptions.assumeTrue(configuredPath != null && !configuredPath.isBlank(),
                () -> "Set " + CHECKPOINT_ENV + " to run the official-checkpoint compatibility test");

        Path checkpoint = Path.of(configuredPath).toAbsolutePath().normalize();
        Assumptions.assumeTrue(Files.isRegularFile(checkpoint),
                () -> "Checkpoint does not exist: " + checkpoint);
        return checkpoint;
    }

    private static EmbeddingGemma2VisionImagePipeline loadPipeline(Path checkpoint) throws Exception {
        Path directory = checkpoint.getParent();
        Path modelConfigPath = directory.resolve("config.json");
        Path processorConfigPath = directory.resolve("processor_config.json");
        assertTrue(Files.isRegularFile(modelConfigPath), "config.json must sit beside the checkpoint");
        assertTrue(Files.isRegularFile(processorConfigPath),
                "processor_config.json must sit beside the checkpoint");

        ObjectMapper json = new ObjectMapper();
        JsonNode modelConfig = json.readTree(modelConfigPath.toFile());
        JsonNode processorConfig = json.readTree(processorConfigPath.toFile());
        EmbeddingGemma2VisionConfig visionConfig =
                EmbeddingGemma2VisionConfig.fromJson(modelConfig, processorConfig);

        JsonNode textConfig = modelConfig.path("text_config");
        assertTrue(textConfig.isObject(), "config.json must contain text_config");
        int textHiddenSize = textConfig.path("hidden_size").asInt(0);
        assertTrue(textHiddenSize > 0, "text_config.hidden_size must be positive");

        return new EmbeddingGemma2VisionSafetensorLoader(testShardLoader(json))
                .load(checkpoint, visionConfig, textHiddenSize, 1e-6f);
    }

    /**
     * Builds the same loader graph as CDI for this plain JUnit test. The
     * production loader beans use field injection, so constructing only the
     * outer SafetensorShardLoader with {@code new} leaves its dependencies
     * unset and makes an opt-in real-checkpoint test fail before reading a file.
     */
    private static SafetensorShardLoader testShardLoader(ObjectMapper json) {
        SafetensorLoaderConfig.Validation validation = proxy(
                SafetensorLoaderConfig.Validation.class,
                Map.of("strict", true, "warnOnEmptyTensors", true, "maxHeaderBytes", 104857600L));
        SafetensorLoaderConfig.Cache cache = proxy(
                SafetensorLoaderConfig.Cache.class,
                Map.of("enabled", false, "maxSize", 1, "ttlSeconds", 300L));
        SafetensorLoaderConfig config = proxy(
                SafetensorLoaderConfig.class,
                Map.of("preferMmap", true, "readChunkBytes", 8388608,
                        "validation", validation, "cache", cache, "logLoadSummary", false));

        SafetensorMetrics metrics = new SafetensorMetrics();
        metrics.registry = new SimpleMeterRegistry();
        metrics.registerGauge();

        SafetensorFFMLoader fileLoader = new SafetensorFFMLoader();
        fileLoader.config = config;
        fileLoader.objectMapper = json;
        fileLoader.metrics = metrics;

        SafetensorShardLoader shardLoader = new SafetensorShardLoader();
        shardLoader.ffmLoader = fileLoader;
        shardLoader.objectMapper = json;
        return shardLoader;
    }

    private static <T> T proxy(Class<T> type, Map<String, Object> values) {
        return type.cast(Proxy.newProxyInstance(type.getClassLoader(), new Class<?>[] {type},
                (instance, method, arguments) -> {
                    if (method.getDeclaringClass() == Object.class) {
                        return switch (method.getName()) {
                            case "toString" -> type.getSimpleName() + "TestProxy";
                            case "hashCode" -> System.identityHashCode(instance);
                            case "equals" -> instance == arguments[0];
                            default -> throw new UnsupportedOperationException(method.getName());
                        };
                    }
                    if (values.containsKey(method.getName())) {
                        return values.get(method.getName());
                    }
                    throw new UnsupportedOperationException("Unexpected config access: " + method.getName());
                }));
    }

    private static EmbeddingGemma2VisionImageProcessor.RgbImage gradientImage(int width, int height) {
        byte[] rgb = new byte[Math.multiplyExact(Math.multiplyExact(width, height), 3)];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int offset = (y * width + x) * 3;
                rgb[offset] = (byte) ((x * 17 + y * 3) & 0xff);
                rgb[offset + 1] = (byte) ((x * 5 + y * 19) & 0xff);
                rgb[offset + 2] = (byte) ((x * 11 + y * 7) & 0xff);
            }
        }
        return EmbeddingGemma2VisionImageProcessor.RgbImage.fromRgb8(width, height, rgb);
    }
}
