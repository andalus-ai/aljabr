package tech.kayys.aljabr.safetensor.loader;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2VisionConfig;
import tech.kayys.aljabr.models.embeddinggemma2.vision.EmbeddingGemma2VisionImagePipeline;
import tech.kayys.aljabr.safetensor.embeddinggemma2.EmbeddingGemma2VisionSafetensorLoader;

/**
 * Opt-in compatibility test for the published google/embeddinggemma-2 checkpoint.
 *
 * <p>Set {@code ALJABR_EMBEDDING_GEMMA2_CHECKPOINT} to the local
 * {@code model.safetensors} path. The matching {@code config.json} and
 * {@code processor_config.json} must be in the same directory. The test
 * materializes the vision and image-bridge tensors, then builds the complete
 * Java image pipeline. It intentionally does not run the expensive vision
 * encoder, so it is suitable for ordinary CPU CI when explicitly enabled.</p>
 */
class EmbeddingGemma2OfficialCheckpointCompatibilityTest {
    private static final String CHECKPOINT_ENV = "ALJABR_EMBEDDING_GEMMA2_CHECKPOINT";

    @Test
    void mapsPublishedCheckpointAndBuildsImagePipeline() throws Exception {
        String configuredPath = System.getenv(CHECKPOINT_ENV);
        Assumptions.assumeTrue(configuredPath != null && !configuredPath.isBlank(),
                () -> "Set " + CHECKPOINT_ENV + " to run the official-checkpoint compatibility test");

        Path checkpoint = Path.of(configuredPath).toAbsolutePath().normalize();
        Assumptions.assumeTrue(Files.isRegularFile(checkpoint),
                () -> "Checkpoint does not exist: " + checkpoint);
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

        EmbeddingGemma2VisionImagePipeline pipeline =
                new EmbeddingGemma2VisionSafetensorLoader(new SafetensorShardLoader())
                        .load(checkpoint, visionConfig, textHiddenSize, 1e-6f);

        assertNotNull(pipeline);
    }
}
