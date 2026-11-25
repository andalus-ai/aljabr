package tech.kayys.aljabr.models.whisper;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/** Lightweight, dependency-contained Hugging Face Whisper checkpoint inspection. */
public final class WhisperCheckpointInspector {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private WhisperCheckpointInspector() {
    }

    /**
     * Returns true when {@code config.json} identifies a Whisper conditional
     * generation checkpoint. This is model-family recognition, not proof that
     * the checkpoint is complete or executable by an installed provider.
     */
    public static boolean isWhisperCheckpoint(Path modelDirectory) {
        if (modelDirectory == null || !Files.isDirectory(modelDirectory)) {
            return false;
        }
        Path configPath = modelDirectory.resolve("config.json");
        if (!Files.isRegularFile(configPath)) {
            return false;
        }
        try {
            JsonNode root = MAPPER.readTree(configPath.toFile());
            if (root == null || !"whisper".equals(root.path("model_type").asText())) {
                return false;
            }
            JsonNode architectures = root.path("architectures");
            if (!architectures.isArray()) {
                return false;
            }
            for (JsonNode architecture : architectures) {
                if ("WhisperForConditionalGeneration".equals(architecture.asText())) {
                    return true;
                }
            }
            return false;
        } catch (IOException | RuntimeException ignored) {
            return false;
        }
    }

    /**
     * Validates the files required to identify and prepare the official
     * large-v3 checkpoint. Weight files may be a single file or sharded.
     */
    public static CheckpointReport inspectLargeV3(Path modelDirectory) {
        Objects.requireNonNull(modelDirectory, "modelDirectory must not be null");
        boolean modelConfig = false;
        boolean processorConfig = false;
        boolean tokenizer = false;
        boolean weights = false;
        String detail = "";

        try {
            WhisperLargeV3Config.fromCheckpoint(modelDirectory);
            modelConfig = true;
        } catch (Exception e) {
            detail = "model config: " + e.getMessage();
        }

        Path processorPath = modelDirectory.resolve("preprocessor_config.json");
        if (Files.isRegularFile(processorPath)) {
            try {
                JsonNode processor = MAPPER.readTree(processorPath.toFile());
                processorConfig = "WhisperFeatureExtractor".equals(
                        processor.path("feature_extractor_type").asText())
                        && processor.path("sampling_rate").asInt(-1) == WhisperLargeV3Config.EXPECTED_SAMPLE_RATE
                        && processor.path("feature_size").asInt(-1) == WhisperLargeV3Config.EXPECTED_MEL_BINS
                        && processor.path("n_fft").asInt(-1) == WhisperLargeV3Config.EXPECTED_FFT_SIZE
                        && processor.path("hop_length").asInt(-1) == WhisperLargeV3Config.EXPECTED_HOP_LENGTH
                        && processor.path("chunk_length").asInt(-1)
                                == WhisperLargeV3Config.EXPECTED_CHUNK_LENGTH_SECONDS;
            } catch (IOException e) {
                detail = append(detail, "processor config: " + e.getMessage());
            }
        }

        tokenizer = Files.isRegularFile(modelDirectory.resolve("tokenizer.json"))
                || (Files.isRegularFile(modelDirectory.resolve("vocab.json"))
                    && Files.isRegularFile(modelDirectory.resolve("merges.txt")));

        weights = Files.isRegularFile(modelDirectory.resolve("model.safetensors"))
                || Files.isRegularFile(modelDirectory.resolve("model.safetensors.index.json"));

        boolean supported = modelConfig && processorConfig && tokenizer && weights;
        if (!processorConfig) {
            detail = append(detail, "missing or incompatible preprocessor_config.json");
        }
        if (!tokenizer) {
            detail = append(detail, "missing tokenizer.json or vocab.json + merges.txt");
        }
        if (!weights) {
            detail = append(detail, "missing model.safetensors or model.safetensors.index.json");
        }
        return new CheckpointReport(modelConfig, processorConfig, tokenizer, weights, supported, detail);
    }

    private static String append(String current, String next) {
        return current == null || current.isBlank() ? next : current + "; " + next;
    }

    public record CheckpointReport(
            boolean modelConfigValid,
            boolean processorConfigValid,
            boolean tokenizerAvailable,
            boolean weightsAvailable,
            boolean supported,
            String diagnostics) {
    }
}
