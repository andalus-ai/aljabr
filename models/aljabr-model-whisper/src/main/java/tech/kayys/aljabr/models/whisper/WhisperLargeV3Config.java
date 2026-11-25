package tech.kayys.aljabr.models.whisper;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Validated architectural configuration for {@code openai/whisper-large-v3}.
 *
 * <p>These values describe the upstream checkpoint; they are not a claim that
 * Aljabr already has a complete Whisper encoder/decoder execution provider.</p>
 */
public record WhisperLargeV3Config(
        int dModel,
        int encoderLayers,
        int decoderLayers,
        int encoderAttentionHeads,
        int decoderAttentionHeads,
        int encoderFfnDim,
        int decoderFfnDim,
        int numMelBins,
        int maxSourcePositions,
        int maxTargetPositions,
        int vocabSize,
        int sampleRate,
        int fftSize,
        int hopLength,
        int chunkLengthSeconds) {

    public static final String MODEL_ID = "openai/whisper-large-v3";
    public static final int EXPECTED_D_MODEL = 1280;
    public static final int EXPECTED_ENCODER_LAYERS = 32;
    public static final int EXPECTED_DECODER_LAYERS = 32;
    public static final int EXPECTED_ATTENTION_HEADS = 20;
    public static final int EXPECTED_FFN_DIM = 5120;
    public static final int EXPECTED_MEL_BINS = 128;
    public static final int EXPECTED_MAX_SOURCE_POSITIONS = 1500;
    public static final int EXPECTED_MAX_TARGET_POSITIONS = 448;
    public static final int EXPECTED_VOCAB_SIZE = 51866;
    public static final int EXPECTED_SAMPLE_RATE = 16000;
    public static final int EXPECTED_FFT_SIZE = 400;
    public static final int EXPECTED_HOP_LENGTH = 160;
    public static final int EXPECTED_CHUNK_LENGTH_SECONDS = 30;

    public WhisperLargeV3Config {
        if (dModel <= 0 || encoderLayers <= 0 || decoderLayers <= 0
                || encoderAttentionHeads <= 0 || decoderAttentionHeads <= 0
                || encoderFfnDim <= 0 || decoderFfnDim <= 0 || numMelBins <= 0
                || maxSourcePositions <= 0 || maxTargetPositions <= 0 || vocabSize <= 0
                || sampleRate <= 0 || fftSize <= 0 || hopLength <= 0 || chunkLengthSeconds <= 0) {
            throw new IllegalArgumentException("Whisper configuration dimensions must all be positive");
        }
    }

    /** Canonical configuration expected from the public large-v3 checkpoint. */
    public static WhisperLargeV3Config canonical() {
        return new WhisperLargeV3Config(
                EXPECTED_D_MODEL,
                EXPECTED_ENCODER_LAYERS,
                EXPECTED_DECODER_LAYERS,
                EXPECTED_ATTENTION_HEADS,
                EXPECTED_ATTENTION_HEADS,
                EXPECTED_FFN_DIM,
                EXPECTED_FFN_DIM,
                EXPECTED_MEL_BINS,
                EXPECTED_MAX_SOURCE_POSITIONS,
                EXPECTED_MAX_TARGET_POSITIONS,
                EXPECTED_VOCAB_SIZE,
                EXPECTED_SAMPLE_RATE,
                EXPECTED_FFT_SIZE,
                EXPECTED_HOP_LENGTH,
                EXPECTED_CHUNK_LENGTH_SECONDS);
    }

    /**
     * Reads and validates the model architecture fields in a Hugging Face
     * {@code config.json}. Audio preprocessing fields are validated separately
     * from {@code preprocessor_config.json}.
     */
    public static WhisperLargeV3Config fromCheckpoint(Path checkpointDirectory) throws IOException {
        Objects.requireNonNull(checkpointDirectory, "checkpointDirectory must not be null");
        Path configPath = checkpointDirectory.resolve("config.json");
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(configPath.toFile());
        if (root == null || !"whisper".equals(root.path("model_type").asText())) {
            throw new IllegalArgumentException("Checkpoint is not a Hugging Face Whisper model: " + configPath);
        }
        JsonNode architectures = root.path("architectures");
        boolean supportedArchitecture = false;
        if (architectures.isArray()) {
            for (JsonNode architecture : architectures) {
                if ("WhisperForConditionalGeneration".equals(architecture.asText())) {
                    supportedArchitecture = true;
                    break;
                }
            }
        }
        if (!supportedArchitecture) {
            throw new IllegalArgumentException(
                    "Expected WhisperForConditionalGeneration in " + configPath);
        }

        WhisperLargeV3Config actual = new WhisperLargeV3Config(
                requiredInt(root, "d_model"),
                requiredInt(root, "encoder_layers"),
                requiredInt(root, "decoder_layers"),
                requiredInt(root, "encoder_attention_heads"),
                requiredInt(root, "decoder_attention_heads"),
                requiredInt(root, "encoder_ffn_dim"),
                requiredInt(root, "decoder_ffn_dim"),
                requiredInt(root, "num_mel_bins"),
                requiredInt(root, "max_source_positions"),
                requiredInt(root, "max_target_positions"),
                requiredInt(root, "vocab_size"),
                EXPECTED_SAMPLE_RATE,
                EXPECTED_FFT_SIZE,
                EXPECTED_HOP_LENGTH,
                EXPECTED_CHUNK_LENGTH_SECONDS);
        actual.requireLargeV3Architecture();
        return actual;
    }

    /** Rejects similarly named or dimensionally incompatible Whisper checkpoints. */
    public void requireLargeV3Architecture() {
        WhisperLargeV3Config expected = canonical();
        if (dModel != expected.dModel
                || encoderLayers != expected.encoderLayers
                || decoderLayers != expected.decoderLayers
                || encoderAttentionHeads != expected.encoderAttentionHeads
                || decoderAttentionHeads != expected.decoderAttentionHeads
                || encoderFfnDim != expected.encoderFfnDim
                || decoderFfnDim != expected.decoderFfnDim
                || numMelBins != expected.numMelBins
                || maxSourcePositions != expected.maxSourcePositions
                || maxTargetPositions != expected.maxTargetPositions
                || vocabSize != expected.vocabSize) {
            throw new IllegalArgumentException(
                    "Checkpoint architecture does not match " + MODEL_ID + ": " + this);
        }
    }

    private static int requiredInt(JsonNode root, String field) {
        JsonNode value = root.get(field);
        if (value == null || !value.canConvertToInt() || value.intValue() <= 0) {
            throw new IllegalArgumentException("Missing or invalid positive integer '" + field + "' in config.json");
        }
        return value.intValue();
    }
}
