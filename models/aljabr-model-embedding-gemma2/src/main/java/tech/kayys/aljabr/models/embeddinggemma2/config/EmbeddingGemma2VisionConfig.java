package tech.kayys.aljabr.models.embeddinggemma2.config;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.vision.EmbeddingGemma2VisionCheckpointMapper;

/**
 * Validated Gemma4 vision and image-processor configuration nested in an
 * EmbeddingGemma 2 checkpoint.
 *
 * <p>The vision dimensions come from the model's {@code vision_config};
 * image patch budget settings come from {@code preprocessor_config.json}.
 * Values duplicated by both files must agree. Defaults match the upstream
 * Gemma4VisionConfig and Gemma4ImageProcessor definitions.</p>
 */
public record EmbeddingGemma2VisionConfig(
        String modelType,
        int hiddenSize,
        int intermediateSize,
        int numHiddenLayers,
        int numAttentionHeads,
        int numKeyValueHeads,
        int headDim,
        int patchSize,
        int positionEmbeddingSize,
        int poolingKernelSize,
        int maxSoftTokens,
        boolean standardize,
        double rmsNormEps,
        double ropeTheta,
        String hiddenActivation,
        boolean useClippedLinears) {

    public static final String MODEL_TYPE = "gemma4_vision";
    public static final int DEFAULT_HIDDEN_SIZE = 768;
    public static final int DEFAULT_INTERMEDIATE_SIZE = 3072;
    public static final int DEFAULT_NUM_HIDDEN_LAYERS = 16;
    public static final int DEFAULT_NUM_ATTENTION_HEADS = 12;
    public static final int DEFAULT_NUM_KEY_VALUE_HEADS = 12;
    public static final int DEFAULT_HEAD_DIM = 64;
    public static final int DEFAULT_PATCH_SIZE = 16;
    public static final int DEFAULT_POSITION_EMBEDDING_SIZE = 10 * 1024;
    public static final int DEFAULT_POOLING_KERNEL_SIZE = 3;
    public static final int DEFAULT_MAX_SOFT_TOKENS = 280;
    public static final double DEFAULT_RMS_NORM_EPS = 1e-6;
    public static final double DEFAULT_ROPE_THETA = 100.0;
    public static final String DEFAULT_HIDDEN_ACTIVATION = "gelu_pytorch_tanh";

    public EmbeddingGemma2VisionConfig {
        modelType = requireModelType(modelType);
        positive(hiddenSize, "hiddenSize");
        positive(intermediateSize, "intermediateSize");
        positive(numHiddenLayers, "numHiddenLayers");
        positive(numAttentionHeads, "numAttentionHeads");
        positive(numKeyValueHeads, "numKeyValueHeads");
        positive(headDim, "headDim");
        positive(patchSize, "patchSize");
        positive(positionEmbeddingSize, "positionEmbeddingSize");
        positive(poolingKernelSize, "poolingKernelSize");
        validateSoftTokens(maxSoftTokens);
        if (numAttentionHeads % numKeyValueHeads != 0) {
            throw new IllegalArgumentException("numAttentionHeads must be divisible by numKeyValueHeads");
        }
        if (Math.multiplyExact(numAttentionHeads, headDim) != hiddenSize) {
            throw new IllegalArgumentException("numAttentionHeads * headDim must equal hiddenSize");
        }
        if (!Double.isFinite(rmsNormEps) || rmsNormEps <= 0d) {
            throw new IllegalArgumentException("rmsNormEps must be finite and positive");
        }
        if (!Double.isFinite(ropeTheta) || ropeTheta <= 0d) {
            throw new IllegalArgumentException("ropeTheta must be finite and positive");
        }
        if (!DEFAULT_HIDDEN_ACTIVATION.equals(hiddenActivation)) {
            throw new IllegalArgumentException("unsupported Gemma4 vision activation: " + hiddenActivation);
        }
        if (useClippedLinears) {
            throw new IllegalArgumentException("use_clipped_linears=true is not supported by the current reference linear kernels");
        }
    }

    /**
     * Parses model config and optional preprocessor config. The model config
     * may be the full EmbeddingGemma 2 config or the vision_config object.
     */
    public static EmbeddingGemma2VisionConfig fromJson(
            JsonNode modelConfig, JsonNode processorConfig) {
        Objects.requireNonNull(modelConfig, "modelConfig must not be null");
        JsonNode vision = modelConfig.path("vision_config");
        if (!vision.isObject()) {
            vision = modelConfig;
        }
        if (!vision.isObject()) {
            throw new IllegalArgumentException("vision_config must be an object");
        }

        String modelType = optionalText(vision, "model_type", MODEL_TYPE);
        int patchSize = positiveInt(vision, "patch_size", DEFAULT_PATCH_SIZE);
        int poolingKernel = positiveInt(vision, "pooling_kernel_size", DEFAULT_POOLING_KERNEL_SIZE);

        // EmbeddingGemma2Processor stores image settings under image_processor,
        // while standalone Gemma4 processor configs may expose them at the root.
        JsonNode imageProcessor = processorConfig;
        if (processorConfig != null && processorConfig.isObject()
                && processorConfig.path("image_processor").isObject()) {
            imageProcessor = processorConfig.path("image_processor");
        }
        if (imageProcessor != null && !imageProcessor.isNull() && !imageProcessor.isMissingNode()) {
            int processorPatch = positiveInt(imageProcessor, "patch_size", patchSize);
            int processorPool = positiveInt(imageProcessor, "pooling_kernel_size", poolingKernel);
            if (processorPatch != patchSize) {
                throw new IllegalArgumentException("preprocessor image_processor.patch_size does not match vision_config.patch_size");
            }
            if (processorPool != poolingKernel) {
                throw new IllegalArgumentException(
                        "preprocessor image_processor.pooling_kernel_size does not match vision_config.pooling_kernel_size");
            }
        }
        int maxSoftTokens = imageProcessor == null || imageProcessor.isNull()
                ? DEFAULT_MAX_SOFT_TOKENS
                : positiveInt(imageProcessor, "max_soft_tokens", DEFAULT_MAX_SOFT_TOKENS);

        double ropeTheta = DEFAULT_ROPE_THETA;
        JsonNode rope = vision.path("rope_parameters");
        if (rope.isObject()) {
            JsonNode theta = rope.path("rope_theta");
            if (!theta.isNumber()) theta = rope.path("theta");
            if (theta.isNumber()) ropeTheta = theta.asDouble();
        }

        return new EmbeddingGemma2VisionConfig(
                modelType,
                positiveInt(vision, "hidden_size", DEFAULT_HIDDEN_SIZE),
                positiveInt(vision, "intermediate_size", DEFAULT_INTERMEDIATE_SIZE),
                positiveInt(vision, "num_hidden_layers", DEFAULT_NUM_HIDDEN_LAYERS),
                positiveInt(vision, "num_attention_heads", DEFAULT_NUM_ATTENTION_HEADS),
                positiveInt(vision, "num_key_value_heads", DEFAULT_NUM_KEY_VALUE_HEADS),
                positiveInt(vision, "head_dim", DEFAULT_HEAD_DIM),
                patchSize,
                positiveInt(vision, "position_embedding_size", DEFAULT_POSITION_EMBEDDING_SIZE),
                poolingKernel,
                maxSoftTokens,
                optionalBoolean(vision, "standardize", false),
                positiveDouble(vision, "rms_norm_eps", DEFAULT_RMS_NORM_EPS),
                ropeTheta,
                optionalText(vision, "hidden_activation", DEFAULT_HIDDEN_ACTIVATION),
                optionalBoolean(vision, "use_clipped_linears", false));
    }

    public EmbeddingGemma2VisionCheckpointMapper.VisionSpec toVisionSpec() {
        return new EmbeddingGemma2VisionCheckpointMapper.VisionSpec(
                hiddenSize, intermediateSize, numHiddenLayers, numAttentionHeads,
                numKeyValueHeads, headDim, patchSize, positionEmbeddingSize, standardize);
    }

    private static String requireModelType(String value) {
        if (!MODEL_TYPE.equals(value)) {
            throw new IllegalArgumentException("vision model_type must be '" + MODEL_TYPE + "' but was '" + value + "'");
        }
        return value;
    }

    private static void validateSoftTokens(int value) {
        if (value != 70 && value != 140 && value != 280 && value != 560 && value != 1120) {
            throw new IllegalArgumentException("maxSoftTokens must be one of [70, 140, 280, 560, 1120]");
        }
    }

    private static void positive(int value, String name) {
        if (value <= 0) throw new IllegalArgumentException(name + " must be positive");
    }

    private static int positiveInt(JsonNode node, String field, int fallback) {
        JsonNode value = node.path(field);
        if (value.isMissingNode() || value.isNull()) return fallback;
        if (!value.canConvertToInt() || value.asInt() <= 0) {
            throw new IllegalArgumentException(field + " must be a positive integer");
        }
        return value.asInt();
    }

    private static double positiveDouble(JsonNode node, String field, double fallback) {
        JsonNode value = node.path(field);
        if (value.isMissingNode() || value.isNull()) return fallback;
        if (!value.isNumber() || !Double.isFinite(value.asDouble()) || value.asDouble() <= 0d) {
            throw new IllegalArgumentException(field + " must be a finite positive number");
        }
        return value.asDouble();
    }

    private static String optionalText(JsonNode node, String field, String fallback) {
        JsonNode value = node.path(field);
        if (value.isMissingNode() || value.isNull()) return fallback;
        if (!value.isTextual() || value.asText().isBlank()) {
            throw new IllegalArgumentException(field + " must be a non-blank string");
        }
        return value.asText();
    }

    private static boolean optionalBoolean(JsonNode node, String field, boolean fallback) {
        JsonNode value = node.path(field);
        if (value.isMissingNode() || value.isNull()) return fallback;
        if (!value.isBoolean()) throw new IllegalArgumentException(field + " must be a boolean");
        return value.asBoolean();
    }
}
