package tech.kayys.aljabr.models.embeddinggemma2.config;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Validated Gemma4 audio-tower and audio-feature-extractor configuration
 * nested in an EmbeddingGemma 2 checkpoint.
 *
 * <p>Model geometry is read from {@code config.json#audio_config}; waveform
 * feature extraction settings are read from
 * {@code processor_config.json#feature_extractor}. This type deliberately
 * describes the published contract only; it does not perform decoding,
 * resampling, FFT, or log-mel extraction.</p>
 */
public record EmbeddingGemma2AudioConfig(
        String modelType,
        int hiddenSize,
        int numHiddenLayers,
        int numAttentionHeads,
        int outputProjectionSize,
        int convolutionKernelSize,
        List<Integer> subsamplingConvolutionChannels,
        int attentionChunkSize,
        int attentionContextLeft,
        int attentionContextRight,
        double rmsNormEps,
        double residualWeight,
        String hiddenActivation,
        String dtype,
        boolean useClippedLinears,
        int samplingRate,
        int featureSize,
        int fftLength,
        int frameLength,
        int hopLength,
        double minFrequency,
        double maxFrequency,
        double melFloor,
        double paddingValue,
        double inputScaleFactor,
        double dither,
        int audioSequenceLength,
        int audioMillisecondsPerToken,
        int audioTokenId,
        int beginAudioTokenId,
        int endAudioTokenId,
        double gradientClipping,
        double attentionLogitCap,
        double attentionInvalidLogitsValue) {

    public static final String MODEL_TYPE = "gemma4_audio";
    public static final String FEATURE_EXTRACTOR_TYPE = "Gemma4AudioFeatureExtractor";

    public EmbeddingGemma2AudioConfig {
        if (!MODEL_TYPE.equals(modelType)) {
            throw new IllegalArgumentException("audio model_type must be '" + MODEL_TYPE + "'");
        }
        positive(hiddenSize, "hiddenSize");
        positive(numHiddenLayers, "numHiddenLayers");
        positive(numAttentionHeads, "numAttentionHeads");
        positive(outputProjectionSize, "outputProjectionSize");
        positive(convolutionKernelSize, "convolutionKernelSize");
        positive(attentionChunkSize, "attentionChunkSize");
        positive(attentionContextLeft, "attentionContextLeft");
        nonNegative(attentionContextRight, "attentionContextRight");
        positive(samplingRate, "samplingRate");
        positive(featureSize, "featureSize");
        positive(fftLength, "fftLength");
        positive(frameLength, "frameLength");
        positive(hopLength, "hopLength");
        positive(audioSequenceLength, "audioSequenceLength");
        positive(audioMillisecondsPerToken, "audioMillisecondsPerToken");
        if (subsamplingConvolutionChannels == null || subsamplingConvolutionChannels.isEmpty()) {
            throw new IllegalArgumentException("subsamplingConvolutionChannels must not be empty");
        }
        subsamplingConvolutionChannels = List.copyOf(subsamplingConvolutionChannels);
        subsamplingConvolutionChannels.forEach(value -> positive(value, "subsampling convolution channel"));
        positiveFinite(rmsNormEps, "rmsNormEps");
        positiveFinite(residualWeight, "residualWeight");
        positiveFinite(gradientClipping, "gradientClipping");
        positiveFinite(attentionLogitCap, "attentionLogitCap");
        finite(attentionInvalidLogitsValue, "attentionInvalidLogitsValue");
        nonNegativeFinite(minFrequency, "minFrequency");
        positiveFinite(maxFrequency, "maxFrequency");
        positiveFinite(melFloor, "melFloor");
        nonNegativeFinite(dither, "dither");
        finite(paddingValue, "paddingValue");
        positiveFinite(inputScaleFactor, "inputScaleFactor");
        if (maxFrequency <= minFrequency || maxFrequency > samplingRate / 2.0) {
            throw new IllegalArgumentException("audio frequency range must satisfy 0 <= min < max <= Nyquist");
        }
        if (frameLength > fftLength) {
            throw new IllegalArgumentException("frameLength must not exceed fftLength");
        }
        if (numAttentionHeads <= 0 || hiddenSize % numAttentionHeads != 0) {
            throw new IllegalArgumentException("hiddenSize must be divisible by numAttentionHeads");
        }
        if (!"silu".equals(hiddenActivation)) {
            throw new IllegalArgumentException("unsupported Gemma4 audio activation: " + hiddenActivation);
        }
        if (!"bfloat16".equalsIgnoreCase(dtype) && !"float32".equalsIgnoreCase(dtype)) {
            throw new IllegalArgumentException("EmbeddingGemma 2 audio supports bfloat16 or float32, not: " + dtype);
        }
        if (audioTokenId < 0 || beginAudioTokenId < 0 || endAudioTokenId < 0) {
            throw new IllegalArgumentException("audio token IDs must be non-negative");
        }
    }

    /** Parses the published model config and processor config together. */
    public static EmbeddingGemma2AudioConfig fromJson(JsonNode modelConfig, JsonNode processorConfig) {
        Objects.requireNonNull(modelConfig, "modelConfig must not be null");
        JsonNode audio = modelConfig.path("audio_config");
        if (!audio.isObject()) {
            throw new IllegalArgumentException("EmbeddingGemma 2 requires an object audio_config");
        }
        JsonNode extractor = processorConfig == null ? null : processorConfig.path("feature_extractor");
        if (extractor == null || !extractor.isObject()) {
            throw new IllegalArgumentException("processor_config.feature_extractor must be an object");
        }
        String extractorType = text(extractor, "feature_extractor_type");
        if (!FEATURE_EXTRACTOR_TYPE.equals(extractorType)) {
            throw new IllegalArgumentException("unsupported audio feature extractor: " + extractorType);
        }

        return new EmbeddingGemma2AudioConfig(
                text(audio, "model_type"),
                positiveInt(audio, "hidden_size"),
                positiveInt(audio, "num_hidden_layers"),
                positiveInt(audio, "num_attention_heads"),
                positiveInt(audio, "output_proj_dims"),
                positiveInt(audio, "conv_kernel_size"),
                integerList(audio.path("subsampling_conv_channels"), "subsampling_conv_channels"),
                positiveInt(audio, "attention_chunk_size"),
                positiveInt(audio, "attention_context_left"),
                nonNegativeInt(audio, "attention_context_right"),
                positiveDouble(audio, "rms_norm_eps"),
                positiveDouble(audio, "residual_weight"),
                text(audio, "hidden_act"),
                optionalText(audio, "dtype", optionalText(modelConfig, "dtype", "bfloat16")),
                optionalBoolean(audio, "use_clipped_linears", false),
                positiveInt(extractor, "sampling_rate"),
                positiveInt(extractor, "feature_size"),
                positiveInt(extractor, "fft_length"),
                positiveInt(extractor, "frame_length"),
                positiveInt(extractor, "hop_length"),
                nonNegativeDouble(extractor, "min_frequency"),
                positiveDouble(extractor, "max_frequency"),
                positiveDouble(extractor, "mel_floor"),
                number(extractor, "padding_value"),
                positiveDouble(extractor, "input_scale_factor"),
                nonNegativeDouble(extractor, "dither"),
                positiveInt(processorConfig, "audio_seq_length"),
                positiveInt(processorConfig, "audio_ms_per_token"),
                positiveInt(modelConfig, "audio_token_id"),
                positiveInt(modelConfig, "boa_token_id"),
                positiveInt(modelConfig, "eoa_token_index"),
                optionalPositiveDouble(audio, "gradient_clipping", 1.0e10),
                optionalPositiveDouble(audio, "attention_logit_cap", 50.0),
                optionalFiniteDouble(audio, "attention_invalid_logits_value", -1.0e9));
    }

    private static int positiveInt(JsonNode n, String f) {
        int v = requiredInt(n, f);
        if (v <= 0) throw new IllegalArgumentException(f + " must be positive");
        return v;
    }
    private static int nonNegativeInt(JsonNode n, String f) {
        int v = requiredInt(n, f);
        if (v < 0) throw new IllegalArgumentException(f + " must be non-negative");
        return v;
    }
    private static int requiredInt(JsonNode n, String f) {
        JsonNode v=n.path(f);
        if (!v.canConvertToInt()) throw new IllegalArgumentException(f + " must be an integer");
        return v.asInt();
    }
    private static double positiveDouble(JsonNode n, String f) {
        double v=number(n,f);
        positiveFinite(v,f);
        return v;
    }
    private static double optionalPositiveDouble(JsonNode n, String f, double fallback) {
        JsonNode value = n.path(f);
        return value.isMissingNode() || value.isNull() ? fallback : positiveDouble(n, f);
    }
    private static double optionalFiniteDouble(JsonNode n, String f, double fallback) {
        JsonNode value = n.path(f);
        return value.isMissingNode() || value.isNull() ? fallback : number(n, f);
    }
    private static double nonNegativeDouble(JsonNode n, String f) {
        double v=number(n,f);
        nonNegativeFinite(v,f);
        return v;
    }
    private static double number(JsonNode n, String f) {
        JsonNode v=n.path(f);
        if (!v.isNumber() || !Double.isFinite(v.asDouble())) {
            throw new IllegalArgumentException(f + " must be a finite number");
        }
        return v.asDouble();
    }
    private static String text(JsonNode n, String f) {
        JsonNode v=n.path(f);
        if (!v.isTextual() || v.asText().isBlank()) {
            throw new IllegalArgumentException(f + " must be a non-blank string");
        }
        return v.asText();
    }
    private static String optionalText(JsonNode n, String f, String fallback) {
        JsonNode v=n.path(f);
        return v.isMissingNode() || v.isNull() ? fallback : text(n,f);
    }
    private static boolean optionalBoolean(JsonNode n, String f, boolean fallback) {
        JsonNode v=n.path(f);
        if (v.isMissingNode() || v.isNull()) return fallback;
        if (!v.isBoolean()) throw new IllegalArgumentException(f + " must be boolean");
        return v.asBoolean();
    }
    private static List<Integer> integerList(JsonNode n, String f) {
        if (!n.isArray()) throw new IllegalArgumentException(f + " must be an array");
        List<Integer> values=new ArrayList<>();
        n.forEach(v -> {
            if (!v.canConvertToInt() || v.asInt() <= 0) {
                throw new IllegalArgumentException(f + " must contain positive integers");
            }
            values.add(v.asInt());
        });
        return List.copyOf(values);
    }
    private static void positive(int v, String f) { if(v<=0) throw new IllegalArgumentException(f+" must be positive"); }
    private static void nonNegative(int v, String f) { if(v<0) throw new IllegalArgumentException(f+" must be non-negative"); }
    private static void finite(double v, String f) { if(!Double.isFinite(v)) throw new IllegalArgumentException(f+" must be finite"); }
    private static void positiveFinite(double v, String f) { if(!Double.isFinite(v)||v<=0) throw new IllegalArgumentException(f+" must be finite and positive"); }
    private static void nonNegativeFinite(double v, String f) { if(!Double.isFinite(v)||v<0) throw new IllegalArgumentException(f+" must be finite and non-negative"); }
}
