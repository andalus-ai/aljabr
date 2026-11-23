package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

/**
 * End-to-end waveform-to-log-mel preprocessing for EmbeddingGemma 2.
 *
 * <p>The processor composes channel downmixing, sample-rate conversion, and
 * the model's log-mel feature extractor. It accepts decoded interleaved
 * floating-point PCM; file/container decoding remains the responsibility of
 * an I/O adapter. The published processor truncates waveforms to 480,000
 * samples (30 seconds at 16 kHz) before feature extraction.</p>
 */
public final class EmbeddingGemma2AudioProcessor {
    public static final int PUBLISHED_MAX_INPUT_SAMPLES = 480_000;

    private final EmbeddingGemma2AudioConfig config;
    private final int maxInputSamples;
    private final EmbeddingGemma2AudioFeatureExtractor featureExtractor;

    public EmbeddingGemma2AudioProcessor(EmbeddingGemma2AudioConfig config) {
        this(config, PUBLISHED_MAX_INPUT_SAMPLES);
    }

    public EmbeddingGemma2AudioProcessor(
            EmbeddingGemma2AudioConfig config, int maxInputSamples) {
        this.config = Objects.requireNonNull(config, "config must not be null");
        if (maxInputSamples <= 0) {
            throw new IllegalArgumentException("maxInputSamples must be positive");
        }
        if (config.dither() != 0.0) {
            throw new IllegalArgumentException(
                    "non-zero dither is not supported by the deterministic Java audio processor");
        }
        if (config.paddingValue() != 0.0) {
            throw new IllegalArgumentException(
                    "only zero waveform padding is currently supported");
        }
        this.maxInputSamples = Math.min(maxInputSamples, maxSamplesForTokenBudget(config));
        this.featureExtractor = new EmbeddingGemma2AudioFeatureExtractor(
                config.samplingRate(),
                config.featureSize(),
                config.fftLength(),
                config.frameLength(),
                config.hopLength(),
                config.minFrequency(),
                config.maxFrequency(),
                config.melFloor(),
                config.inputScaleFactor());
    }

    /**
     * Processes interleaved decoded PCM into [frames][melFeatures] log-mel
     * values and a frame-validity mask.
     */
    public ProcessedAudio process(float[] interleavedPcm, int channels, int sourceSampleRate) {
        EmbeddingGemma2AudioInput.MonoPcm mono = EmbeddingGemma2AudioInput.prepare(
                interleavedPcm, channels, sourceSampleRate, config.samplingRate());
        float[] samples = mono.samples();
        boolean truncated = samples.length > maxInputSamples;
        if (truncated) {
            samples = java.util.Arrays.copyOf(samples, maxInputSamples);
        }
        EmbeddingGemma2AudioFeatureExtractor.Features features =
                featureExtractor.extract(samples, config.samplingRate());
        return new ProcessedAudio(
                features,
                config.samplingRate(),
                channels,
                sourceSampleRate,
                samples.length,
                truncated);
    }

    /**
     * Computes the number of audio soft-token slots from the feature mask.
     * The audio tower downsamples its time mask twice by taking every second
     * element, so this count is exactly the number of valid tower positions.
     */
    public int audioPlaceholderCount(ProcessedAudio audio) {
        Objects.requireNonNull(audio, "audio must not be null");
        boolean[] frameMask = audio.features().inputFeaturesMask();
        if (frameMask == null || frameMask.length == 0) {
            throw new IllegalArgumentException("audio feature mask must contain at least one frame");
        }
        boolean[] downsampled = downsampleMask(downsampleMask(frameMask));
        int validTokens = 0;
        for (boolean valid : downsampled) {
            if (valid) validTokens++;
        }
        if (validTokens == 0) {
            throw new IllegalArgumentException("audio input produced no valid soft-token slots");
        }
        return Math.min(config.audioSequenceLength(), validTokens);
    }

    private static boolean[] downsampleMask(boolean[] mask) {
        boolean[] downsampled = new boolean[(mask.length + 1) / 2];
        for (int index = 0; index < downsampled.length; index++) {
            downsampled[index] = mask[index * 2];
        }
        return downsampled;
    }

    /**
     * Replaces exactly one audio marker in a prompt with the configured number
     * of repeated audio placeholder IDs. Tokenization and chat-template policy
     * remain outside this processor.
     */
    public int[] expandAudioPlaceholders(int[] inputIds, ProcessedAudio audio) {
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        Objects.requireNonNull(audio, "audio must not be null");
        if (inputIds.length == 0) {
            throw new IllegalArgumentException("inputIds must not be empty");
        }
        int markerCount = 0;
        int markerIndex = -1;
        for (int index = 0; index < inputIds.length; index++) {
            if (inputIds[index] == config.audioTokenId()) {
                markerCount++;
                markerIndex = index;
            }
        }
        if (markerCount != 1) {
            throw new IllegalArgumentException(
                    "exactly one audio placeholder is required for one waveform; found " + markerCount);
        }
        int expandedCount = audioPlaceholderCount(audio);
        int replacementLength = Math.addExact(expandedCount, 2);
        int outputLength = Math.addExact(inputIds.length - 1, replacementLength);
        int[] expanded = new int[outputLength];
        System.arraycopy(inputIds, 0, expanded, 0, markerIndex);
        expanded[markerIndex] = config.beginAudioTokenId();
        java.util.Arrays.fill(expanded, markerIndex + 1, markerIndex + 1 + expandedCount,
                config.audioTokenId());
        expanded[markerIndex + 1 + expandedCount] = config.endAudioTokenId();
        System.arraycopy(inputIds, markerIndex + 1, expanded, markerIndex + replacementLength,
                inputIds.length - markerIndex - 1);
        return expanded;
    }

    /**
     * Returns the sample limit implied by the configured audio soft-token budget.
     * One soft-token slot represents {@code audioMillisecondsPerToken} milliseconds.
     * The published 30-second processor limit remains an independent upper bound.
     */
    private static int maxSamplesForTokenBudget(EmbeddingGemma2AudioConfig config) {
        long budgetMillis = Math.multiplyExact(
                (long) config.audioSequenceLength(), (long) config.audioMillisecondsPerToken());
        long budgetSamples = Math.multiplyExact(budgetMillis, (long) config.samplingRate()) / 1_000L;
        long boundedSamples = Math.min((long) PUBLISHED_MAX_INPUT_SAMPLES, budgetSamples);
        if (boundedSamples <= 0L) {
            throw new IllegalArgumentException("audio token budget must allow at least one input sample");
        }
        return Math.toIntExact(boundedSamples);
    }

    public EmbeddingGemma2AudioConfig config() {
        return config;
    }

    /** Immutable result; feature arrays remain defensively copied by Features. */
    public record ProcessedAudio(
            EmbeddingGemma2AudioFeatureExtractor.Features features,
            int sampleRate,
            int sourceChannels,
            int sourceSampleRate,
            int processedSampleCount,
            boolean truncated) {
        public ProcessedAudio {
            Objects.requireNonNull(features, "features must not be null");
            if (sampleRate <= 0 || sourceChannels <= 0 || sourceSampleRate <= 0
                    || processedSampleCount <= 0) {
                throw new IllegalArgumentException("invalid processed audio metadata");
            }
        }

        public double durationSeconds() {
            return (double) processedSampleCount / sampleRate;
        }
    }
}
