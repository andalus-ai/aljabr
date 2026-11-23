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
     * Counts valid audio soft tokens using the same two stride-2 mask reductions
     * as the audio tower. The processor sequence-length setting is the upper bound.
     */
    public int audioPlaceholderCount(ProcessedAudio audio) {
        Objects.requireNonNull(audio, "audio must not be null");
        boolean[] mask = audio.features().inputFeaturesMask();
        for (int layer = 0; layer < 2; layer++) {
            boolean[] downsampled = new boolean[(mask.length + 1) / 2];
            for (int index = 0; index < downsampled.length; index++) {
                downsampled[index] = mask[index * 2];
            }
            mask = downsampled;
        }
        int validTokens = 0;
        for (boolean valid : mask) if (valid) validTokens++;
        return Math.min(validTokens, config.audioSequenceLength());
    }

    /** Expands exactly one audio marker into the number of soft-token slots for this waveform. */
    public int[] expandAudioPlaceholders(int[] inputIds, ProcessedAudio audio) {
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        Objects.requireNonNull(audio, "audio must not be null");
        if (inputIds.length == 0) throw new IllegalArgumentException("inputIds must not be empty");
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
        int[] expanded = new int[Math.addExact(inputIds.length - 1, expandedCount)];
        System.arraycopy(inputIds, 0, expanded, 0, markerIndex);
        java.util.Arrays.fill(expanded, markerIndex, markerIndex + expandedCount, config.audioTokenId());
        System.arraycopy(inputIds, markerIndex + 1, expanded, markerIndex + expandedCount,
                inputIds.length - markerIndex - 1);
        return expanded;
    }

    public int maxInputSamples() {
        return maxInputSamples;
    }

    private static int maxSamplesForTokenBudget(EmbeddingGemma2AudioConfig config) {
        long samples = Math.multiplyExact(
                Math.multiplyExact((long) config.audioSequenceLength(), 4L),
                (long) config.hopLength());
        return (int) Math.min(Integer.MAX_VALUE, samples);
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
