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
        this.maxInputSamples = maxInputSamples;
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
