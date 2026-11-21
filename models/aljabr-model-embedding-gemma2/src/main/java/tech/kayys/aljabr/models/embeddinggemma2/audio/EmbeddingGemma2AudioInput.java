package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.Objects;

/**
 * Pure-Java PCM preparation for the EmbeddingGemma 2 audio frontend.
 *
 * <p>Accepts interleaved floating-point PCM, averages channels to mono, and
 * converts the sample rate with a windowed-sinc low-pass resampler. It does
 * not decode containers (WAV/FLAC/MP3), infer channel layouts, or normalize
 * amplitude; those decisions belong to the caller.</p>
 */
public final class EmbeddingGemma2AudioInput {
    private static final int RESAMPLER_RADIUS = 16;

    private EmbeddingGemma2AudioInput() {
    }

    /**
     * Downmixes interleaved PCM to mono and resamples it to the target rate.
     *
     * @param interleavedPcm samples laid out frame-major, then channel
     * @param channels number of channels in each frame
     * @param sourceSampleRate input sample rate in Hz
     * @param targetSampleRate desired output sample rate in Hz
     * @return mono PCM at targetSampleRate
     */
    public static MonoPcm prepare(
            float[] interleavedPcm, int channels, int sourceSampleRate, int targetSampleRate) {
        Objects.requireNonNull(interleavedPcm, "interleavedPcm must not be null");
        if (channels <= 0) {
            throw new IllegalArgumentException("channels must be positive");
        }
        if (sourceSampleRate <= 0 || targetSampleRate <= 0) {
            throw new IllegalArgumentException("sample rates must be positive");
        }
        if (interleavedPcm.length == 0 || interleavedPcm.length % channels != 0) {
            throw new IllegalArgumentException("PCM must contain complete, non-empty interleaved frames");
        }

        int inputFrames = interleavedPcm.length / channels;
        float[] mono = new float[inputFrames];
        for (int frame = 0; frame < inputFrames; frame++) {
            double sum = 0.0;
            for (int channel = 0; channel < channels; channel++) {
                float sample = interleavedPcm[frame * channels + channel];
                if (!Float.isFinite(sample)) {
                    throw new IllegalArgumentException(
                            "PCM contains a non-finite sample at frame " + frame + ", channel " + channel);
                }
                sum += sample;
            }
            mono[frame] = (float) (sum / channels);
        }

        float[] resampled = sourceSampleRate == targetSampleRate
                ? mono
                : resample(mono, sourceSampleRate, targetSampleRate);
        return new MonoPcm(resampled, targetSampleRate, channels, inputFrames);
    }

    private static float[] resample(float[] input, int sourceRate, int targetRate) {
        int outputLength = Math.toIntExact(Math.round((double) input.length * targetRate / sourceRate));
        if (outputLength <= 0) {
            throw new IllegalArgumentException("resampled waveform is empty");
        }

        float[] output = new float[outputLength];
        double sourcePerOutput = (double) sourceRate / targetRate;
        // Reduce the passband when downsampling to suppress frequencies above
        // the new Nyquist frequency.
        double cutoff = Math.min(1.0, (double) targetRate / sourceRate);
        for (int out = 0; out < outputLength; out++) {
            double center = out * sourcePerOutput;
            int first = (int) Math.floor(center) - RESAMPLER_RADIUS + 1;
            int last = (int) Math.floor(center) + RESAMPLER_RADIUS;
            double weighted = 0.0;
            double weightSum = 0.0;
            for (int index = first; index <= last; index++) {
                if (index < 0 || index >= input.length) {
                    continue;
                }
                double distance = center - index;
                double normalizedDistance = distance / RESAMPLER_RADIUS;
                if (Math.abs(normalizedDistance) >= 1.0) {
                    continue;
                }
                // Blackman-windowed, low-pass sinc. The window is centered
                // around the fractional source position for every output sample.
                double window = 0.42
                        + 0.5 * Math.cos(Math.PI * normalizedDistance)
                        + 0.08 * Math.cos(2.0 * Math.PI * normalizedDistance);
                double weight = cutoff * sinc(cutoff * distance) * window;
                weighted += input[index] * weight;
                weightSum += weight;
            }
            output[out] = (float) (Math.abs(weightSum) < 1e-12 ? 0.0 : weighted / weightSum);
        }
        return output;
    }

    private static double sinc(double value) {
        if (Math.abs(value) < 1e-12) {
            return 1.0;
        }
        double x = Math.PI * value;
        return Math.sin(x) / x;
    }

    /** Immutable mono waveform result. */
    public record MonoPcm(float[] samples, int sampleRate, int sourceChannels, int sourceFrameCount) {
        public MonoPcm {
            Objects.requireNonNull(samples, "samples must not be null");
            if (sampleRate <= 0 || sourceChannels <= 0 || sourceFrameCount <= 0 || samples.length == 0) {
                throw new IllegalArgumentException("invalid mono PCM metadata");
            }
            samples = samples.clone();
        }

        @Override
        public float[] samples() {
            return samples.clone();
        }

        public int sampleCount() {
            return samples.length;
        }

        public double durationSeconds() {
            return (double) samples.length / sampleRate;
        }
    }
}
