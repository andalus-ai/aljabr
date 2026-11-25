package tech.kayys.aljabr.models.whisper;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Reusable PCM-to-Whisper frontend for mono conversion, resampling, and
 * large-v3 log-Mel chunk preparation.
 *
 * <p>Input samples are interleaved floating-point PCM in the normalized
 * [-1, 1] range. Downmixing averages channels; resampling uses a windowed-sinc
 * low-pass interpolator to avoid aliasing when reducing sample rates.</p>
 */
public final class WhisperAudioPreprocessor {
    public static final int TARGET_SAMPLE_RATE = WhisperLargeV3Config.EXPECTED_SAMPLE_RATE;
    private static final int RESAMPLER_RADIUS = 16;

    private final WhisperLogMelSpectrogram spectrogram;

    public WhisperAudioPreprocessor() {
        this(new WhisperLogMelSpectrogram());
    }

    public WhisperAudioPreprocessor(WhisperLogMelSpectrogram spectrogram) {
        this.spectrogram = Objects.requireNonNull(spectrogram, "spectrogram must not be null");
    }

    /**
     * Converts interleaved PCM to mono 16 kHz and extracts consecutive
     * 30-second Whisper feature windows. The final window is zero-padded.
     */
    public List<AudioChunk> prepare(float[] interleavedPcm, int sampleRate, int channels) {
        Objects.requireNonNull(interleavedPcm, "interleavedPcm must not be null");
        if (sampleRate <= 0) {
            throw new IllegalArgumentException("sampleRate must be positive");
        }
        if (channels <= 0) {
            throw new IllegalArgumentException("channels must be positive");
        }
        if (interleavedPcm.length == 0) {
            return List.of();
        }
        if (interleavedPcm.length % channels != 0) {
            throw new IllegalArgumentException("interleaved PCM length must be divisible by channel count");
        }

        float[] mono = downmix(interleavedPcm, channels);
        float[] normalized = sampleRate == TARGET_SAMPLE_RATE
                ? mono : resample(mono, sampleRate, TARGET_SAMPLE_RATE);

        List<AudioChunk> chunks = new ArrayList<>();
        for (int offset = 0, chunkIndex = 0; offset < normalized.length;
                offset += WhisperLogMelSpectrogram.CHUNK_SAMPLES, chunkIndex++) {
            int length = Math.min(WhisperLogMelSpectrogram.CHUNK_SAMPLES, normalized.length - offset);
            float[] chunkSamples = new float[length];
            System.arraycopy(normalized, offset, chunkSamples, 0, length);
            float[][] features = spectrogram.extract(chunkSamples);
            chunks.add(new AudioChunk(chunkIndex, offset, length, features));
        }
        return List.copyOf(chunks);
    }

    /** A single feature window; feature matrix is [melBin][frame]. */
    public record AudioChunk(
            int index,
            long startSample,
            int sourceSampleCount,
            float[][] logMelFeatures) {
        public AudioChunk {
            if (index < 0 || startSample < 0 || sourceSampleCount <= 0) {
                throw new IllegalArgumentException("audio chunk metadata is invalid");
            }
            Objects.requireNonNull(logMelFeatures, "logMelFeatures must not be null");
            if (logMelFeatures.length != WhisperLogMelSpectrogram.MEL_BINS) {
                throw new IllegalArgumentException("expected 128 Mel feature bins");
            }
            for (float[] row : logMelFeatures) {
                if (row == null || row.length != WhisperLogMelSpectrogram.FRAME_COUNT) {
                    throw new IllegalArgumentException("expected 3000 frames for every Mel feature bin");
                }
            }
        }
    }

    private static float[] downmix(float[] interleaved, int channels) {
        int frames = interleaved.length / channels;
        float[] mono = new float[frames];
        for (int frame = 0; frame < frames; frame++) {
            double sum = 0.0;
            for (int channel = 0; channel < channels; channel++) {
                float sample = interleaved[frame * channels + channel];
                if (!Float.isFinite(sample)) {
                    throw new IllegalArgumentException("audio samples must be finite");
                }
                sum += Math.max(-1.0f, Math.min(1.0f, sample));
            }
            mono[frame] = (float) (sum / channels);
        }
        return mono;
    }

    private static float[] resample(float[] input, int sourceRate, int targetRate) {
        int outputLength = Math.toIntExact(Math.round((double) input.length * targetRate / sourceRate));
        float[] output = new float[outputLength];
        double cutoff = Math.min(1.0, (double) targetRate / sourceRate) * 0.95;

        for (int out = 0; out < outputLength; out++) {
            double sourcePosition = (double) out * sourceRate / targetRate;
            int center = (int) Math.floor(sourcePosition);
            double weighted = 0.0;
            double weightSum = 0.0;
            for (int tap = center - RESAMPLER_RADIUS + 1;
                    tap <= center + RESAMPLER_RADIUS; tap++) {
                if (tap < 0 || tap >= input.length) {
                    continue;
                }
                double distance = sourcePosition - tap;
                double normalizedDistance = distance / RESAMPLER_RADIUS;
                if (Math.abs(normalizedDistance) >= 1.0) {
                    continue;
                }
                double window = 0.5 + 0.5 * Math.cos(Math.PI * normalizedDistance);
                double weight = cutoff * sinc(distance * cutoff) * window;
                weighted += input[tap] * weight;
                weightSum += weight;
            }
            output[out] = weightSum == 0.0 ? 0.0f : (float) (weighted / weightSum);
        }
        return output;
    }

    private static double sinc(double value) {
        if (Math.abs(value) < 1.0e-12) {
            return 1.0;
        }
        double argument = Math.PI * value;
        return Math.sin(argument) / argument;
    }
}
