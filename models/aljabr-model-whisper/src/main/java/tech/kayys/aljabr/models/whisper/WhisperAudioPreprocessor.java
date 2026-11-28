package tech.kayys.aljabr.models.whisper;

import tech.kayys.aljabr.audio.PcmChannelMixer;
import tech.kayys.aljabr.audio.WindowedSincPcmResampler;
import tech.kayys.aljabr.audio.HeapAudioBuffer;
import tech.kayys.aljabr.spi.audio.AudioFormat;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Whisper-specific frontend: canonical mono sampling, 30-second windowing,
 * and large-v3 log-Mel feature extraction.
 *
 * <p>Generic channel mixing and sample-rate conversion are delegated to the
 * reusable audio engine. Whisper's exact feature geometry remains model-owned.</p>
 */
public final class WhisperAudioPreprocessor {
    public static final int TARGET_SAMPLE_RATE = WhisperLargeV3Config.EXPECTED_SAMPLE_RATE;

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

        AudioFormat inputFormat = new AudioFormat(sampleRate, channels, AudioFormat.ChannelLayout.UNKNOWN);
        try (HeapAudioBuffer input = HeapAudioBuffer.copyOf(inputFormat, interleavedPcm);
             HeapAudioBuffer mono = PcmChannelMixer.downmixToMono(input);
             HeapAudioBuffer normalized = sampleRate == TARGET_SAMPLE_RATE
                     ? HeapAudioBuffer.copyOf(mono.format(), readSamples(mono))
                     : WindowedSincPcmResampler.resample(mono, TARGET_SAMPLE_RATE)) {
            float[] monoSamples = readSamples(normalized);
            List<AudioChunk> chunks = new ArrayList<>();
            for (int offset = 0, chunkIndex = 0; offset < monoSamples.length;
                    offset += WhisperLogMelSpectrogram.CHUNK_SAMPLES, chunkIndex++) {
                int length = Math.min(WhisperLogMelSpectrogram.CHUNK_SAMPLES, monoSamples.length - offset);
                float[] chunkSamples = new float[length];
                System.arraycopy(monoSamples, offset, chunkSamples, 0, length);
                float[][] features = spectrogram.extract(chunkSamples);
                chunks.add(new AudioChunk(chunkIndex, offset, length, features));
            }
            return List.copyOf(chunks);
        }
    }

    private static float[] readSamples(HeapAudioBuffer audio) {
        var samples = audio.samples();
        float[] result = new float[samples.remaining()];
        samples.get(result);
        return result;
    }

    /** A single feature window; feature matrix is [melBin][frame]. */
    public record AudioChunk(int index, long startSample, int sourceSampleCount, float[][] logMelFeatures) {
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
}
