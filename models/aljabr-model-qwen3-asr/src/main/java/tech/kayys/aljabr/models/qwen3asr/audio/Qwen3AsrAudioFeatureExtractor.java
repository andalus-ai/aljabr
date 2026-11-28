package tech.kayys.aljabr.models.qwen3asr.audio;

import tech.kayys.aljabr.models.whisper.WhisperAudioPreprocessor;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Qwen3-ASR input-contract adapter over Aljabr's existing Whisper-compatible
 * frontend. It owns Qwen processor validation/chunk metadata, not FFT/Mel DSP.
 * Input is decoded interleaved float PCM; encoded-media decoding stays external.
 */
public final class Qwen3AsrAudioFeatureExtractor {
    public static final int SAMPLE_RATE = 16_000;
    public static final int FEATURE_SIZE = 128;
    public static final int FFT_SIZE = 400;
    public static final int HOP_LENGTH = 160;
    public static final int CHUNK_LENGTH_SECONDS = 30;
    public static final int CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH_SECONDS;
    public static final int MAX_FRAMES = 3_000;
    public static final int MIN_SAMPLES = 8_000;

    private final WhisperAudioPreprocessor delegate;

    public Qwen3AsrAudioFeatureExtractor() { this(new WhisperAudioPreprocessor()); }

    public Qwen3AsrAudioFeatureExtractor(WhisperAudioPreprocessor delegate) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
    }

    /** Converts decoded PCM to right-padded 30-second Qwen3-ASR feature chunks. */
    public List<AudioFeatures> extract(float[] interleavedPcm, int inputSampleRate, int channels) {
        Objects.requireNonNull(interleavedPcm, "interleavedPcm");
        if (inputSampleRate <= 0) throw new IllegalArgumentException("inputSampleRate must be positive");
        if (channels <= 0) throw new IllegalArgumentException("channels must be positive");
        if (interleavedPcm.length == 0) throw new IllegalArgumentException("audio must not be empty");
        if (interleavedPcm.length % channels != 0) {
            throw new IllegalArgumentException("interleaved sample count must be divisible by channels");
        }
        for (int i = 0; i < interleavedPcm.length; i++) {
            if (!Float.isFinite(interleavedPcm[i])) {
                throw new IllegalArgumentException("audio contains a non-finite sample at index " + i);
            }
        }
        long inputFrames = interleavedPcm.length / channels;
        long targetFrames = Math.multiplyExact(inputFrames, SAMPLE_RATE) / inputSampleRate;
        if (targetFrames < MIN_SAMPLES) {
            throw new IllegalArgumentException("Qwen3-ASR requires at least 8000 samples at 16000 Hz (500 ms)");
        }

        List<WhisperAudioPreprocessor.AudioChunk> chunks =
                delegate.prepare(interleavedPcm, inputSampleRate, channels);
        if (chunks.isEmpty()) throw new IllegalArgumentException("audio frontend produced no feature chunks");

        List<AudioFeatures> result = new ArrayList<>(chunks.size());
        for (WhisperAudioPreprocessor.AudioChunk chunk : chunks) {
            result.add(new AudioFeatures(chunk.index(), chunk.startSample(),
                    chunk.sourceSampleCount(), chunk.logMelFeatures()));
        }
        return List.copyOf(result);
    }

    /** One 30-second feature window, laid out [melBin][frame]. */
    public record AudioFeatures(int chunkIndex, long startSample, int sourceSampleCount,
                                float[][] inputFeatures) {
        public AudioFeatures {
            if (chunkIndex < 0 || startSample < 0 || sourceSampleCount <= 0) {
                throw new IllegalArgumentException("invalid chunk metadata");
            }
            Objects.requireNonNull(inputFeatures, "inputFeatures");
            if (inputFeatures.length != FEATURE_SIZE) {
                throw new IllegalArgumentException("expected 128 Mel bins");
            }
            float[][] copy = new float[inputFeatures.length][];
            for (int i = 0; i < inputFeatures.length; i++) {
                if (inputFeatures[i] == null || inputFeatures[i].length != MAX_FRAMES) {
                    throw new IllegalArgumentException("expected 3000 frames per Mel bin");
                }
                copy[i] = inputFeatures[i].clone();
            }
            inputFeatures = copy;
        }
        @Override public float[][] inputFeatures() {
            float[][] copy = new float[inputFeatures.length][];
            for (int i = 0; i < inputFeatures.length; i++) copy[i] = inputFeatures[i].clone();
            return copy;
        }
    }
}
