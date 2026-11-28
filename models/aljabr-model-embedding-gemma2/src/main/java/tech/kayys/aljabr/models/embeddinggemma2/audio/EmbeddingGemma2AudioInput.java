package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.Objects;
import tech.kayys.aljabr.audio.HeapAudioBuffer;
import tech.kayys.aljabr.audio.PcmChannelMixer;
import tech.kayys.aljabr.audio.WindowedSincPcmResampler;
import tech.kayys.aljabr.spi.audio.AudioFormat;

/**
 * EmbeddingGemma 2 PCM adapter over Aljabr's reusable audio engine.
 *
 * <p>It retains the model's published Blackman-windowed sinc parameters and
 * amplitude-preserving channel average. Container decoding, channel-layout
 * interpretation, and amplitude normalization remain caller responsibilities.</p>
 */
public final class EmbeddingGemma2AudioInput {
    private static final WindowedSincPcmResampler.Options RESAMPLER_OPTIONS =
            WindowedSincPcmResampler.Options.blackmanReference();

    private EmbeddingGemma2AudioInput() { }

    /**
     * Downmixes interleaved PCM to mono and resamples it to the target rate.
     * The source array is copied into an owned AudioBuffer and is never mutated.
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
        AudioFormat inputFormat = new AudioFormat(sourceSampleRate, channels, AudioFormat.ChannelLayout.UNKNOWN);
        try (HeapAudioBuffer input = HeapAudioBuffer.copyOf(inputFormat, interleavedPcm);
             HeapAudioBuffer mono = PcmChannelMixer.downmixToMono(
                     input, PcmChannelMixer.ClippingPolicy.PRESERVE);
             HeapAudioBuffer resampled = sourceSampleRate == targetSampleRate
                     ? HeapAudioBuffer.copyOf(mono.format(), readSamples(mono))
                     : WindowedSincPcmResampler.resample(mono, targetSampleRate, RESAMPLER_OPTIONS)) {
            float[] samples = readSamples(resampled);
            return new MonoPcm(samples, targetSampleRate, channels, inputFrames);
        }
    }

    private static float[] readSamples(HeapAudioBuffer audio) {
        var samples = audio.samples();
        float[] result = new float[samples.remaining()];
        samples.get(result);
        return result;
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
