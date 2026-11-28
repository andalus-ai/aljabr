/*
 * Aljabr — Audio Core
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.audio;

import tech.kayys.aljabr.spi.audio.AudioBuffer;
import tech.kayys.aljabr.spi.audio.AudioFormat;

import java.nio.FloatBuffer;
import java.util.Objects;

/**
 * Portable windowed-sinc resampler for interleaved floating-point PCM.
 *
 * <p>This is a deterministic reference implementation, not a claim of parity
 * with every production codec/DSP resampler. It preserves channel count and
 * applies the same time-domain filter independently to each channel.</p>
 */
public final class WindowedSincPcmResampler {
    private static final int RADIUS = 16;
    private static final Options DEFAULT_OPTIONS = new Options(WindowFunction.HANN, 0.95);

    private WindowedSincPcmResampler() { }

    /** Resamples using the default Hann window and a conservative 0.95 cutoff margin. */
    public static HeapAudioBuffer resample(AudioBuffer input, int targetSampleRate) {
        return resample(input, targetSampleRate, DEFAULT_OPTIONS);
    }

    /**
     * Resamples to {@code targetSampleRate} with explicit window/cutoff semantics.
     * The input remains caller-owned and the returned heap buffer is independently owned.
     */
    public static HeapAudioBuffer resample(AudioBuffer input, int targetSampleRate, Options options) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(options, "options");
        if (targetSampleRate <= 0) {
            throw new IllegalArgumentException("targetSampleRate must be positive");
        }
        AudioFormat sourceFormat = input.format();
        FloatBuffer source = input.samples();
        if (sourceFormat.sampleRate() == targetSampleRate) {
            float[] copy = new float[source.remaining()];
            source.get(copy);
            requireFinite(copy);
            return HeapAudioBuffer.copyOf(sourceFormat, copy);
        }

        int channels = sourceFormat.channels();
        int sourceFrames = Math.toIntExact(input.frameCount());
        long outputFramesLong = Math.round((double) sourceFrames * targetSampleRate / sourceFormat.sampleRate());
        int outputFrames = Math.toIntExact(outputFramesLong);
        if (outputFrames <= 0) {
            throw new IllegalArgumentException("resampled audio would contain no frames");
        }
        float[] output = new float[Math.multiplyExact(outputFrames, channels)];
        double cutoff = Math.min(1.0, (double) targetSampleRate / sourceFormat.sampleRate())
                * options.cutoffScale();

        for (int outFrame = 0; outFrame < outputFrames; outFrame++) {
            double sourcePosition = (double) outFrame * sourceFormat.sampleRate() / targetSampleRate;
            int center = (int) Math.floor(sourcePosition);
            for (int channel = 0; channel < channels; channel++) {
                double weighted = 0.0;
                double weightSum = 0.0;
                for (int tap = center - RADIUS + 1; tap <= center + RADIUS; tap++) {
                    if (tap < 0 || tap >= sourceFrames) {
                        continue;
                    }
                    double distance = sourcePosition - tap;
                    double normalizedDistance = distance / RADIUS;
                    if (Math.abs(normalizedDistance) >= 1.0) {
                        continue;
                    }
                    double window = options.windowFunction().weight(normalizedDistance);
                    double weight = cutoff * sinc(distance * cutoff) * window;
                    float sample = source.get(tap * channels + channel);
                    if (!Float.isFinite(sample)) {
                        throw new IllegalArgumentException("audio samples must be finite");
                    }
                    weighted += sample * weight;
                    weightSum += weight;
                }
                output[outFrame * channels + channel] =
                        Math.abs(weightSum) < 1.0e-12 ? 0.0f : (float) (weighted / weightSum);
            }
        }
        AudioFormat outputFormat = new AudioFormat(targetSampleRate, channels, sourceFormat.channelLayout());
        return HeapAudioBuffer.copyOf(outputFormat, output);
    }

    private static void requireFinite(float[] samples) {
        for (float sample : samples) {
            if (!Float.isFinite(sample)) {
                throw new IllegalArgumentException("audio samples must be finite");
            }
        }
    }

    private static double sinc(double value) {
        if (Math.abs(value) < 1.0e-12) {
            return 1.0;
        }
        double argument = Math.PI * value;
        return Math.sin(argument) / argument;
    }

    /** Named window functions available to model frontends with different reference semantics. */
    public enum WindowFunction {
        HANN {
            @Override
            double weight(double normalizedDistance) {
                return 0.5 + 0.5 * Math.cos(Math.PI * normalizedDistance);
            }
        },
        BLACKMAN {
            @Override
            double weight(double normalizedDistance) {
                return 0.42 + 0.5 * Math.cos(Math.PI * normalizedDistance)
                        + 0.08 * Math.cos(2.0 * Math.PI * normalizedDistance);
            }
        };

        abstract double weight(double normalizedDistance);
    }

    /**
     * DSP parameters are explicit so model frontends can reuse the algorithm
     * without silently changing their published preprocessing semantics.
     *
     * @param windowFunction finite-support window applied to each sinc tap
     * @param cutoffScale multiplier on the anti-aliasing cutoff, in (0, 1]
     */
    public record Options(WindowFunction windowFunction, double cutoffScale) {
        public Options {
            Objects.requireNonNull(windowFunction, "windowFunction");
            if (!Double.isFinite(cutoffScale) || cutoffScale <= 0.0 || cutoffScale > 1.0) {
                throw new IllegalArgumentException("cutoffScale must be finite and in (0, 1]");
            }
        }

        public static Options whisperReference() {
            return new Options(WindowFunction.HANN, 0.95);
        }

        public static Options blackmanReference() {
            return new Options(WindowFunction.BLACKMAN, 1.0);
        }
    }
}
