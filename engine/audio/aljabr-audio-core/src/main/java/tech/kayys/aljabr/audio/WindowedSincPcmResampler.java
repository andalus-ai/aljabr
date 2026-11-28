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
 * Portable windowed-sinc resampler for interleaved normalized floating-point PCM.
 *
 * <p>This is a deterministic reference implementation, not a claim of parity with
 * every production codec/DSP resampler. It preserves channel count and applies
 * the same time-domain filter independently to each channel.</p>
 */
public final class WindowedSincPcmResampler {
    private static final int RADIUS = 16;

    private WindowedSincPcmResampler() { }

    /** Resamples to {@code targetSampleRate}; the input remains caller-owned. */
    public static HeapAudioBuffer resample(AudioBuffer input, int targetSampleRate) {
        Objects.requireNonNull(input, "input");
        if (targetSampleRate <= 0) {
            throw new IllegalArgumentException("targetSampleRate must be positive");
        }
        AudioFormat sourceFormat = input.format();
        if (sourceFormat.sampleRate() == targetSampleRate) {
            FloatBuffer source = input.samples();
            float[] copy = new float[source.remaining()];
            source.get(copy);
            return HeapAudioBuffer.copyOf(sourceFormat, copy);
        }

        int channels = sourceFormat.channels();
        int sourceFrames = Math.toIntExact(input.frameCount());
        long outputFramesLong = Math.round((double) sourceFrames * targetSampleRate / sourceFormat.sampleRate());
        int outputFrames = Math.toIntExact(outputFramesLong);
        float[] output = new float[Math.multiplyExact(outputFrames, channels)];
        FloatBuffer source = input.samples();
        double cutoff = Math.min(1.0, (double) targetSampleRate / sourceFormat.sampleRate()) * 0.95;

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
                    double window = 0.5 + 0.5 * Math.cos(Math.PI * normalizedDistance);
                    double weight = cutoff * sinc(distance * cutoff) * window;
                    float sample = source.get(tap * channels + channel);
                    if (!Float.isFinite(sample)) {
                        throw new IllegalArgumentException("audio samples must be finite");
                    }
                    weighted += sample * weight;
                    weightSum += weight;
                }
                output[outFrame * channels + channel] =
                        weightSum == 0.0 ? 0.0f : (float) (weighted / weightSum);
            }
        }
        AudioFormat outputFormat = new AudioFormat(targetSampleRate, channels, sourceFormat.channelLayout());
        return HeapAudioBuffer.copyOf(outputFormat, output);
    }

    private static double sinc(double value) {
        if (Math.abs(value) < 1.0e-12) {
            return 1.0;
        }
        double argument = Math.PI * value;
        return Math.sin(argument) / argument;
    }
}
