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

/** Stateless channel-layout conversion primitives for interleaved floating-point PCM. */
public final class PcmChannelMixer {
    private PcmChannelMixer() { }

    /** Default policy for normalized PCM: average channels and clamp each input sample to [-1, 1]. */
    public static HeapAudioBuffer downmixToMono(AudioBuffer input) {
        return downmixToMono(input, ClippingPolicy.CLAMP_TO_UNIT_RANGE);
    }

    /**
     * Averages all input channels into mono using the requested clipping policy.
     * The input remains caller-owned; the returned heap buffer is independently owned.
     */
    public static HeapAudioBuffer downmixToMono(AudioBuffer input, ClippingPolicy clippingPolicy) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(clippingPolicy, "clippingPolicy");
        AudioFormat format = input.format();
        FloatBuffer samples = input.samples();
        int channels = format.channels();
        int frames = Math.toIntExact(input.frameCount());
        float[] mono = new float[frames];
        for (int frame = 0; frame < frames; frame++) {
            double sum = 0.0;
            for (int channel = 0; channel < channels; channel++) {
                float sample = samples.get(frame * channels + channel);
                if (!Float.isFinite(sample)) {
                    throw new IllegalArgumentException("audio samples must be finite");
                }
                if (clippingPolicy == ClippingPolicy.CLAMP_TO_UNIT_RANGE) {
                    sample = Math.max(-1.0f, Math.min(1.0f, sample));
                }
                sum += sample;
            }
            mono[frame] = (float) (sum / channels);
        }
        return HeapAudioBuffer.copyOf(AudioFormat.mono(format.sampleRate()), mono);
    }

    /** Explicit amplitude treatment when mixing source channels. */
    public enum ClippingPolicy {
        /** Preserve source sample values; useful when the model frontend owns amplitude scaling. */
        PRESERVE,
        /** Clamp each source sample to the conventional normalized PCM interval. */
        CLAMP_TO_UNIT_RANGE
    }
}
