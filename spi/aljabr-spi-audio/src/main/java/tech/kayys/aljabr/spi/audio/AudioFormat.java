/*
 * Aljabr — Audio SPI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.spi.audio;

/**
 * Describes the logical PCM format exposed by an {@link AudioBuffer}.
 *
 * <p>Samples in an AudioBuffer are normalized floating-point values in
 * interleaved frame order, regardless of the encoding of the original file.</p>
 *
 * @param sampleRate sample rate in frames per second
 * @param channels number of interleaved channels
 * @param channelLayout semantic channel layout
 */
public record AudioFormat(int sampleRate, int channels, ChannelLayout channelLayout) {
    public AudioFormat {
        if (sampleRate <= 0) {
            throw new IllegalArgumentException("sampleRate must be positive");
        }
        if (channels <= 0) {
            throw new IllegalArgumentException("channels must be positive");
        }
        if (channelLayout == null) {
            throw new IllegalArgumentException("channelLayout must not be null");
        }
        if (channelLayout.channelCount() != 0 && channelLayout.channelCount() != channels) {
            throw new IllegalArgumentException(
                    "channelLayout " + channelLayout + " requires " + channelLayout.channelCount()
                            + " channels, got " + channels);
        }
    }

    /** Common layouts. UNKNOWN permits application-defined channel arrangements. */
    public enum ChannelLayout {
        MONO(1),
        STEREO(2),
        QUAD(4),
        FIVE_POINT_ONE(6),
        SEVEN_POINT_ONE(8),
        UNKNOWN(0);

        private final int channelCount;

        ChannelLayout(int channelCount) {
            this.channelCount = channelCount;
        }

        public int channelCount() {
            return channelCount;
        }
    }

    public static AudioFormat mono(int sampleRate) {
        return new AudioFormat(sampleRate, 1, ChannelLayout.MONO);
    }

    public static AudioFormat stereo(int sampleRate) {
        return new AudioFormat(sampleRate, 2, ChannelLayout.STEREO);
    }
}
