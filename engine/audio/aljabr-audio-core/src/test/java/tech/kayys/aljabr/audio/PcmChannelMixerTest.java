/*
 * Aljabr — Audio Core
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.audio;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.audio.AudioFormat;

import static org.junit.jupiter.api.Assertions.*;

class PcmChannelMixerTest {
    @Test
    void averagesInterleavedChannelsWithoutMutatingInput() {
        try (var input = HeapAudioBuffer.copyOf(AudioFormat.stereo(48_000),
                    new float[]{0.25f, 0.75f, -0.5f, 0.5f});
             var mono = PcmChannelMixer.downmixToMono(input)) {
            assertEquals(AudioFormat.mono(48_000), mono.format());
            assertEquals(2, mono.frameCount());
            assertArrayEquals(new float[]{0.5f, 0.0f}, read(mono));
            assertArrayEquals(new float[]{0.25f, 0.75f, -0.5f, 0.5f}, read(input));
        }
    }

    @Test
    void rejectsNonFiniteSamples() {
        try (var input = HeapAudioBuffer.copyOf(AudioFormat.stereo(48_000),
                new float[]{Float.NaN, 0.0f})) {
            assertThrows(IllegalArgumentException.class, () -> PcmChannelMixer.downmixToMono(input));
        }
    }

    private static float[] read(HeapAudioBuffer buffer) {
        var samples = buffer.samples();
        float[] values = new float[samples.remaining()];
        samples.get(values);
        return values;
    }
}
