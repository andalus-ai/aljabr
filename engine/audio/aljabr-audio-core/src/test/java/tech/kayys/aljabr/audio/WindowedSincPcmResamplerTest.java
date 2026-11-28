/*
 * Aljabr — Audio Core
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.audio;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.audio.AudioFormat;

import static org.junit.jupiter.api.Assertions.*;

class WindowedSincPcmResamplerTest {
    @Test
    void convertsFrameCountAndPreservesChannels() {
        float[] stereo = new float[48_000 * 2];
        for (int frame = 0; frame < 48_000; frame++) {
            stereo[frame * 2] = 0.25f;
            stereo[frame * 2 + 1] = -0.25f;
        }
        try (var input = HeapAudioBuffer.copyOf(AudioFormat.stereo(48_000), stereo);
             var output = WindowedSincPcmResampler.resample(input, 16_000)) {
            assertEquals(AudioFormat.stereo(16_000), output.format());
            assertEquals(16_000, output.frameCount());
            var samples = output.samples();
            assertEquals(0.25f, samples.get(100 * 2), 0.01f);
            assertEquals(-0.25f, samples.get(100 * 2 + 1), 0.01f);
        }
    }

    @Test
    void blackmanOptionsRetainPublishedReferenceSamples() {
        float[] mono = new float[32];
        for (int index = 0; index < mono.length; index++) {
            mono[index] = (float) (0.2 * Math.sin(2.0 * Math.PI * 440.0 * index / 8_000));
        }
        try (var input = HeapAudioBuffer.copyOf(AudioFormat.mono(8_000), mono);
             var output = WindowedSincPcmResampler.resample(input, 16_000,
                     WindowedSincPcmResampler.Options.blackmanReference())) {
            var samples = output.samples();
            assertEquals(64, samples.remaining());
            assertEquals(0.024800611f, samples.get(1), 1.0e-6f);
            assertEquals(0.19418792f, samples.get(7), 1.0e-6f);
            assertEquals(0.19753767f, samples.get(10), 1.0e-6f);
            assertEquals(-0.15993640f, samples.get(31), 1.0e-6f);
        }
    }

    @Test
    void sameRateResamplingReturnsIndependentCopy() {
        try (var input = HeapAudioBuffer.copyOf(AudioFormat.mono(16_000), new float[]{0.1f, 0.2f});
             var output = WindowedSincPcmResampler.resample(input, 16_000)) {
            assertNotSame(input, output);
            assertArrayEquals(new float[]{0.1f, 0.2f}, read(output));
        }
    }

    @Test
    void validatesTargetRateAndOptions() {
        try (var input = HeapAudioBuffer.copyOf(AudioFormat.mono(16_000), new float[]{0.1f})) {
            assertThrows(IllegalArgumentException.class,
                    () -> WindowedSincPcmResampler.resample(input, 0));
        }
        assertThrows(IllegalArgumentException.class,
                () -> new WindowedSincPcmResampler.Options(
                        WindowedSincPcmResampler.WindowFunction.HANN, 1.01));
    }

    private static float[] read(HeapAudioBuffer buffer) {
        var samples = buffer.samples();
        float[] values = new float[samples.remaining()];
        samples.get(values);
        return values;
    }
}
