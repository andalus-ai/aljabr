/*
 * Aljabr — Audio Core
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.aljabr.audio;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.audio.AudioFormat;

import static org.junit.jupiter.api.Assertions.*;

class HeapAudioBufferTest {
    @Test
    void copiesInputAndExposesReadOnlyInterleavedSamples() {
        float[] input = {0.1f, 0.2f, 0.3f, 0.4f};
        try (var audio = HeapAudioBuffer.copyOf(AudioFormat.stereo(16_000), input)) {
            input[0] = 9.0f;
            assertEquals(2, audio.frameCount());
            var samples = audio.samples();
            assertTrue(samples.isReadOnly());
            assertEquals(0.1f, samples.get(0), 0.00001f);
            assertThrows(java.nio.ReadOnlyBufferException.class, () -> samples.put(0, 1.0f));
        }
    }

    @Test
    void rejectsMisalignedSampleCount() {
        assertThrows(IllegalArgumentException.class,
                () -> HeapAudioBuffer.copyOf(AudioFormat.stereo(16_000), new float[]{0.1f}));
    }

    @Test
    void rejectsUseAfterClose() {
        var audio = HeapAudioBuffer.copyOf(AudioFormat.mono(16_000), new float[]{0.1f});
        audio.close();
        assertTrue(audio.isClosed());
        assertThrows(IllegalStateException.class, audio::samples);
    }
}
