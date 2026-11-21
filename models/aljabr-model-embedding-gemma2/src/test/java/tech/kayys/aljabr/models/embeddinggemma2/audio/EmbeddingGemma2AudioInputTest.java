package tech.kayys.aljabr.models.embeddinggemma2.audio;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingGemma2AudioInputTest {

    @Test
    void averagesInterleavedStereoChannelsToMono() {
        var result = EmbeddingGemma2AudioInput.prepare(
                new float[] {1.0f, -1.0f, 0.5f, 0.5f, -0.25f, 0.25f},
                2, 16_000, 16_000);

        assertArrayEquals(new float[] {0.0f, 0.5f, 0.0f}, result.samples(), 1e-7f);
        assertEquals(16_000, result.sampleRate());
        assertEquals(2, result.sourceChannels());
        assertEquals(3, result.sourceFrameCount());
    }

    @Test
    void resamplesEightKhzPcmToSixteenKhz() {
        float[] input = new float[800];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.sin(2.0 * Math.PI * 440.0 * i / 8_000);
        }

        var result = EmbeddingGemma2AudioInput.prepare(input, 1, 8_000, 16_000);

        assertEquals(1_600, result.sampleCount());
        assertEquals(0.1, result.durationSeconds(), 1e-9);
        for (float sample : result.samples()) {
            assertTrue(Float.isFinite(sample));
        }
        // Away from the boundary, interpolation should preserve a unit-scale tone.
        assertTrue(Math.abs(result.samples()[400]) < 1.1f);
    }

    @Test
    void downsamplingUsesLowPassInterpolationAndExpectedLength() {
        float[] input = new float[1_600];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.sin(2.0 * Math.PI * 440.0 * i / 16_000);
        }

        var result = EmbeddingGemma2AudioInput.prepare(input, 1, 16_000, 8_000);

        assertEquals(800, result.sampleCount());
        assertEquals(0.1, result.durationSeconds(), 1e-9);
        assertTrue(Math.abs(result.samples()[200]) <= 1.01f);
    }

    @Test
    void rejectsIncompleteFramesAndInvalidSamples() {
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2AudioInput.prepare(new float[] {1, 2, 3}, 2, 16_000, 16_000));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2AudioInput.prepare(new float[] {Float.NaN}, 1, 16_000, 16_000));
    }

    @Test
    void resultOwnsItsWaveform() {
        var result = EmbeddingGemma2AudioInput.prepare(new float[] {0.25f}, 1, 16_000, 16_000);
        float[] copy = result.samples();
        copy[0] = 99.0f;
        assertEquals(0.25f, result.samples()[0]);
    }
}
