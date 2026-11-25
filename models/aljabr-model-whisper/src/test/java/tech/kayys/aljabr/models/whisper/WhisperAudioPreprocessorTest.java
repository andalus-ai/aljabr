package tech.kayys.aljabr.models.whisper;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class WhisperAudioPreprocessorTest {

    private final WhisperAudioPreprocessor preprocessor = new WhisperAudioPreprocessor();

    @Test
    void createsLargeV3FeatureShapeForMono16KhzAudio() {
        float[] audio = new float[16_000];
        audio[100] = 0.75f;

        var chunks = preprocessor.prepare(audio, 16_000, 1);

        assertEquals(1, chunks.size());
        assertEquals(0, chunks.get(0).index());
        assertEquals(0, chunks.get(0).startSample());
        assertEquals(16_000, chunks.get(0).sourceSampleCount());
        assertEquals(128, chunks.get(0).logMelFeatures().length);
        assertEquals(3_000, chunks.get(0).logMelFeatures()[0].length);
        for (float[] row : chunks.get(0).logMelFeatures()) {
            for (float value : row) {
                assertTrue(Float.isFinite(value));
            }
        }
    }

    @Test
    void downmixesStereoAndResamplesToSixteenKhz() {
        float[] stereo = new float[48_000 * 2];
        for (int frame = 0; frame < 48_000; frame++) {
            stereo[frame * 2] = 0.25f;
            stereo[frame * 2 + 1] = 0.75f;
        }

        var chunks = preprocessor.prepare(stereo, 48_000, 2);

        assertEquals(1, chunks.size());
        assertEquals(16_000, chunks.get(0).sourceSampleCount());
        assertEquals(128, chunks.get(0).logMelFeatures().length);
    }

    @Test
    void splitsLongAudioIntoIndependentThirtySecondWindows() {
        float[] audio = new float[WhisperLogMelSpectrogram.CHUNK_SAMPLES + 12_000];

        var chunks = preprocessor.prepare(audio, 16_000, 1);

        assertEquals(2, chunks.size());
        assertEquals(WhisperLogMelSpectrogram.CHUNK_SAMPLES, chunks.get(1).startSample());
        assertEquals(12_000, chunks.get(1).sourceSampleCount());
    }

    @Test
    void validatesInterleavedInputAndSampleValues() {
        assertThrows(IllegalArgumentException.class,
                () -> preprocessor.prepare(new float[3], 16_000, 2));
        assertThrows(IllegalArgumentException.class,
                () -> preprocessor.prepare(new float[]{Float.NaN}, 16_000, 1));
        assertThrows(IllegalArgumentException.class,
                () -> preprocessor.prepare(new float[]{1.0f}, 0, 1));
    }

    @Test
    void emptyAudioReturnsNoChunks() {
        assertTrue(preprocessor.prepare(new float[0], 16_000, 1).isEmpty());
    }
}
