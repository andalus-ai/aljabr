package tech.kayys.aljabr.models.qwen3asr.audio;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3AsrAudioFeatureExtractorTest {
    @Test void exposesPublishedProcessorGeometry() {
        assertEquals(16_000, Qwen3AsrAudioFeatureExtractor.SAMPLE_RATE);
        assertEquals(128, Qwen3AsrAudioFeatureExtractor.FEATURE_SIZE);
        assertEquals(400, Qwen3AsrAudioFeatureExtractor.FFT_SIZE);
        assertEquals(160, Qwen3AsrAudioFeatureExtractor.HOP_LENGTH);
        assertEquals(480_000, Qwen3AsrAudioFeatureExtractor.CHUNK_SAMPLES);
        assertEquals(3_000, Qwen3AsrAudioFeatureExtractor.MAX_FRAMES);
        assertEquals(8_000, Qwen3AsrAudioFeatureExtractor.MIN_SAMPLES);
    }
    @Test void rejectsAudioShorterThanHalfASecond() {
        assertThrows(IllegalArgumentException.class,
                () -> new Qwen3AsrAudioFeatureExtractor().extract(new float[7_999], 16_000, 1));
    }
    @Test void rejectsInvalidInterleavedChannelGeometry() {
        assertThrows(IllegalArgumentException.class,
                () -> new Qwen3AsrAudioFeatureExtractor().extract(new float[8_001], 16_000, 2));
    }
    @Test void rejectsNonFinitePcm() {
        float[] samples = new float[8_000]; samples[123] = Float.NaN;
        assertThrows(IllegalArgumentException.class,
                () -> new Qwen3AsrAudioFeatureExtractor().extract(samples, 16_000, 1));
    }
    @Test void audioFeaturesDefensivelyCopyMatrix() {
        float[][] input = new float[128][3_000];
        var feature = new Qwen3AsrAudioFeatureExtractor.AudioFeatures(0, 0, 8_000, input);
        input[0][0] = 7.0f;
        assertEquals(0.0f, feature.inputFeatures()[0][0]);
        float[][] returned = feature.inputFeatures(); returned[0][0] = 9.0f;
        assertEquals(0.0f, feature.inputFeatures()[0][0]);
    }
}
