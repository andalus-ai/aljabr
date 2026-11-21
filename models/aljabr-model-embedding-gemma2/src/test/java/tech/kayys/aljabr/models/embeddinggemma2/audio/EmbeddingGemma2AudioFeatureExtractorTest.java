package tech.kayys.aljabr.models.embeddinggemma2.audio;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class EmbeddingGemma2AudioFeatureExtractorTest {
    private final EmbeddingGemma2AudioFeatureExtractor extractor =
            EmbeddingGemma2AudioFeatureExtractor.publishedDefaults();

    @Test
    void extractsPublishedFeatureShapeAndFiniteLogMelValues() {
        float[] waveform = new float[16_000];
        for (int i = 0; i < waveform.length; i++) {
            waveform[i] = (float) (0.2 * Math.sin(2.0 * Math.PI * 440.0 * i / 16_000));
        }
        var result = extractor.extract(waveform, 16_000);
        assertEquals(128, result.inputFeatures()[0].length);
        assertEquals(result.inputFeatures().length, result.inputFeaturesMask().length);
        for (float[] frame : result.inputFeatures()) {
            for (float value : frame) assertTrue(Float.isFinite(value));
        }
        assertTrue(result.inputFeaturesMask()[result.inputFeaturesMask().length - 1]);
    }

    @Test
    void silenceProducesLogMelFloorAndNoNonFiniteValues() {
        var result = extractor.extract(new float[16_000], 16_000);
        float expected = (float) Math.log(0.001);
        for (float[] frame : result.inputFeatures()) {
            for (float value : frame) assertEquals(expected, value, 1e-6f);
        }
    }

    @Test
    void rejectsWrongSampleRateInsteadOfSilentlyMislabelingAudio() {
        assertThrows(IllegalArgumentException.class,
                () -> extractor.extract(new float[1_600], 8_000));
    }

    @Test
    void rejectsNonFiniteWaveformSamples() {
        float[] waveform = new float[1_000];
        waveform[12] = Float.NaN;
        assertThrows(IllegalArgumentException.class,
                () -> extractor.extract(waveform, 16_000));
    }

    @Test
    void returnsDefensiveCopies() {
        var result = extractor.extract(new float[1_000], 16_000);
        float[][] first = result.inputFeatures();
        first[0][0] = 12345f;
        assertNotEquals(12345f, result.inputFeatures()[0][0]);
        boolean[] mask = result.inputFeaturesMask();
        mask[0] = !mask[0];
        assertNotEquals(mask[0], result.inputFeaturesMask()[0]);
    }
}
