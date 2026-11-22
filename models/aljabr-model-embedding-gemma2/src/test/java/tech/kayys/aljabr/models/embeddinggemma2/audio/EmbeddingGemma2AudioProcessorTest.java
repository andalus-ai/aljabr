package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingGemma2AudioProcessorTest {
    private static EmbeddingGemma2AudioConfig config() {
        return new EmbeddingGemma2AudioConfig(
                "gemma4_audio",
                1024, 12, 8, 1536, 5, List.of(128, 32),
                12, 13, 0, 1e-6, 0.5, "silu", "bfloat16", false,
                16_000, 128, 512, 320, 160, 0.0, 8_000.0,
                0.001, 0.0, 1.0, 0.0,
                280, 40, 1, 2, 3, 1.0e10);
    }

    @Test
    void composesStereoDownmixResamplingAndLogMelExtraction() {
        float[] stereoAtEightKhz = new float[800 * 2];
        for (int i = 0; i < 800; i++) {
            float sample = (float) (0.2 * Math.sin(2.0 * Math.PI * 440.0 * i / 8_000));
            stereoAtEightKhz[i * 2] = sample;
            stereoAtEightKhz[i * 2 + 1] = sample;
        }

        var result = new EmbeddingGemma2AudioProcessor(config())
                .process(stereoAtEightKhz, 2, 8_000);

        assertEquals(16_000, result.sampleRate());
        assertEquals(2, result.sourceChannels());
        assertEquals(8_000, result.sourceSampleRate());
        assertEquals(1_600, result.processedSampleCount());
        assertEquals(0.1, result.durationSeconds(), 1e-9);
        assertFalse(result.truncated());
        assertEquals(128, result.features().inputFeatures()[0].length);
        assertEquals(result.features().inputFeatures().length,
                result.features().inputFeaturesMask().length);
    }

    @Test
    void truncatesAtConfiguredInputLimitBeforeFeatureExtraction() {
        float[] longWaveform = new float[2_000];
        var result = new EmbeddingGemma2AudioProcessor(config(), 1_024)
                .process(longWaveform, 1, 16_000);

        assertTrue(result.truncated());
        assertEquals(1_024, result.processedSampleCount());
        assertEquals(1_024.0 / 16_000, result.durationSeconds(), 1e-9);
    }

    @Test
    void rejectsUnsupportedNonZeroDitherRatherThanSilentlyIgnoringIt() {
        var base = config();
        var withDither = new EmbeddingGemma2AudioConfig(
                base.modelType(), base.hiddenSize(), base.numHiddenLayers(),
                base.numAttentionHeads(), base.outputProjectionSize(),
                base.convolutionKernelSize(), base.subsamplingConvolutionChannels(),
                base.attentionChunkSize(), base.attentionContextLeft(),
                base.attentionContextRight(), base.rmsNormEps(), base.residualWeight(),
                base.hiddenActivation(), base.dtype(), base.useClippedLinears(),
                base.samplingRate(), base.featureSize(), base.fftLength(),
                base.frameLength(), base.hopLength(), base.minFrequency(),
                base.maxFrequency(), base.melFloor(), base.paddingValue(),
                base.inputScaleFactor(), 0.01, base.audioSequenceLength(),
                base.audioMillisecondsPerToken(), base.audioTokenId(),
                base.beginAudioTokenId(), base.endAudioTokenId(), base.gradientClipping());

        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2AudioProcessor(withDither));
    }
}
