package tech.kayys.aljabr.models.qwen3asr;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3AsrConfigTest {
    @Test
    void published17BConfigurationHasExpectedEncoderAndQwen3BackboneDimensions() {
        Qwen3AsrConfig config = Qwen3AsrConfig.qwen3Asr17B();
        assertEquals(128, config.audio().numMelBins());
        assertEquals(1024, config.audio().modelDimension());
        assertEquals(24, config.audio().layers());
        assertEquals(16, config.audio().attentionHeads());
        assertEquals(2048, config.audio().outputDimension());
        assertEquals(2048, config.text().hiddenSize());
        assertEquals(28, config.text().layers());
        assertEquals(16, config.text().attentionHeads());
        assertEquals(8, config.text().keyValueHeads());
        assertEquals(151_936, config.text().vocabularySize());
        assertEquals(151_676, config.audioTokenId());
        assertEquals(151_705, config.timestampTokenId());
        assertEquals(151_645, config.padTokenId());
        assertEquals(2, config.eosTokenIds().size());
        assertTrue(config.tieWordEmbeddings());
    }

    @Test
    void rejectsMisalignedAudioWindowGeometry() {
        assertThrows(IllegalArgumentException.class, () ->
                new Qwen3AsrConfig.AudioEncoderConfig(
                        128, 1024, 24, 16, 16, 4096, 480, 2048, 50, 801, 500));
    }

    @Test
    void rejectsInconsistentTextAttentionDimensions() {
        assertThrows(IllegalArgumentException.class, () ->
                new Qwen3AsrConfig.TextDecoderConfig(
                        2048, 6144, 28, 16, 8, 64, 151_936, 65_536, 1.0e-6, 1_000_000.0));
    }

    @Test
    void specialTokenListIsDefensivelyCopied() {
        var eos = new java.util.ArrayList<>(java.util.List.of(151_643, 151_645));
        var preset = Qwen3AsrConfig.qwen3Asr17B();
        var config = new Qwen3AsrConfig(preset.audio(), preset.text(),
                151_676, 151_705, 151_645, eos, true);
        eos.clear();
        assertEquals(2, config.eosTokenIds().size());
        assertThrows(UnsupportedOperationException.class, () -> config.eosTokenIds().add(1));
    }
}
