package tech.kayys.aljabr.models.whisper;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class WhisperModelFamilyPluginTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void declaresAudioAndTokenizerCapabilitiesWithoutClaimingInferenceReadiness() {
        WhisperModelFamilyPlugin plugin = new WhisperModelFamilyPlugin();

        assertEquals("whisper", plugin.descriptor().id());
        assertTrue(plugin.descriptor().capabilities().contains(ModelFamilyCapability.AUDIO));
        assertTrue(plugin.descriptor().capabilities().contains(ModelFamilyCapability.TOKENIZER));
        assertEquals("pending_encoder_decoder_audio_runtime",
                plugin.descriptor().metadata().get("direct_safetensor"));
        assertEquals("128", plugin.descriptor().metadata().get("large_v3_mel_bins"));
    }

    @Test
    void recognizesWhisperConditionalGenerationCheckpoint() throws Exception {
        Files.writeString(temporaryDirectory.resolve("config.json"), """
                {
                  "model_type": "whisper",
                  "architectures": ["WhisperForConditionalGeneration"]
                }
                """);

        assertTrue(new WhisperModelFamilyPlugin().matchesCheckpoint(temporaryDirectory));
    }

    @Test
    void rejectsUnrelatedAndMalformedCheckpoints() throws Exception {
        Files.writeString(temporaryDirectory.resolve("config.json"), """
                {"model_type":"llama","architectures":["LlamaForCausalLM"]}
                """);
        assertFalse(WhisperCheckpointInspector.isWhisperCheckpoint(temporaryDirectory));

        Files.writeString(temporaryDirectory.resolve("config.json"), "{not-json");
        assertFalse(WhisperCheckpointInspector.isWhisperCheckpoint(temporaryDirectory));
    }

    @Test
    void canonicalLargeV3DimensionsAreExplicitAndValidated() {
        WhisperLargeV3Config config = WhisperLargeV3Config.canonical();

        assertEquals(1280, config.dModel());
        assertEquals(32, config.encoderLayers());
        assertEquals(32, config.decoderLayers());
        assertEquals(128, config.numMelBins());
        assertEquals(16000, config.sampleRate());
        assertEquals(400, config.fftSize());
        assertEquals(160, config.hopLength());
        assertEquals(30, config.chunkLengthSeconds());
        assertDoesNotThrow(config::requireLargeV3Architecture);
    }

    @Test
    void rejectsDifferentWhisperVariant() {
        WhisperLargeV3Config smallVariant = new WhisperLargeV3Config(
                768, 12, 12, 12, 12, 3072, 3072, 80,
                1500, 448, 51866, 16000, 400, 160, 30);

        assertThrows(IllegalArgumentException.class, smallVariant::requireLargeV3Architecture);
    }

    @Test
    void reportsIncompleteCheckpointRatherThanCallingItSupported() throws Exception {
        Files.writeString(temporaryDirectory.resolve("config.json"), """
                {
                  "model_type":"whisper",
                  "architectures":["WhisperForConditionalGeneration"],
                  "d_model":1280,
                  "encoder_layers":32,
                  "decoder_layers":32,
                  "encoder_attention_heads":20,
                  "decoder_attention_heads":20,
                  "encoder_ffn_dim":5120,
                  "decoder_ffn_dim":5120,
                  "num_mel_bins":128,
                  "max_source_positions":1500,
                  "max_target_positions":448,
                  "vocab_size":51866
                }
                """);

        WhisperCheckpointInspector.CheckpointReport report =
                WhisperCheckpointInspector.inspectLargeV3(temporaryDirectory);

        assertTrue(report.modelConfigValid());
        assertFalse(report.processorConfigValid());
        assertFalse(report.tokenizerAvailable());
        assertFalse(report.weightsAvailable());
        assertFalse(report.supported());
    }
}
