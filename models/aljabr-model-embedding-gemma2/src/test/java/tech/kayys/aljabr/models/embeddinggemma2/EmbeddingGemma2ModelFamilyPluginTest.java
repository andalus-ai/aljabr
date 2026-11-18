package tech.kayys.aljabr.models.embeddinggemma2;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyContractValidator;
import tech.kayys.aljabr.spi.model.ModelFamilyContractViolation;
import tech.kayys.aljabr.spi.model.ModelFamilyDirectSupport;
import tech.kayys.aljabr.spi.model.ModelTokenizerKind;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmbeddingGemma2ModelFamilyPluginTest {

    @Test
    void pluginSatisfiesSharedModelFamilyContract() {
        var plugin = new EmbeddingGemma2ModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                .map(ModelFamilyContractViolation::summary)
                .toList());
    }

    @Test
    void declaresDistinctEmbeddingFamilyWithoutCausalOrFalseReadinessClaims() {
        var plugin = new EmbeddingGemma2ModelFamilyPlugin();
        var descriptor = plugin.descriptor();

        assertEquals("embedding_gemma2", descriptor.id());
        assertEquals(List.of("embedding_gemma2", "embedding_gemma2_text"), descriptor.modelTypes());
        assertEquals(List.of("EmbeddingGemma2Model"), descriptor.architectureClassNames());
        assertTrue(descriptor.capabilities().contains(ModelFamilyCapability.EMBEDDING));
        assertTrue(descriptor.capabilities().contains(ModelFamilyCapability.TOKENIZER));
        assertFalse(descriptor.capabilities().contains(ModelFamilyCapability.CAUSAL_LM));
        assertFalse(descriptor.capabilities().contains(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE));
        assertEquals("pending_text_encoder_and_embedding_head",
                descriptor.metadata().get("direct_safetensor"));
    }

    @Test
    void declaresGemmaSentencePieceTokenizerAndTextOnlyFirstMilestone() {
        var plugin = new EmbeddingGemma2ModelFamilyPlugin();

        assertEquals(1, plugin.tokenizerDescriptors().size());
        assertEquals("embedding-gemma2-sentencepiece", plugin.tokenizerDescriptors().getFirst().id());
        assertEquals(ModelTokenizerKind.SENTENCE_PIECE_BPE,
                plugin.tokenizerDescriptors().getFirst().kind());
        assertEquals("text", plugin.descriptor().metadata().get("supported_modalities"));
        assertEquals("image,audio,video", plugin.descriptor().metadata().get("deferred_modalities"));
    }

    @Test
    void supportReportKeepsRunnerPending() {
        var report = new EmbeddingGemma2ModelFamilyPlugin().supportReport();

        assertEquals(ModelFamilyDirectSupport.PENDING, report.directSafetensorStatus());
    }
}
