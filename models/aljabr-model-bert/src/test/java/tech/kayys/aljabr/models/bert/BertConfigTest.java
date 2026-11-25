package tech.kayys.aljabr.models.bert;

import org.junit.jupiter.api.Test;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.*;

class BertConfigTest {
    @Test
    void parsesMiniLmL6H384Configuration() {
        BertConfig config = BertConfig.from(Map.of(
                "model_type", "bert", "architectures", new String[] {"BertModel"},
                "vocab_size", 30522, "hidden_size", 384, "num_hidden_layers", 6,
                "num_attention_heads", 12, "intermediate_size", 1536,
                "max_position_embeddings", 512, "type_vocab_size", 2,
                "pad_token_id", 0, "layer_norm_eps", 1.0e-12, "hidden_act", "gelu"));
        assertEquals(30522, config.vocabularySize());
        assertEquals(384, config.hiddenSize());
        assertEquals(6, config.layerCount());
        assertEquals(12, config.attentionHeadCount());
        assertEquals(32, config.headDimension());
        assertEquals(1536, config.intermediateSize());
        assertEquals(512, config.maxPositionEmbeddings());
        assertEquals(0, config.paddingTokenId());
    }

    @Test
    void appliesHuggingFaceCompatibleOptionalDefaults() {
        BertConfig config = BertConfig.from(Map.of("vocab_size", 100, "hidden_size", 32,
                "num_hidden_layers", 2, "num_attention_heads", 4, "intermediate_size", 64));
        assertEquals(512, config.maxPositionEmbeddings());
        assertEquals(2, config.typeVocabularySize());
        assertEquals(0, config.paddingTokenId());
        assertEquals(1.0e-12d, config.layerNormEpsilon());
        assertEquals("gelu", config.hiddenActivation());
    }

    @Test
    void rejectsIncompatibleAttentionDimensions() {
        assertThrows(IllegalArgumentException.class, () -> BertConfig.from(Map.of(
                "vocab_size", 100, "hidden_size", 30, "num_hidden_layers", 2,
                "num_attention_heads", 8, "intermediate_size", 64)));
    }

    @Test
    void rejectsMissingRequiredFieldsAndInvalidPaddingId() {
        assertThrows(IllegalArgumentException.class, () -> BertConfig.from(Map.of("hidden_size", 32)));
        assertThrows(IllegalArgumentException.class, () -> BertConfig.from(Map.of(
                "vocab_size", 100, "hidden_size", 32, "num_hidden_layers", 2,
                "num_attention_heads", 4, "intermediate_size", 64, "pad_token_id", 100)));
    }

    @Test
    void rejectsFractionalDimensions() {
        assertThrows(IllegalArgumentException.class, () -> BertConfig.from(Map.of(
                "vocab_size", 100, "hidden_size", 32.5, "num_hidden_layers", 2,
                "num_attention_heads", 4, "intermediate_size", 64)));
    }
}
