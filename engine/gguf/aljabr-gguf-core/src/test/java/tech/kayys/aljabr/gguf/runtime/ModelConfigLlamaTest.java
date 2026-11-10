package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.mapper.GgufMetadataMapper;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ModelConfigLlamaTest {
    @Test
    void mapsLlamaGgufMetadataToDirectEngineConfig() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "llama");
        metadata.put("llama.context_length", 8192);
        metadata.put("llama.embedding_length", 4096);
        metadata.put("llama.block_count", 32);
        metadata.put("llama.feed_forward_length", 11008);
        metadata.put("llama.attention.head_count", 32);
        metadata.put("llama.attention.head_count_kv", 8);
        metadata.put("llama.attention.layer_norm_rms_epsilon", 0.00001d);
        metadata.put("llama.rope.freq_base", 500000.0d);
        metadata.put("tokenizer.ggml.tokens", List.of("a", "b", "c"));
        metadata.put("tokenizer.ggml.bos_token_id", 1);
        metadata.put("tokenizer.ggml.eos_token_id", 2);

        ModelConfig config = new GgufMetadataMapper().fromGgufMetadata(metadata);

        assertEquals("llama", config.getModelType());
        assertEquals("LlamaForCausalLM", config.getPrimaryArchitecture());
        assertEquals(8192, config.getMaxPositionEmbeddings());
        assertEquals(4096, config.getHiddenSize());
        assertEquals(32, config.getNumHiddenLayers());
        assertEquals(11008, config.getIntermediateSize());
        assertEquals(32, config.getNumAttentionHeads());
        assertEquals(8, config.resolvedNumKvHeads());
        assertEquals(128, config.getResolvedHeadDim());
        assertEquals(3, config.getVocabSize());
        assertEquals(500000.0d, config.getRopeTheta(), 0.0001d);
        assertEquals(1, config.getBosTokenId().orElseThrow());
        assertEquals(2, config.getEosTokenId().orElseThrow());
    }
}
