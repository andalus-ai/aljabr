package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.mapper.GgufMetadataMapper;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelConfigGemma4Test {
    @Test
    void mapsGemma4GgufMetadataIncludingHeterogeneousAttentionHints() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "gemma4");
        metadata.put("gemma4.context_length", 32768);
        metadata.put("gemma4.embedding_length", 2304);
        metadata.put("gemma4.block_count", 5);
        metadata.put("gemma4.feed_forward_length", List.of(6144, 6144, 12288));
        metadata.put("gemma4.attention.head_count", 8);
        metadata.put("gemma4.attention.head_count_kv", 4);
        metadata.put("gemma4.attention.key_length", 256);
        metadata.put("gemma4.attention.key_length_swa", 128);
        metadata.put("gemma4.rope.freq_base", 1_000_000.0d);
        metadata.put("gemma4.rope.freq_base_swa", 10_000.0d);
        metadata.put("gemma4.attention.shared_kv_layers", 2);
        metadata.put("gemma4.attention.sliding_window", 512);
        metadata.put("gemma4.attention.sliding_window_pattern", List.of(true, true, false, true, false));
        metadata.put("gemma4.embedding_length_per_layer_input", 256);
        metadata.put("gemma4.vocab_size", 262144);
        metadata.put("gemma4.final_logit_softcapping", 30.0d);

        ModelConfig config = new GgufMetadataMapper().fromGgufMetadata(metadata);

        assertEquals("gemma4", config.getModelType());
        assertEquals("Gemma4ForConditionalGeneration", config.getPrimaryArchitecture());
        assertEquals(32768, config.getMaxPositionEmbeddings());
        assertEquals(2304, config.getHiddenSize());
        assertEquals(5, config.getNumHiddenLayers());
        assertEquals(12288, config.getIntermediateSize());
        assertEquals(8, config.getNumAttentionHeads());
        assertEquals(4, config.resolvedNumKvHeads());
        assertEquals(128, config.getResolvedHeadDim());
        assertEquals(256, config.getResolvedMaxHeadDim());
        assertEquals(2, config.getResolvedNumKvSharedLayers());
        assertEquals(256, config.getHiddenSizePerLayerInput());
        assertEquals(262144, config.getVocabSize());
        assertTrue(config.hasSlidingWindow());
        assertEquals(512, config.getSlidingWindowSize());
        assertEquals("sliding_attention", config.getLayerType(0));
        assertEquals("full_attention", config.getLayerType(2));
        assertEquals(1_000_000.0d, config.ropeThetaForLayer(2), 0.0001d);
        assertEquals(10_000.0d, config.ropeThetaForLayer(0), 0.0001d);
        assertEquals(30.0d, config.getFinalLogitSoftcapping(), 0.0001d);
    }

    @Test
    void mapsGemma4UnifiedGgufArchitectureClassName() {
        ModelConfig config = new GgufMetadataMapper().fromGgufMetadata(Map.of("general.architecture", "gemma4_unified"));

        assertEquals("gemma4_unified", config.getModelType());
        assertEquals("Gemma4ForMultimodalLM", config.getPrimaryArchitecture());
    }
}
