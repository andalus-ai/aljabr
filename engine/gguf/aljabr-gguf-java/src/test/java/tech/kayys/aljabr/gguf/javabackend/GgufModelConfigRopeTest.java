package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufModelConfigRopeTest {

    @Test
    void acceptsCompletePerLayerRoutingMetadata() {
        GgufModelConfig.validateLayerMetadata(4, new int[] {2, 2, 4, 4}, new int[] {1, 1, 0, 1});
    }

    @Test
    void rejectsIncompleteKvHeadMetadata() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufModelConfig.validateLayerMetadata(4, new int[] {2, 2, 4}, null));
    }

    @Test
    void rejectsIncompleteSlidingWindowPattern() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufModelConfig.validateLayerMetadata(4, null, new int[] {1, 0, 1}));
    }

    @Test
    void rejectsInvalidPerLayerKvHeadCounts() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufModelConfig.validateLayerMetadata(2, new int[] {2, 0}, null));
    }

    @Test
    void rejectsInvalidSlidingWindowPatternValues() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufModelConfig.validateLayerMetadata(2, null, new int[] {0, 2}));
    }

    @Test
    void prefersConfiguredRotaryDimensionsFromGgufMetadata() {
        assertEquals(128, GgufModelConfig.resolveRotaryDimension("gemma4", false, 512, 128));
        assertEquals(256, GgufModelConfig.resolveRotaryDimension("gemma4", true, 256, 256));
    }

    @Test
    void defaultsGemma4GlobalAttentionToProportionalRope() {
        assertEquals(128, GgufModelConfig.resolveRotaryDimension("gemma4", false, 512, 0));
    }

    @Test
    void defaultsGemma4SlidingWindowAttentionToFullRope() {
        assertEquals(256, GgufModelConfig.resolveRotaryDimension("gemma4", true, 256, 0));
    }

    @Test
    void preservesFullRopeByDefaultForOtherArchitectures() {
        assertEquals(128, GgufModelConfig.resolveRotaryDimension("llama", false, 128, 0));
        assertEquals(256, GgufModelConfig.resolveRotaryDimension("qwen2", true, 256, 0));
    }

    @Test
    void capsMetadataRotaryDimensionToTheHeadDimensionAndRoundsDownToEven() {
        assertEquals(6, GgufModelConfig.resolveRotaryDimension("llama", false, 8, 7));
        assertEquals(8, GgufModelConfig.resolveRotaryDimension("llama", false, 8, 12));
    }
}
