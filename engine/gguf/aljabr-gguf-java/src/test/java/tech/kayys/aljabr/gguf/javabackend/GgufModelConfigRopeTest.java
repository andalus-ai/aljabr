package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufModelConfigRopeTest {

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
