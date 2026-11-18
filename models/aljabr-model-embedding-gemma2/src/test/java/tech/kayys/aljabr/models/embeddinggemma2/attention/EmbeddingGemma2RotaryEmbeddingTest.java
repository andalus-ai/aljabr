package tech.kayys.aljabr.models.embeddinggemma2.attention;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class EmbeddingGemma2RotaryEmbeddingTest {

    @Test
    void rotatesSplitHalfPairsAndPreservesInput() {
        float[][][] input = {{{1f, 0f, 0f, 1f}}};
        float[][][] output = EmbeddingGemma2RotaryEmbedding.apply(input, new int[]{1}, 1.0d, null);
        assertEquals((float) Math.cos(1.0d), output[0][0][0], 1e-6f);
        assertEquals((float) Math.sin(1.0d), output[0][0][1], 1e-6f);
        assertEquals(-(float) Math.sin(1.0d), output[0][0][2], 1e-6f);
        assertEquals((float) Math.cos(1.0d), output[0][0][3], 1e-6f);
        assertEquals(1f, input[0][0][0]);
    }

    @Test
    void leavesDimensionsOutsidePartialRotaryRangeUnchanged() {
        float[][][] input = {{{1f, 0f, 7f, 8f}}};
        float[][][] output = EmbeddingGemma2RotaryEmbedding.apply(input, new int[]{0}, 10000d, 0.5d);
        assertEquals(1f, output[0][0][0]);
        assertEquals(0f, output[0][0][1]);
        assertEquals(7f, output[0][0][2]);
        assertEquals(8f, output[0][0][3]);
    }

    @Test
    void supportsNonContiguousPositionIds() {
        float[][][] input = {{{1f, 0f}}};
        float[][][] output = EmbeddingGemma2RotaryEmbedding.apply(input, new int[]{3}, 10000d, null);
        assertEquals((float) Math.cos(3d), output[0][0][0], 1e-6f);
        assertEquals((float) Math.sin(3d), output[0][0][1], 1e-6f);
    }
}
