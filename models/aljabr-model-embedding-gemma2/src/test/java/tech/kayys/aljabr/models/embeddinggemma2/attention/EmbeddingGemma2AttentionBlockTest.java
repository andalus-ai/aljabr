package tech.kayys.aljabr.models.embeddinggemma2.attention;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2LayerConfig;

class EmbeddingGemma2AttentionBlockTest {

    @Test
    void composesCheckpointProjectionsNormsBidirectionalAttentionAndOutputProjection() {
        var layer = new EmbeddingGemma2LayerConfig(0,
                EmbeddingGemma2LayerConfig.AttentionType.FULL,
                2, 1, 2, 10000d, "default", null, null);
        float[][] identity2 = {{1f, 0f}, {0f, 1f}};
        var weights = new EmbeddingGemma2AttentionBlock.Weights(
                identity2, null, identity2, null, identity2, null, identity2, null,
                new float[]{1f, 1f}, new float[]{1f, 1f});
        var block = new EmbeddingGemma2AttentionBlock(2, layer, weights, 1e-6d);

        float[][] output = block.compute(new float[][]{{1f, 1f}, {1f, 1f}},
                new int[]{0, 1}, null);

        assertEquals(2, output.length);
        assertEquals(2, output[0].length);
        assertEquals(1f, output[0][0], 1e-5f);
        assertEquals(1f, output[0][1], 1e-5f);
        assertEquals(1f, output[1][0], 1e-5f);
        assertEquals(1f, output[1][1], 1e-5f);
    }

    @Test
    void validatesPerLayerProjectionWidthsBeforeExecution() {
        var layer = new EmbeddingGemma2LayerConfig(3,
                EmbeddingGemma2LayerConfig.AttentionType.SLIDING,
                4, 2, 2, 10000d, "default", null, 512);
        float[][] wrong = {{1f, 0f}, {0f, 1f}};
        var weights = new EmbeddingGemma2AttentionBlock.Weights(
                wrong, null, wrong, null, wrong, null, wrong, null,
                new float[]{1f, 1f}, new float[]{1f, 1f});
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2AttentionBlock(2, layer, weights, 1e-6d));
    }
}
