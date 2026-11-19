package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.attention.EmbeddingGemma2Mlp;

class EmbeddingGemma2VisionEncoderTest {
    @Test
    void executesEveryLayerInCheckpointOrderAndReturnsUnpooledStates() {
        AtomicInteger calls = new AtomicInteger();
        var first = zeroLayer((states, positions, valid) -> {
            calls.incrementAndGet();
            return new float[states.length][2];
        });
        var second = zeroLayer((states, positions, valid) -> {
            calls.incrementAndGet();
            return new float[states.length][2];
        });
        var encoder = new EmbeddingGemma2VisionEncoder(null, List.of(first, second), 2);

        float[][] input = {{1f, 2f}, {3f, 4f}};
        float[][] output = encoder.encodeEmbeddings(
                input, new int[][]{{0, 0}, {1, 0}}, new boolean[]{true, true});

        assertEquals(2, calls.get());
        assertEquals(2, encoder.layerCount());
        assertArrayEquals(input[0], output[0], 1e-5f);
        assertArrayEquals(input[1], output[1], 1e-5f);
        assertNotSame(input[0], output[0]);
    }

    @Test
    void patchEntryPointEmbedsBeforeRunningLayers() {
        var patchEmbedder = new EmbeddingGemma2VisionPatchEmbedder(
                1, 2, 2,
                new EmbeddingGemma2VisionPatchEmbedder.Weights(
                        new float[][]{{1f, 0f, 0f}, {0f, 1f, 0f}},
                        new float[][][]{{{0f, 0f}, {0f, 0f}},
                                {{0f, 0f}, {0f, 0f}}}));
        var encoder = new EmbeddingGemma2VisionEncoder(
                patchEmbedder,
                List.of(zeroLayer((states, positions, valid) -> new float[states.length][2])),
                1);

        float[][] output = encoder.encodePatches(
                new float[][]{{1f, 0f, 0f}},
                new int[][]{{0, 0}}, null);

        assertEquals(1, output.length);
        assertEquals(2, output[0].length);
        assertEquals(1f, output[0][0], 1e-5f);
        assertEquals(-1f, output[0][1], 1e-5f);
    }

    @Test
    void rejectsLayerCountMismatchAndAllPadding() {
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2VisionEncoder(null, List.of(zeroLayer(
                        (states, positions, valid) -> states)), 2));
        var encoder = new EmbeddingGemma2VisionEncoder(null,
                List.of(zeroLayer((states, positions, valid) -> states)), 1);
        assertThrows(IllegalArgumentException.class,
                () -> encoder.encodeEmbeddings(
                        new float[][]{{1f, 2f}, {2f, 3f}},
                        new int[][]{{-1, -1}, {-1, -1}}, null));
    }

    private static EmbeddingGemma2VisionEncoderLayer zeroLayer(
            EmbeddingGemma2VisionEncoderLayer.Attention attention) {
        float[][] zero = {{0f, 0f}, {0f, 0f}};
        float[] ones = {1f, 1f};
        return new EmbeddingGemma2VisionEncoderLayer(
                2, 2, 1e-6f, attention,
                new EmbeddingGemma2Mlp.Weights(zero, zero, zero),
                new EmbeddingGemma2VisionEncoderLayer.Weights(ones, ones, ones, ones));
    }
}
