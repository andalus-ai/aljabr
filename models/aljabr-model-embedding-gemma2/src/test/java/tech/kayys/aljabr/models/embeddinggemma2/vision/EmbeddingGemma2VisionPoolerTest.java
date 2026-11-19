package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class EmbeddingGemma2VisionPoolerTest {
    @Test
    void spatiallyAveragesKByKCellsAndScalesByRootHiddenSize() {
        var pooler = new EmbeddingGemma2VisionPooler(1);
        float[][] states = {{1f}, {3f}, {5f}, {7f}};
        int[][] positions = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};

        var result = pooler.pool(states, positions, new boolean[4], 1);

        assertArrayEquals(new float[]{4f}, result.hiddenStates()[0], 1e-6f);
        assertArrayEquals(new boolean[]{false}, result.paddingMask());
    }

    @Test
    void paddingContributesZeroButStillUsesFixedKernelAreaDenominator() {
        var pooler = new EmbeddingGemma2VisionPooler(2);
        float[][] states = {{2f, 4f}, {6f, 8f}, {100f, 100f}, {100f, 100f}};
        int[][] positions = {{0, 0}, {1, 0}, {-1, -1}, {-1, -1}};

        var result = pooler.pool(states, positions,
                new boolean[]{false, false, true, true}, 1);

        // (2+6)/4 and (4+8)/4, then sqrt(2) scaling.
        assertEquals(2f * (float) Math.sqrt(2), result.hiddenStates()[0][0], 1e-6f);
        assertEquals(3f * (float) Math.sqrt(2), result.hiddenStates()[0][1], 1e-6f);
        assertFalse(result.paddingMask()[0]);
    }

    @Test
    void preservesSequenceWhenNoSpatialPoolingIsRequestedAndZerosPadding() {
        var pooler = new EmbeddingGemma2VisionPooler(1);
        var result = pooler.pool(
                new float[][]{{3f}, {9f}},
                new int[][]{{0, 0}, {-1, -1}},
                new boolean[]{false, true},
                2);

        assertArrayEquals(new float[]{3f, 0f},
                new float[]{result.hiddenStates()[0][0], result.hiddenStates()[1][0]}, 1e-6f);
        assertArrayEquals(new boolean[]{false, true}, result.paddingMask());
    }

    @Test
    void rejectsInvalidGeometryAndContradictoryPaddingMetadata() {
        var pooler = new EmbeddingGemma2VisionPooler(1);
        assertThrows(IllegalArgumentException.class, () -> pooler.pool(
                new float[][]{{1f}, {2f}, {3f}},
                new int[][]{{0, 0}, {1, 0}, {2, 0}},
                new boolean[3], 1));
        assertThrows(IllegalArgumentException.class, () -> pooler.pool(
                new float[][]{{1f}},
                new int[][]{{-1, -1}},
                new boolean[]{false}, 1));
    }

    @Test
    void returnsDefensiveCopies() {
        var pooler = new EmbeddingGemma2VisionPooler(1);
        var result = pooler.pool(
                new float[][]{{2f}},
                new int[][]{{0, 0}},
                new boolean[]{false}, 1);
        float[][] first = result.hiddenStates();
        first[0][0] = 99f;
        assertEquals(2f, result.hiddenStates()[0][0], 1e-6f);
    }
}
