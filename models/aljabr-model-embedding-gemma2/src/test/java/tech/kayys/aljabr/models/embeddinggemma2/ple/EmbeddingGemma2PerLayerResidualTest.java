package tech.kayys.aljabr.models.embeddinggemma2.ple;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class EmbeddingGemma2PerLayerResidualTest {

    @Test
    void gatesThePooledLayerSignalAndAddsNormalizedUpdate() {
        float[][] hidden = {{1f, 2f}};
        float[][] ple = {{0.5f, 1f}};
        float[][] gate = {{1f, 0f}, {0f, 1f}};
        float[][] projection = {{1f, 0f}, {0f, 1f}};
        float[][] output = EmbeddingGemma2PerLayerResidual.apply(
                hidden, ple, gate, projection, new float[] {1f, 1f}, 1e-6f);

        assertEquals(1, output.length);
        assertEquals(2, output[0].length);
        assertTrue(Float.isFinite(output[0][0]));
        assertTrue(Float.isFinite(output[0][1]));
        assertNotEquals(hidden[0][0], output[0][0]);
        assertNotEquals(hidden[0][1], output[0][1]);
        assertEquals(1f, hidden[0][0]);
        assertEquals(2f, hidden[0][1]);
    }

    @Test
    void rejectsMismatchedSequenceAndMatrixShapes() {
        assertThrows(IllegalArgumentException.class, () ->
                EmbeddingGemma2PerLayerResidual.apply(
                        new float[][] {{1f, 2f}}, new float[][] {{1f}, {2f}},
                        new float[][] {{1f, 0f}}, new float[][] {{1f}, {0f}},
                        new float[] {1f, 1f}, 1e-6f));
    }
}
