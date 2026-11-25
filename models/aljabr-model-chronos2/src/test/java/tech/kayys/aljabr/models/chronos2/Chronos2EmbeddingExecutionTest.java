package tech.kayys.aljabr.models.chronos2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.common.transformer.ResidualProjectionReference;
import tech.kayys.aljabr.models.common.transformer.ResidualProjectionReference.Activation;
import tech.kayys.aljabr.models.common.transformer.ResidualProjectionReference.Weights;

class Chronos2EmbeddingExecutionTest {
    @Test
    void residualProjectionAddsLearnedSkipProjection() {
        Weights weights = new Weights(
                new float[][] {{1f}, {2f}}, new float[] {0f, 0f},
                new float[][] {{1f}}, new float[] {0f},
                new float[][] {{3f}}, new float[] {1f});
        float[][] result = ResidualProjectionReference.apply(new float[][] {{2f}}, weights, Activation.RELU);
        assertEquals(13f, result[0][0], 1e-6f);
    }

    @Test
    void embedsContextRegressionTokenAndFuturePatches() {
        Chronos2Config config = config(true);
        Weights input = identityResidual(3, 2);
        Weights output = identityResidual(2, 2);
        Chronos2EmbeddingExecution execution = new Chronos2EmbeddingExecution(config, input, output,
                new float[][] {{0f, 0f}, {7f, 8f}});
        float[][][] embedded = execution.embed(
                new float[][][] {{{1f, 0f, 0f}, {0f, 1f, 0f}}},
                new float[][][] {{{0f, 0f, 1f}}});
        assertEquals(3, embedded[0].length);
        assertEquals(7f, embedded[0][1][0], 1e-6f);
        assertEquals(8f, embedded[0][1][1], 1e-6f);
    }

    @Test
    void quantileProjectionUsesQuantileMajorPatchOrder() {
        Chronos2Config config = config(false);
        Weights input = identityResidual(3, 2);
        Weights output = identityResidual(2, 4);
        Chronos2EmbeddingExecution execution = new Chronos2EmbeddingExecution(config, input, output,
                new float[][] {{0f, 0f}});
        float[][][] values = execution.projectQuantiles(new float[][][] {{{1f, 2f}, {3f, 4f}}});
        assertEquals(2, values[0].length);
        assertEquals(4, values[0][0].length);
        assertEquals(1f, values[0][0][0], 1e-6f);
        assertEquals(2f, values[0][0][1], 1e-6f);
        assertEquals(3f, values[0][1][0], 1e-6f);
        assertEquals(4f, values[0][1][1], 1e-6f);
    }

    @Test
    void rejectsInvalidProjectionGeometry() {
        Chronos2Config config = config(false);
        Chronos2EmbeddingExecution execution = new Chronos2EmbeddingExecution(config,
                identityResidual(3, 2), identityResidual(2, 4), new float[][] {{0f, 0f}});
        assertThrows(IllegalArgumentException.class,
                () -> execution.embed(new float[][][] {{{1f}}}, new float[][][] {{{1f, 2f, 3f}}}));
    }

    private static Chronos2Config config(boolean reg) {
        return new Chronos2Config(16, 1, 1, 1, 4, 16, 2, 2, 2, 1, 1,
                1e-6, 10000d, reg, false, List.of(0.1d, 0.9d));
    }

    private static Weights identityResidual(int input, int output) {
        float[][] hidden = new float[input][input];
        float[][] projected = new float[output][input];
        float[][] residual = new float[output][input];
        for (int i = 0; i < input; i++) hidden[i][i] = 1f;
        for (int i = 0; i < Math.min(input, output); i++) residual[i][i] = 1f;
        return new Weights(hidden, new float[input], projected, new float[output],
                residual, new float[output]);
    }
}
