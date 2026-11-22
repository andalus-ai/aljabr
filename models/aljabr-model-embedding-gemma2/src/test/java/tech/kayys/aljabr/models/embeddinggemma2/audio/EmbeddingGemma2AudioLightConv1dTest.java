package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.List;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingGemma2AudioLightConv1dTest {
    private static EmbeddingGemma2AudioConfig config(boolean clipped) {
        return new EmbeddingGemma2AudioConfig(
                "gemma4_audio", 2, 1, 1, 8, 3, List.of(2, 2),
                12, 13, 0, 1e-6, 0.5, "silu", "float32", clipped,
                16_000, 4, 8, 4, 2, 0.0, 8_000.0, 0.001,
                0.0, 1.0, 0.0, 280, 40, 1, 2, 3, 1.0e10);
    }

    private static EmbeddingGemma2AudioLightConv1d.Weights weights() {
        float[][] start = new float[4][2];
        start[0][0] = 1f;
        start[1][1] = 1f;
        start[2][0] = 1f;
        start[3][1] = 1f;
        float[][] end = {{1f, 0f}, {0f, 1f}};
        return new EmbeddingGemma2AudioLightConv1d.Weights(
                new EmbeddingGemma2AudioLightConv1d.LinearWeights(
                        start, EmbeddingGemma2AudioLightConv1d.ClipRange.unbounded()),
                new EmbeddingGemma2AudioLightConv1d.LinearWeights(
                        end, EmbeddingGemma2AudioLightConv1d.ClipRange.unbounded()),
                new float[][]{{0f, 0f, 1f}, {0f, 0f, 1f}},
                new float[]{1f, 1f},
                new float[]{1f, 1f});
    }

    @Test
    void preservesShapeAndAddsResidualWhenDepthwiseKernelIsZero() {
        var base = weights();
        var zeroKernel = new EmbeddingGemma2AudioLightConv1d.Weights(
                base.linearStart(), base.linearEnd(), new float[2][3],
                base.preNormScale(), base.convolutionNormScale());
        float[][][] input = {{{1f, 2f}, {-1f, -2f}}};

        float[][][] output = new EmbeddingGemma2AudioLightConv1d(config(false), zeroKernel).compute(input);

        assertEquals(1, output.length);
        assertEquals(2, output[0].length);
        assertEquals(2, output[0][0].length);
        assertArrayEquals(input[0][0], output[0][0], 1e-6f);
        assertArrayEquals(input[0][1], output[0][1], 1e-6f);
    }

    @Test
    void convolutionIsCausalAndDoesNotReadFutureFrames() {
        var model = new EmbeddingGemma2AudioLightConv1d(config(false), weights());
        float[][][] first = {{{1f, 2f}, {3f, 4f}}};
        float[][][] changedFuture = {{{1f, 2f}, {300f, 400f}}};

        float[][][] a = model.compute(first);
        float[][][] b = model.compute(changedFuture);

        assertArrayEquals(a[0][0], b[0][0], 1e-6f);
        assertNotEquals(a[0][1][0], b[0][1][0], 1e-6f);
    }

    @Test
    void appliesCheckpointLinearClippingWhenEnabled() {
        var base = weights();
        var boundedStart = new EmbeddingGemma2AudioLightConv1d.LinearWeights(
                base.linearStart().kernel(),
                new EmbeddingGemma2AudioLightConv1d.ClipRange(0, 0, 0, 0));
        var bounded = new EmbeddingGemma2AudioLightConv1d.Weights(
                boundedStart, base.linearEnd(), base.depthwiseKernel(),
                base.preNormScale(), base.convolutionNormScale());
        float[][][] input = {{{1f, 2f}}};

        float[][][] output = new EmbeddingGemma2AudioLightConv1d(config(true), bounded).compute(input);

        assertArrayEquals(input[0][0], output[0][0], 1e-6f);
    }

    @Test
    void rejectsIncorrectDepthwiseKernelShape() {
        var base = weights();
        var malformed = new EmbeddingGemma2AudioLightConv1d.Weights(
                base.linearStart(), base.linearEnd(), new float[2][2],
                base.preNormScale(), base.convolutionNormScale());

        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2AudioLightConv1d(config(false), malformed));
    }

    @Test
    void rejectsMalformedLinearDimensions() {
        var base = weights();
        var malformed = new EmbeddingGemma2AudioLightConv1d.Weights(
                new EmbeddingGemma2AudioLightConv1d.LinearWeights(
                        new float[3][2], EmbeddingGemma2AudioLightConv1d.ClipRange.unbounded()),
                base.linearEnd(), base.depthwiseKernel(),
                base.preNormScale(), base.convolutionNormScale());

        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2AudioLightConv1d(config(false), malformed));
    }

    @Test
    void doesNotMutateInputOrExposeCheckpointArrays() {
        var base = weights();
        float[][][] input = {{{1f, 2f}}};
        float original = input[0][0][0];
        float[][] kernel = base.depthwiseKernel();
        kernel[0][2] = 1000f;

        new EmbeddingGemma2AudioLightConv1d(config(false), base).compute(input);

        assertEquals(original, input[0][0][0]);
        assertEquals(1f, base.depthwiseKernel()[0][2]);
    }
}
