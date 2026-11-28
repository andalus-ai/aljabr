package tech.kayys.aljabr.models.qwen3asr.encoder;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3AsrAudioEncoderLayerTest {
    @Test
    void appliesBidirectionalAttentionOnlyWithinEachPackedWindow() {
        var layer = new Qwen3AsrAudioEncoderLayer(2, 1, 2, weights());
        float[][] input = {{1.0f, 0.0f}, {0.0f, 2.0f}};

        float[][] separate = layer.compute(input, new int[]{0, 1, 2});
        float[][] together = layer.compute(input, new int[]{0, 2});

        assertNotEquals(separate[0][0], together[0][0], 1.0e-4f);
        assertNotEquals(separate[1][1], together[1][1], 1.0e-4f);
    }

    @Test
    void preservesTokenCountAndHiddenWidth() {
        var layer = new Qwen3AsrAudioEncoderLayer(2, 1, 2, weights());
        float[][] output = layer.compute(
                new float[][]{{0.25f, -0.5f}, {1.0f, 0.75f}, {-0.25f, 0.5f}},
                new int[]{0, 2, 3});

        assertEquals(3, output.length);
        for (float[] token : output) {
            assertEquals(2, token.length);
            for (float value : token) assertTrue(Float.isFinite(value));
        }
    }

    @Test
    void rejectsNonMonotonicOrIncompleteWindowBoundaries() {
        var layer = new Qwen3AsrAudioEncoderLayer(2, 1, 2, weights());
        float[][] input = {{1.0f, 0.0f}, {0.0f, 1.0f}};

        assertThrows(IllegalArgumentException.class,
                () -> layer.compute(input, new int[]{0, 0, 2}));
        assertThrows(IllegalArgumentException.class,
                () -> layer.compute(input, new int[]{0, 1}));
        assertThrows(IllegalArgumentException.class,
                () -> layer.compute(input, new int[]{1, 2}));
    }

    @Test
    void rejectsNonFiniteInput() {
        var layer = new Qwen3AsrAudioEncoderLayer(2, 1, 2, weights());
        assertThrows(IllegalArgumentException.class,
                () -> layer.compute(new float[][]{{Float.NaN, 1.0f}}, new int[]{0, 1}));
    }

    @Test
    void rejectsCheckpointWeightShapeMismatch() {
        var invalid = new Qwen3AsrAudioEncoderLayer.Weights(
                new float[]{1, 1}, new float[]{0, 0},
                new float[][]{{0}}, new float[]{0, 0},
                identity(), new float[]{0, 0},
                identity(), new float[]{0, 0},
                identity(), new float[]{0, 0},
                new float[]{1, 1}, new float[]{0, 0},
                new float[][]{{0, 0}, {0, 0}}, new float[]{0, 0},
                new float[][]{{0, 0}, {0, 0}}, new float[]{0, 0}, 1.0e-5);
        assertThrows(IllegalArgumentException.class,
                () -> new Qwen3AsrAudioEncoderLayer(2, 1, 2, invalid));
    }

    @Test
    void configFactoryUsesQwen3AsrAudioDimensions() {
        var config = tech.kayys.aljabr.models.qwen3asr.Qwen3AsrConfig.qwen3Asr17B().audio();
        assertThrows(IllegalArgumentException.class,
                () -> Qwen3AsrAudioEncoderLayer.from(config, weights()));
    }

    private static Qwen3AsrAudioEncoderLayer.Weights weights() {
        return new Qwen3AsrAudioEncoderLayer.Weights(
                new float[]{1, 1}, new float[]{0, 0},
                zeroMatrix(2, 2), new float[]{0, 0},
                zeroMatrix(2, 2), new float[]{0, 0},
                identity(), new float[]{0, 0},
                identity(), new float[]{0, 0},
                new float[]{1, 1}, new float[]{0, 0},
                zeroMatrix(2, 2), new float[]{0, 0},
                zeroMatrix(2, 2), new float[]{0, 0},
                1.0e-5);
    }

    private static float[][] identity() {
        return new float[][]{{1, 0}, {0, 1}};
    }

    private static float[][] zeroMatrix(int rows, int columns) {
        return new float[rows][columns];
    }
}
