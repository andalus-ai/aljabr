package tech.kayys.aljabr.models.chronos2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class Chronos2EncoderTest {
    @Test
    void groupAttentionMixesOnlySeriesInTheSameGroup() {
        Chronos2Encoder encoder = encoder();
        float[][][] input = {
            {{1f, 0f}},
            {{0f, 1f}}
        };
        boolean[][] valid = {{true}, {true}};

        float[][][] sameGroup = encoder.encode(input, new int[] {3, 3}, valid, new int[] {0});
        float[][][] separateGroups = encoder.encode(input, new int[] {3, 8}, valid, new int[] {0});
        // Series 0 receives information from series 1 only when they share a group.
        assertNotEquals(sameGroup[0][0][1], separateGroups[0][0][1], 1e-4f);
        assertNotEquals(sameGroup[1][0][0], separateGroups[1][0][0], 1e-4f);
    }

    @Test
    void fullyMaskedSeriesDoesNotCreateNanAttention() {
        Chronos2Encoder encoder = encoder();
        float[][][] input = {{{1f, 0f}}, {{0f, 1f}}};
        float[][][] output = encoder.encode(input, new int[] {1, 1},
                new boolean[][] {{false}, {true}}, new int[] {0});
        for (float[][] series : output) {
            for (float[] patch : series) {
                for (float value : patch) {
                    if (!Float.isFinite(value)) throw new AssertionError("encoder output must remain finite");
                }
            }
        }
    }

    @Test
    void validatesInputGeometry() {
        Chronos2Encoder encoder = encoder();
        assertThrows(IllegalArgumentException.class,
                () -> encoder.encode(new float[][][] {{{1f, 2f}}}, new int[] {1},
                        new boolean[][] {{true}}, new int[] {0, 1}));
    }

    private static Chronos2Encoder encoder() {
        Chronos2Config config = new Chronos2Config(
                16, 1, 1, 1, 1, 1,
                2, 2, 2, 1, 1,
                1e-6, 10000d, false, false, java.util.List.of(0.5d));

        float[][] zero = new float[2][2];
        float[][] identity = {{1f, 0f}, {0f, 1f}};
        Chronos2Encoder.AttentionWeights zeroAttention =
                new Chronos2Encoder.AttentionWeights(zero, zero, zero, zero);
        Chronos2Encoder.AttentionWeights groupAttention =
                new Chronos2Encoder.AttentionWeights(zero, zero, identity, identity);

        Chronos2Encoder.LayerWeights layer = new Chronos2Encoder.LayerWeights(
                new Chronos2Encoder.AttentionLayerWeights(new float[] {1f, 1f}, zeroAttention),
                new Chronos2Encoder.AttentionLayerWeights(new float[] {1f, 1f}, groupAttention),
                new Chronos2Encoder.FeedForwardWeights(new float[] {1f, 1f},
                        new float[2][2], new float[2][2]));
        return new Chronos2Encoder(config, new Chronos2Encoder.LayerWeights[] {layer},
                new float[] {1f, 1f});
    }
}
