package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufAttentionOutputProjectionLayoutTest {
    private static final int QUERY_HEADS = 4;
    private static final int HEAD_DIMENSION = 32;
    private static final int EMBEDDING_DIMENSION = 128;

    @Test
    void acceptsProjectionMatchingQueryContextAndEmbeddingDimensions() {
        assertDoesNotThrow(() -> validate(new GGUFTensorInfo[] {
                tensor(0, EMBEDDING_DIMENSION, QUERY_HEADS * HEAD_DIMENSION)
        }, new int[] {HEAD_DIMENSION}));
    }

    @Test
    void rejectsProjectionWithWrongQueryContextInputWidth() {
        assertThrows(IllegalArgumentException.class,
                () -> validate(new GGUFTensorInfo[] {
                        tensor(0, EMBEDDING_DIMENSION, QUERY_HEADS * HEAD_DIMENSION - 1)
                }, new int[] {HEAD_DIMENSION}));
    }

    @Test
    void rejectsProjectionWithWrongEmbeddingOutputWidth() {
        assertThrows(IllegalArgumentException.class,
                () -> validate(new GGUFTensorInfo[] {
                        tensor(0, EMBEDDING_DIMENSION - 1, QUERY_HEADS * HEAD_DIMENSION)
                }, new int[] {HEAD_DIMENSION}));
    }

    @Test
    void validatesHeterogeneousPerLayerHeadDimensions() {
        assertDoesNotThrow(() -> validate(new GGUFTensorInfo[] {
                tensor(0, EMBEDDING_DIMENSION, QUERY_HEADS * 16),
                tensor(1, EMBEDDING_DIMENSION, QUERY_HEADS * 32)
        }, new int[] {16, 32}));
    }

    @Test
    void rejectsMissingProjection() {
        assertThrows(IllegalArgumentException.class,
                () -> validate(new GGUFTensorInfo[] {null}, new int[] {HEAD_DIMENSION}));
    }

    @Test
    void rejectsNonPositiveQueryHeadGeometry() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufAttentionOutputProjectionLayout.validate(
                        new GGUFTensorInfo[] {tensor(0, EMBEDDING_DIMENSION, 128)},
                        0, new int[] {HEAD_DIMENSION}, EMBEDDING_DIMENSION));
    }

    @Test
    void rejectsMismatchedLayerArrayLengths() {
        assertThrows(IllegalArgumentException.class,
                () -> validate(new GGUFTensorInfo[] {
                        tensor(0, EMBEDDING_DIMENSION, QUERY_HEADS * HEAD_DIMENSION)
                }, new int[] {HEAD_DIMENSION, HEAD_DIMENSION}));
    }

    private static void validate(GGUFTensorInfo[] outputWeights, int[] headDimensions) {
        GgufAttentionOutputProjectionLayout.validate(
                outputWeights, QUERY_HEADS, headDimensions, EMBEDDING_DIMENSION);
    }

    private static GGUFTensorInfo tensor(int layer, int outputDimension, int inputDimension) {
        return new GGUFTensorInfo("blk." + layer + ".attn_output.weight",
                new long[] {inputDimension, outputDimension}, 0, 0L,
                (long) inputDimension * outputDimension * Float.BYTES);
    }
}
