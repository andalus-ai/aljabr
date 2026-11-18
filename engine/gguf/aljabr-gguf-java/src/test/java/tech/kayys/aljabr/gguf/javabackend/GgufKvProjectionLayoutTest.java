package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufKvProjectionLayoutTest {
    private static final int EMBEDDING_DIMENSION = 128;

    @Test
    void acceptsIndependentKeyAndValueProjections() {
        assertDoesNotThrow(() -> validate(keys(2, 2, 64, 0, 1), values(2, 2, 64, 0, 1),
                new int[] {2, 2}, new int[] {64, 64}, new boolean[] {false, false}));
    }

    @Test
    void acceptsKeyOnlyOwnerBecauseRunnerUsesKeyAsValue() {
        assertDoesNotThrow(() -> validate(keys(1, 2, 64, 0), absent(1),
                new int[] {2}, new int[] {64}, new boolean[] {false}));
    }

    @Test
    void acceptsSharedLayerWithCompatibleOwnerGeometry() {
        assertDoesNotThrow(() -> validate(keys(2, 2, 64, 0), values(2, 2, 64, 0),
                new int[] {2, 2}, new int[] {64, 64}, new boolean[] {false, false}));
    }

    @Test
    void rejectsValueProjectionWithoutKeyProjection() {
        assertThrows(IllegalArgumentException.class,
                () -> validate(absent(1), values(1, 2, 64, 0),
                        new int[] {2}, new int[] {64}, new boolean[] {false}));
    }

    @Test
    void rejectsSharedLayerWithoutKeyOwner() {
        assertThrows(IllegalStateException.class,
                () -> validate(absent(2), absent(2),
                        new int[] {2, 2}, new int[] {64, 64}, new boolean[] {false, false}));
    }

    @Test
    void rejectsSharedLayerWithIncompatibleOwnerGeometry() {
        assertThrows(IllegalArgumentException.class,
                () -> validate(keys(2, 2, 64, 0), values(2, 2, 64, 0),
                        new int[] {2, 4}, new int[] {64, 64}, new boolean[] {false, false}));
    }

    @Test
    void rejectsMismatchedLayoutArrayLengths() {
        assertThrows(IllegalArgumentException.class,
                () -> validate(keys(1, 2, 64, 0), absent(2),
                        new int[] {2}, new int[] {64}, new boolean[] {false}));
    }

    @Test
    void rejectsKeyProjectionWithWrongInputWidth() {
        assertThrows(IllegalArgumentException.class,
                () -> validate(new GGUFTensorInfo[] {tensor(0, "attn_k", 96, 128)},
                        absent(1), new int[] {2}, new int[] {64}, new boolean[] {false}));
    }

    @Test
    void rejectsKeyProjectionWithWrongOutputWidth() {
        assertThrows(IllegalArgumentException.class,
                () -> validate(new GGUFTensorInfo[] {tensor(0, "attn_k", 128, 96)},
                        absent(1), new int[] {2}, new int[] {64}, new boolean[] {false}));
    }

    @Test
    void rejectsValueProjectionWithWrongOutputWidth() {
        assertThrows(IllegalArgumentException.class,
                () -> validate(keys(1, 2, 64, 0),
                        new GGUFTensorInfo[] {tensor(0, "attn_v", 128, 96)},
                        new int[] {2}, new int[] {64}, new boolean[] {false}));
    }

    @Test
    void rejectsNonPositiveEmbeddingDimension() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufKvProjectionLayout.validate(keys(1, 2, 64, 0), values(1, 2, 64, 0),
                        new int[] {2}, new int[] {64}, new boolean[] {false}, 0));
    }

    private static void validate(GGUFTensorInfo[] keys, GGUFTensorInfo[] values,
            int[] kvHeads, int[] headDimensions, boolean[] swa) {
        GgufKvProjectionLayout.validate(
                keys, values, kvHeads, headDimensions, swa, EMBEDDING_DIMENSION);
    }

    private static GGUFTensorInfo[] keys(int layerCount, int kvHeads, int headDimension, int... presentLayers) {
        return projections(layerCount, "attn_k", kvHeads, headDimension, presentLayers);
    }

    private static GGUFTensorInfo[] values(int layerCount, int kvHeads, int headDimension, int... presentLayers) {
        return projections(layerCount, "attn_v", kvHeads, headDimension, presentLayers);
    }

    private static GGUFTensorInfo[] absent(int layers) {
        return new GGUFTensorInfo[layers];
    }

    private static GGUFTensorInfo[] projections(
            int layerCount, String name, int kvHeads, int headDimension, int... presentLayers) {
        GGUFTensorInfo[] result = new GGUFTensorInfo[layerCount];
        for (int layer : presentLayers) {
            result[layer] = tensor(layer, name, EMBEDDING_DIMENSION, kvHeads * headDimension);
        }
        return result;
    }

    private static GGUFTensorInfo tensor(int layer, String projection, int inputDimension, int outputDimension) {
        return new GGUFTensorInfo("blk." + layer + "." + projection + ".weight",
                new long[] {inputDimension, outputDimension}, 0, 0L,
                (long) inputDimension * outputDimension * Float.BYTES);
    }
}
