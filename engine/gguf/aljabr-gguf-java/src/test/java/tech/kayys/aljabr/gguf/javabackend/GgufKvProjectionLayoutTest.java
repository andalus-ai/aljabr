package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufKvProjectionLayoutTest {

    @Test
    void acceptsIndependentKeyAndValueProjections() {
        assertDoesNotThrow(() -> validate(keys(true, true), values(true, true),
                new int[] {2, 2}, new int[] {64, 64}, new boolean[] {false, false}));
    }

    @Test
    void acceptsKeyOnlyOwnerBecauseRunnerUsesKeyAsValue() {
        assertDoesNotThrow(() -> validate(keys(true), values(false),
                new int[] {2}, new int[] {64}, new boolean[] {false}));
    }

    @Test
    void acceptsSharedLayerWithCompatibleOwnerGeometry() {
        assertDoesNotThrow(() -> validate(keys(true, false), values(true, false),
                new int[] {2, 2}, new int[] {64, 64}, new boolean[] {false, false}));
    }

    @Test
    void rejectsValueProjectionWithoutKeyProjection() {
        assertThrows(IllegalArgumentException.class,
                () -> validate(keys(false), values(true),
                        new int[] {2}, new int[] {64}, new boolean[] {false}));
    }

    @Test
    void rejectsSharedLayerWithoutKeyOwner() {
        assertThrows(IllegalStateException.class,
                () -> validate(keys(false, false), values(false, false),
                        new int[] {2, 2}, new int[] {64, 64}, new boolean[] {false, false}));
    }

    @Test
    void rejectsSharedLayerWithIncompatibleOwnerGeometry() {
        assertThrows(IllegalArgumentException.class,
                () -> validate(keys(true, false), values(true, false),
                        new int[] {2, 4}, new int[] {64, 64}, new boolean[] {false, false}));
    }

    @Test
    void rejectsMismatchedLayoutArrayLengths() {
        assertThrows(IllegalArgumentException.class,
                () -> validate(keys(true), values(false, false),
                        new int[] {2}, new int[] {64}, new boolean[] {false}));
    }

    private static void validate(
            GGUFTensorInfo[] keys,
            GGUFTensorInfo[] values,
            int[] kvHeads,
            int[] headDimensions,
            boolean[] swa) {
        GgufKvProjectionLayout.validate(keys, values, kvHeads, headDimensions, swa);
    }

    private static GGUFTensorInfo[] keys(boolean... present) {
        GGUFTensorInfo[] result = new GGUFTensorInfo[present.length];
        for (int layer = 0; layer < present.length; layer++) {
            if (present[layer]) {
                result[layer] = tensor(layer, "attn_k");
            }
        }
        return result;
    }

    private static GGUFTensorInfo[] values(boolean... present) {
        GGUFTensorInfo[] result = new GGUFTensorInfo[present.length];
        for (int layer = 0; layer < present.length; layer++) {
            if (present[layer]) {
                result[layer] = tensor(layer, "attn_v");
            }
        }
        return result;
    }

    private static GGUFTensorInfo tensor(int layer, String projection) {
        return new GGUFTensorInfo("blk." + layer + "." + projection + ".weight",
                new long[] {1}, 0, 0L, Float.BYTES);
    }
}
