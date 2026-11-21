package tech.kayys.aljabr.core.tensor;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class TensorLayoutTest {

    @Test
    void createsCanonicalRowMajorLayout() {
        TensorLayout layout = TensorLayout.contiguous(new Shape(2, 3, 4));

        assertArrayEquals(new long[] {12, 4, 1}, layout.strides());
        assertEquals(0, layout.storageOffset());
        assertEquals(24, layout.requiredStorageElements());
        assertTrue(layout.isContiguous());
    }

    @Test
    void supportsScalarAndEmptyShapes() {
        TensorLayout scalar = TensorLayout.contiguous(new Shape());
        assertArrayEquals(new long[0], scalar.strides());
        assertEquals(1, scalar.requiredStorageElements());
        assertTrue(scalar.isContiguous());

        TensorLayout empty = TensorLayout.contiguous(new Shape(2, 0, 3));
        assertArrayEquals(new long[] {0, 3, 1}, empty.strides());
        assertEquals(0, empty.requiredStorageElements());
        assertTrue(empty.isContiguous());
    }

    @Test
    void identifiesTransposedAndBroadcastLayoutsAsNonContiguous() {
        TensorLayout transposed = new TensorLayout(new Shape(3, 2), new long[] {1, 3}, 0);
        assertFalse(transposed.isContiguous());
        assertEquals(6, transposed.requiredStorageElements());

        TensorLayout broadcast = new TensorLayout(new Shape(2, 3), new long[] {0, 1}, 0);
        assertFalse(broadcast.isContiguous());
        assertEquals(3, broadcast.requiredStorageElements());
    }

    @Test
    void accountsForStorageOffsetAndStridedReachableRange() {
        TensorLayout layout = new TensorLayout(new Shape(2, 2), new long[] {4, 1}, 3);

        assertEquals(9, layout.requiredStorageElements());
        assertFalse(layout.isContiguous());
    }

    @Test
    void defensivelyCopiesStrideInputAndOutput() {
        long[] strides = {3, 1};
        TensorLayout layout = new TensorLayout(new Shape(2, 3), strides, 0);
        strides[0] = 99;

        long[] returned = layout.strides();
        returned[0] = 77;

        assertArrayEquals(new long[] {3, 1}, layout.strides());
    }

    @Test
    void rejectsRankMismatchNegativeStrideAndNegativeOffset() {
        assertThrows(IllegalArgumentException.class,
                () -> new TensorLayout(new Shape(2, 3), new long[] {1}, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new TensorLayout(new Shape(2), new long[] {-1}, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new TensorLayout(new Shape(2), new long[] {1}, -1));
    }

    @Test
    void rejectsAddressRangeOverflow() {
        assertThrows(IllegalArgumentException.class,
                () -> new TensorLayout(new Shape(Long.MAX_VALUE, 2), new long[] {2, 1}, 0));
        assertThrows(IllegalArgumentException.class,
                () -> new TensorLayout(new Shape(2), new long[] {1}, Long.MAX_VALUE));
    }

    @Test
    void equalityUsesLogicalShapeStridesAndOffset() {
        TensorLayout first = new TensorLayout(new Shape(2, 3), new long[] {3, 1}, 2);
        TensorLayout same = new TensorLayout(new Shape(2, 3), new long[] {3, 1}, 2);
        TensorLayout differentOffset = new TensorLayout(new Shape(2, 3), new long[] {3, 1}, 3);

        assertEquals(first, same);
        assertEquals(first.hashCode(), same.hashCode());
        assertNotEquals(first, differentOffset);
    }
}
