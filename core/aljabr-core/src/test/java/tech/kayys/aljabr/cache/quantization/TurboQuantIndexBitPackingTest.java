package tech.kayys.aljabr.cache.quantization;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class TurboQuantIndexBitPackingTest {

    @Test
    void roundTripsEverySupportedBitWidthAcrossByteBoundaries() {
        Random random = new Random(0x5EEDL);
        for (int bits = 1; bits <= 8; bits++) {
            int maximum = (1 << bits) - 1;
            for (int count = 0; count <= 65; count++) {
                int[] indices = new int[count];
                for (int i = 0; i < count; i++) {
                    indices[i] = random.nextInt(maximum + 1);
                }

                byte[] packed = TurboQuantIndexBitPacking.pack(indices, bits);
                assertEquals((count * bits + 7) / 8, packed.length);
                assertArrayEquals(indices,
                        TurboQuantIndexBitPacking.unpack(packed, count, bits),
                        "bits=" + bits + ", count=" + count);
            }
        }
    }

    @Test
    void usesLeastSignificantBitFirstCanonicalLayout() {
        byte[] packed = TurboQuantIndexBitPacking.pack(new int[]{1, 2, 7}, 3);

        assertArrayEquals(new byte[]{(byte) 0xD1, 0x01}, packed);
        assertArrayEquals(new int[]{1, 2, 7},
                TurboQuantIndexBitPacking.unpack(packed, 3, 3));
    }

    @Test
    void computesPayloadSizeWithoutIntegerOverflow() {
        assertEquals(0, TurboQuantIndexBitPacking.byteCount(0, 3));
        assertEquals(1, TurboQuantIndexBitPacking.byteCount(1, 1));
        assertEquals(2, TurboQuantIndexBitPacking.byteCount(3, 3));
        assertEquals(3, TurboQuantIndexBitPacking.byteCount(5, 4));
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantIndexBitPacking.byteCount(Integer.MAX_VALUE, 8));
    }

    @Test
    void rejectsOutOfRangeIndicesAndInvalidWidths() {
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantIndexBitPacking.pack(new int[]{-1}, 3));
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantIndexBitPacking.pack(new int[]{8}, 3));
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantIndexBitPacking.pack(new int[0], 0));
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantIndexBitPacking.unpack(new byte[0], 0, 9));
    }

    @Test
    void rejectsIncorrectPayloadLengthAndNonCanonicalPadding() {
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantIndexBitPacking.unpack(new byte[]{0}, 2, 4));
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantIndexBitPacking.unpack(new byte[]{(byte) 0x80}, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantIndexBitPacking.unpack(new byte[0], -1, 2));
    }

    @Test
    void rejectsNullArrays() {
        assertThrows(NullPointerException.class,
                () -> TurboQuantIndexBitPacking.pack(null, 2));
        assertThrows(NullPointerException.class,
                () -> TurboQuantIndexBitPacking.unpack(null, 0, 2));
    }
}
