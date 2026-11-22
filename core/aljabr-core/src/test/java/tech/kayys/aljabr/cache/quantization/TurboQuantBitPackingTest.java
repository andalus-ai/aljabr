package tech.kayys.aljabr.cache.quantization;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class TurboQuantBitPackingTest {

    @Test
    void roundTripsEverySupportedWidthAndBoundaryLength() {
        Random random = new Random(0x545552424fL);
        int[] lengths = {0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 257};

        for (int bits = 1; bits <= 8; bits++) {
            int maximum = (1 << bits) - 1;
            for (int length : lengths) {
                int[] indices = new int[length];
                for (int i = 0; i < length; i++) {
                    indices[i] = random.nextInt(maximum + 1);
                }

                byte[] packed = TurboQuantBitPacking.packIndices(indices, bits);
                assertEquals(((long) length * bits + 7L) / 8L, packed.length);
                assertArrayEquals(indices,
                        TurboQuantBitPacking.unpackIndices(packed, length, bits),
                        "round trip failed for bits=" + bits + ", length=" + length);
            }
        }
    }

    @Test
    void packsAcrossByteBoundariesInLeastSignificantBitFirstOrder() {
        // Three 3-bit values occupy nine bits: 001, 010, 111 -> 0xD1, 0x01.
        byte[] packed = TurboQuantBitPacking.packIndices(new int[]{1, 2, 7}, 3);

        assertArrayEquals(new byte[]{(byte) 0xD1, 0x01}, packed);
        assertArrayEquals(new int[]{1, 2, 7},
                TurboQuantBitPacking.unpackIndices(packed, 3, 3));
    }

    @Test
    void usesOneBytePerIndexAtEightBits() {
        int[] indices = {0, 1, 127, 128, 254, 255};

        assertArrayEquals(new byte[]{0, 1, 127, (byte) 128, (byte) 254, (byte) 255},
                TurboQuantBitPacking.packIndices(indices, 8));
        assertArrayEquals(indices,
                TurboQuantBitPacking.unpackIndices(
                        TurboQuantBitPacking.packIndices(indices, 8), indices.length, 8));
    }

    @Test
    void rejectsIndicesThatDoNotFitTheRequestedWidth() {
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantBitPacking.packIndices(new int[]{-1}, 3));
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantBitPacking.packIndices(new int[]{8}, 3));
    }

    @Test
    void rejectsInvalidWidthsNullsAndNegativeElementCounts() {
        assertThrows(NullPointerException.class,
                () -> TurboQuantBitPacking.packIndices(null, 4));
        assertThrows(NullPointerException.class,
                () -> TurboQuantBitPacking.unpackIndices(null, 0, 4));
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantBitPacking.packIndices(new int[0], 0));
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantBitPacking.unpackIndices(new byte[0], 0, 9));
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantBitPacking.unpackIndices(new byte[0], -1, 4));
    }

    @Test
    void rejectsWrongPayloadLengthAndNonZeroPaddingBits() {
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantBitPacking.unpackIndices(new byte[]{0}, 1, 3));
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantBitPacking.unpackIndices(new byte[]{0, 0}, 1, 3));
        assertThrows(IllegalArgumentException.class,
                () -> TurboQuantBitPacking.unpackIndices(new byte[]{(byte) 0xF1}, 1, 3));
    }

    @Test
    void canonicalPaddingBitsAreZero() {
        byte[] packed = TurboQuantBitPacking.packIndices(new int[]{7, 7, 7}, 3);

        assertEquals(0x01, packed[1] & 0xFF);
        assertArrayEquals(new int[]{7, 7, 7},
                TurboQuantBitPacking.unpackIndices(packed, 3, 3));
    }
}
