package tech.kayys.aljabr.cache.quantization;

import java.util.Objects;

/**
 * Canonical little-endian bit packing for small unsigned quantization indices.
 *
 * <p>Index bits are written least-significant bit first, and the first index
 * occupies the least-significant bits of the first byte. The final byte is
 * zero-padded in its unused high bits. This representation supports widths
 * that do not divide eight, such as 2, 3, 5, and 6 bits.</p>
 *
 * <p>This class only packs integer indices. It does not define codebooks,
 * vector norms, block metadata, or a complete KV-cache codec.</p>
 */
public final class TurboQuantIndexBitPacking {

    private TurboQuantIndexBitPacking() {
    }

    /**
     * Packs unsigned indices into a compact byte array.
     *
     * @param indices quantized indices, each in {@code [0, 2^bits)}
     * @param bits number of bits per index, from 1 through 8
     * @return canonical packed representation
     */
    public static byte[] pack(int[] indices, int bits) {
        Objects.requireNonNull(indices, "indices");
        validateBitWidth(bits);
        int byteCount = byteCount(indices.length, bits);
        byte[] packed = new byte[byteCount];
        int mask = (1 << bits) - 1;

        long bitOffset = 0;
        for (int index : indices) {
            if (index < 0 || index > mask) {
                throw new IllegalArgumentException(
                        "index must be between 0 and " + mask + ": " + index);
            }
            for (int bit = 0; bit < bits; bit++, bitOffset++) {
                if ((index & (1 << bit)) != 0) {
                    int byteIndex = (int) (bitOffset >>> 3);
                    int bitInByte = (int) (bitOffset & 7);
                    packed[byteIndex] |= (byte) (1 << bitInByte);
                }
            }
        }
        return packed;
    }

    /**
     * Unpacks exactly {@code elementCount} unsigned indices.
     *
     * <p>The payload length must be exact and unused high bits in the final
     * byte must be zero. Rejecting non-canonical padding prevents multiple
     * byte representations of the same logical index sequence.</p>
     *
     * @param packed canonical packed bytes
     * @param elementCount number of indices to decode
     * @param bits number of bits per index, from 1 through 8
     * @return decoded indices
     */
    public static int[] unpack(byte[] packed, int elementCount, int bits) {
        Objects.requireNonNull(packed, "packed");
        validateBitWidth(bits);
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount must not be negative");
        }
        int expectedLength = byteCount(elementCount, bits);
        if (packed.length != expectedLength) {
            throw new IllegalArgumentException(
                    "packed length must be " + expectedLength + " bytes, got " + packed.length);
        }

        int usedBitsInLastByte = (int) (((long) elementCount * bits) & 7);
        if (usedBitsInLastByte != 0 && packed.length > 0) {
            int unusedMask = 0xFF << usedBitsInLastByte;
            if ((packed[packed.length - 1] & 0xFF & unusedMask) != 0) {
                throw new IllegalArgumentException("unused high bits in final byte must be zero");
            }
        }

        int[] indices = new int[elementCount];
        long bitOffset = 0;
        for (int i = 0; i < elementCount; i++) {
            int value = 0;
            for (int bit = 0; bit < bits; bit++, bitOffset++) {
                int byteIndex = (int) (bitOffset >>> 3);
                int bitInByte = (int) (bitOffset & 7);
                if (((packed[byteIndex] >>> bitInByte) & 1) != 0) {
                    value |= 1 << bit;
                }
            }
            indices[i] = value;
        }
        return indices;
    }

    /** Returns the exact number of bytes required for the supplied shape. */
    public static int byteCount(int elementCount, int bits) {
        validateBitWidth(bits);
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount must not be negative");
        }
        long byteCount = (((long) elementCount * bits) + 7) >>> 3;
        if (byteCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("packed payload exceeds maximum Java array size");
        }
        return (int) byteCount;
    }

    private static void validateBitWidth(int bits) {
        if (bits < 1 || bits > 8) {
            throw new IllegalArgumentException("bits must be between 1 and 8");
        }
    }
}
