package tech.kayys.aljabr.cache.quantization;

import java.util.Objects;

/**
 * Compact little-endian bit packing for unsigned TurboQuant scalar-codebook indices.
 *
 * <p>Indices are written least-significant bit first, and each index occupies
 * exactly the requested number of bits. The bitstream is therefore independent
 * of byte boundaries and supports widths from one through eight bits. Unused
 * high bits in the final byte are zero, giving each encoded sequence a
 * canonical representation.</p>
 *
 * <p>This class only packs index values. It does not encode codebook identity,
 * vector norms, transform seeds, or KV-cache block metadata.</p>
 */
public final class TurboQuantBitPacking {

    private TurboQuantBitPacking() {
    }

    /**
     * Packs unsigned indices into a dense byte array.
     *
     * @param indices codebook indices, each in {@code [0, 2^bits)}
     * @param bits number of bits per index, from one through eight
     * @return packed bytes using least-significant-bit-first ordering
     */
    public static byte[] packIndices(int[] indices, int bits) {
        Objects.requireNonNull(indices, "indices");
        validateBits(bits);

        long totalBits = (long) indices.length * bits;
        int byteCount = checkedByteCount(totalBits);
        byte[] packed = new byte[byteCount];
        int maximum = (1 << bits) - 1;

        for (int i = 0; i < indices.length; i++) {
            int value = indices[i];
            if (value < 0 || value > maximum) {
                throw new IllegalArgumentException(
                        "index at position " + i + " does not fit in " + bits + " bits: " + value);
            }

            long bitPosition = (long) i * bits;
            int byteIndex = (int) (bitPosition >>> 3);
            int bitOffset = (int) (bitPosition & 7);
            int remaining = bits;
            int valueShift = 0;

            while (remaining > 0) {
                int chunkBits = Math.min(remaining, 8 - bitOffset);
                int chunkMask = (1 << chunkBits) - 1;
                int chunk = (value >>> valueShift) & chunkMask;
                packed[byteIndex] = (byte) (packed[byteIndex] | (chunk << bitOffset));

                remaining -= chunkBits;
                valueShift += chunkBits;
                byteIndex++;
                bitOffset = 0;
            }
        }
        return packed;
    }

    /**
     * Unpacks exactly {@code elementCount} unsigned indices from a packed byte array.
     *
     * <p>The payload length must match the requested element count and bit width.
     * Non-zero unused high bits in the final byte are rejected.</p>
     *
     * @param packed packed index bytes
     * @param elementCount number of indices to decode
     * @param bits number of bits per index, from one through eight
     * @return decoded indices
     */
    public static int[] unpackIndices(byte[] packed, int elementCount, int bits) {
        Objects.requireNonNull(packed, "packed");
        validateBits(bits);
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount must not be negative");
        }

        long totalBits = (long) elementCount * bits;
        int expectedByteCount = checkedByteCount(totalBits);
        if (packed.length != expectedByteCount) {
            throw new IllegalArgumentException(
                    "packed length must be " + expectedByteCount + " bytes for "
                            + elementCount + " indices at " + bits + " bits");
        }

        int usedBitsInLastByte = (int) (totalBits & 7);
        if (usedBitsInLastByte != 0 && packed.length > 0) {
            int unusedMask = 0xFF ^ ((1 << usedBitsInLastByte) - 1);
            if (((packed[packed.length - 1] & 0xFF) & unusedMask) != 0) {
                throw new IllegalArgumentException("unused padding bits must be zero");
            }
        }

        int[] indices = new int[elementCount];
        for (int i = 0; i < elementCount; i++) {
            long bitPosition = (long) i * bits;
            int byteIndex = (int) (bitPosition >>> 3);
            int bitOffset = (int) (bitPosition & 7);
            int remaining = bits;
            int valueShift = 0;
            int value = 0;

            while (remaining > 0) {
                int chunkBits = Math.min(remaining, 8 - bitOffset);
                int chunkMask = (1 << chunkBits) - 1;
                int chunk = ((packed[byteIndex] & 0xFF) >>> bitOffset) & chunkMask;
                value |= chunk << valueShift;

                remaining -= chunkBits;
                valueShift += chunkBits;
                byteIndex++;
                bitOffset = 0;
            }
            indices[i] = value;
        }
        return indices;
    }

    private static void validateBits(int bits) {
        if (bits < 1 || bits > 8) {
            throw new IllegalArgumentException("bits must be between 1 and 8");
        }
    }

    private static int checkedByteCount(long totalBits) {
        long byteCount = (totalBits + 7L) >>> 3;
        if (byteCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("packed payload exceeds maximum Java array size");
        }
        return (int) byteCount;
    }
}
