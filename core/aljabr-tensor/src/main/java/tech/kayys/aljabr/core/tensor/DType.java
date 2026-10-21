package tech.kayys.aljabr.core.tensor;
/**
 * Enumeration of d values.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core enum operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public enum DType {
    // ── Floating point ────────────────────────────────────────────────────────
    F32(1, 4),
    F16(1, 2),
    BF16(1, 2),

    // ── Integer ───────────────────────────────────────────────────────────────
    I32(1, 4),
    I8(1, 1),
    INT8(1, 1),

    // ── Block Quantized (GGML style) ──────────────────────────────────────────

    /**
     * Q8_0: 32-element block.
     * - 32 bytes of 8-bit weights.
     * - 2 bytes for the FP16 scale factor.
     * Total: 34 bytes per block.
     */
    Q8_0(32, 34),

    /**
     * Q4_K: 256-element block.
     * - 128 bytes of 4-bit weights.
     * - 12 bytes of 6-bit scales.
     * - 2 bytes of FP16 super-scale.
     * - 2 bytes of FP16 super-min.
     * Total: 144 bytes per block.
     */
    Q4_K(256, 144),

    /**
     * Q4_0: 32-element block.
     * - 16 bytes of 4-bit weights.
     * - 2 bytes for the FP16 scale factor.
     * Total: 18 bytes per block.
     */
    Q4_0(32, 18);

    private final int blockSize;
    private final int blockByteSize;

    DType(int blockSize, int blockByteSize) {
        this.blockSize = blockSize;
        this.blockByteSize = blockByteSize;
    }

    /**
     * Number of elements grouped together in one storage block.
     * For unquantized types (e.g. F32), this is always 1.
     */
    public int blockSize() {
        return blockSize;
    }

    /**
     * Number of bytes required to store one block of elements.
     * For unquantized types, this is simply the byte size of the primitive.
     */
    public int blockByteSize() {
        return blockByteSize;
    }

    /**
     * Returns true if this data type uses block quantization (i.e. blockSize > 1).
     */
    public boolean isQuantized() {
        return blockSize > 1;
    }

    /**
     * Computes the total memory footprint in bytes required to store a tensor
     * with the given number of elements in this data type.
     * 
     * @param numElements total elements
     * @return memory required in bytes
     * @throws IllegalArgumentException if numElements is not a multiple of
     *                                  blockSize
     */
    public long memoryFootprintBytes(long numElements) {
        if (numElements % blockSize != 0) {
            throw new IllegalArgumentException(
                    "Tensor size (" + numElements + ") is not a multiple of block size ("
                            + blockSize + ") for DType " + this);
        }
        return (numElements / blockSize) * blockByteSize;
    }
}
