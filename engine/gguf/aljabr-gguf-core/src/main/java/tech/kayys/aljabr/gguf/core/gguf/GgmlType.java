package tech.kayys.aljabr.ml.gguf;

/**
 * GGUF tensor data types.
 * @author bhangun
 */
public enum GgmlType {
    F32(0, 1, 4),
    F16(1, 1, 2),
    Q4_0(2, 32, 18),
    Q4_1(3, 32, 20),
    Q5_0(6, 32, 22),
    Q5_1(7, 32, 24),
    Q8_0(8, 32, 34),
    Q8_1(9, 32, 36),
    Q2_K(10, 256, 84),
    Q3_K(11, 256, 110),
    Q4_K(12, 256, 144),
    Q5_K(13, 256, 176),
    Q6_K(14, 256, 210),
    Q8_K(15, 256, 292),
    I8(24, 1, 1),
    I16(25, 1, 2),
    I32(26, 1, 4),
    I64(27, 1, 8),
    F64(28, 1, 8),
    IQ1_M(29, 256, 56),
    BF16(30, 1, 2),
    Q4_0_4_4(31, 0, 0),
    Q4_0_4_8(32, 0, 0),
    Q4_0_8_8(33, 0, 0),
    TQ1_0(34, 256, 54),
    TQ2_0(35, 256, 66),
    IQ4_NL_4_4(36, 0, 0),
    IQ4_NL_4_8(37, 0, 0),
    IQ4_NL_8_8(38, 0, 0),
    MXFP4(39, 32, 17),
    NVFP4(40, 64, 36),
    Q1_0(41, 128, 18);

    public final int id;
    public final int blockSize;
    public final int typeSize;

    GgmlType(int id, int blockSize, int typeSize) {
        this.id = id;
        this.blockSize = blockSize;
        this.typeSize = typeSize;
    }

    public static GgmlType fromId(int id) {
        for (GgmlType t : values()) {
            if (t.id == id) return t;
        }
        throw new IllegalArgumentException("Unknown GGML type id: " + id);
    }

    public long bytesFor(long numElements) {
        if (blockSize <= 0 || typeSize <= 0) {
            throw new IllegalArgumentException("GGML type id " + id + " is not loadable from GGUF data");
        }
        if (numElements % blockSize != 0) {
            throw new IllegalArgumentException("Not multiple of block size");
        }
        return (numElements / blockSize) * typeSize;
    }
}
