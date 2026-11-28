package tech.kayys.aljabr.cache;

/** Element encodings supported by backend-aware KV cache region descriptors. */
public enum KVCacheElementType {
    FLOAT32(4),
    FLOAT16(2),
    BFLOAT16(2),
    INT8(1),
    UINT8(1),
    FP8_E4M3(1),
    FP8_E5M2(1);

    private final int bytes;

    KVCacheElementType(int bytes) {
        this.bytes = bytes;
    }

    /** Number of bytes occupied by one encoded element. */
    public int bytes() {
        return bytes;
    }
}
