package tech.kayys.aljabr.cache;

/**
 * Logical tensor geometry for one borrowed KV buffer region.
 *
 * <p>Dimensions and strides are expressed in elements, not bytes. The
 * descriptor is independent of the allocation's memory domain and backend.
 * Consumers must validate the region's element encoding and byte length
 * against this geometry before interpreting the buffer.</p>
 *
 * @param sequenceLength visible sequence positions
 * @param kvHeads number of key or value heads
 * @param headDimension elements per head
 * @param sequenceStride distance between adjacent sequence positions
 * @param headStride distance between adjacent heads
 * @param elementStride distance between adjacent elements within a head
 */
public record KVCacheBufferLayout(
        int sequenceLength,
        int kvHeads,
        int headDimension,
        long sequenceStride,
        long headStride,
        long elementStride) {

    public KVCacheBufferLayout {
        if (sequenceLength < 0 || kvHeads <= 0 || headDimension <= 0) {
            throw new IllegalArgumentException("Invalid KV buffer dimensions");
        }
        if (sequenceStride <= 0 || headStride <= 0 || elementStride <= 0) {
            throw new IllegalArgumentException("KV buffer strides must be positive");
        }
        long minimumHeadStride = Math.multiplyExact((long) headDimension, elementStride);
        long minimumSequenceStride = Math.multiplyExact((long) kvHeads, headStride);
        if (headStride < minimumHeadStride || sequenceStride < minimumSequenceStride) {
            throw new IllegalArgumentException("KV buffer strides overlap logical elements");
        }
    }

    /** Standard contiguous [sequence, head, dimension] layout. */
    public static KVCacheBufferLayout sequenceMajor(
            int sequenceLength, int kvHeads, int headDimension) {
        long headStride = headDimension;
        long sequenceStride = Math.multiplyExact((long) kvHeads, headStride);
        return new KVCacheBufferLayout(sequenceLength, kvHeads, headDimension,
                sequenceStride, headStride, 1);
    }

    /** Number of logical elements represented by this layout. */
    public long elementCount() {
        return Math.multiplyExact((long) sequenceLength,
                Math.multiplyExact((long) kvHeads, headDimension));
    }

    /** Minimum byte length needed to represent this layout for an element width. */
    public long minimumByteLength(int elementBytes) {
        if (elementBytes <= 0) {
            throw new IllegalArgumentException("elementBytes must be positive");
        }
        if (sequenceLength == 0) return 0;
        long lastElement = Math.addExact(
                Math.addExact(
                        Math.multiplyExact((long) (sequenceLength - 1), sequenceStride),
                        Math.multiplyExact((long) (kvHeads - 1), headStride)),
                Math.multiplyExact((long) (headDimension - 1), elementStride));
        return Math.multiplyExact(Math.addExact(lastElement, 1), elementBytes);
    }
}
