package tech.kayys.aljabr.cache;

import java.util.Objects;

/**
 * Request/sequence-scoped view over KV storage with an explicit committed
 * sequence length.
 *
 * <p>The session coordinates logical visibility of positions while the
 * supplied {@link KVCacheStorage} owns only tensor bytes. Writes may target
 * uncommitted positions (for prefill or speculative decode), but reads are
 * restricted to the committed prefix. The caller must commit only after all
 * required layer writes for those positions have completed.</p>
 *
 * <p>This class does not own or close the storage, allocate it, or provide
 * cross-thread synchronization. A session should normally be confined to one
 * active model sequence.</p>
 */
public final class KVCacheSession {

    private final KVCacheStorage storage;
    private int committedLength;

    public KVCacheSession(KVCacheStorage storage) {
        this.storage = Objects.requireNonNull(storage, "storage");
    }

    /** Returns the immutable logical cache geometry. */
    public KVCacheLayout layout() {
        return storage.layout();
    }

    /** Number of sequence positions visible to attention reads. */
    public int committedLength() {
        return committedLength;
    }

    /** Writes one layer/position, including positions not yet committed. */
    public void write(int layer, int position,
            float[] key, int keyOffset, float[] value, int valueOffset) {
        checkWritablePosition(position);
        storage.write(layer, position, key, keyOffset, value, valueOffset);
    }

    /** Writes a contiguous layer range, commonly used by prefill. */
    public void writeRange(int layer, int firstPosition, int tokenCount,
            float[] keys, int keysOffset, float[] values, int valuesOffset) {
        checkWritableRange(firstPosition, tokenCount);
        storage.writeRange(layer, firstPosition, tokenCount,
                keys, keysOffset, values, valuesOffset);
    }

    /** Reads one committed K position. */
    public void readKey(int layer, int position, float[] destination, int destinationOffset) {
        checkReadableRange(position, 1);
        storage.readKey(layer, position, destination, destinationOffset);
    }

    /** Reads one committed V position. */
    public void readValue(int layer, int position, float[] destination, int destinationOffset) {
        checkReadableRange(position, 1);
        storage.readValue(layer, position, destination, destinationOffset);
    }

    /** Reads a contiguous committed K range. */
    public void readKeyRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        checkReadableRange(firstPosition, tokenCount);
        storage.readKeyRange(layer, firstPosition, tokenCount, destination, destinationOffset);
    }

    /** Reads a contiguous committed V range. */
    public void readValueRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        checkReadableRange(firstPosition, tokenCount);
        storage.readValueRange(layer, firstPosition, tokenCount, destination, destinationOffset);
    }

    /**
     * Reads K data for the in-progress position of a layer.
     *
     * <p>Unlike ordinary reads, this permits the range to extend one position
     * beyond the committed prefix. It is intended for single-sequence decoder
     * execution after that layer has written the current token's K/V values.
     * The caller must not use it to expose partially written positions to
     * another layer, sequence, or thread.</p>
     */
    public void readKeyRangeForAttention(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        checkAttentionReadRange(firstPosition, tokenCount);
        storage.readKeyRange(layer, firstPosition, tokenCount, destination, destinationOffset);
    }

    /** See {@link #readKeyRangeForAttention(int, int, int, float[], int)}. */
    public void readValueRangeForAttention(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        checkAttentionReadRange(firstPosition, tokenCount);
        storage.readValueRange(layer, firstPosition, tokenCount, destination, destinationOffset);
    }

    private void checkAttentionReadRange(int firstPosition, int tokenCount) {
        int visibleLimit = committedLength + 1;
        if (firstPosition < 0 || tokenCount < 0
                || firstPosition > visibleLimit
                || tokenCount > visibleLimit - firstPosition) {
            throw new IndexOutOfBoundsException(
                    "Attention read exceeds committed prefix plus current decode position");
        }
    }

    /**
     * Publishes positions up to {@code newCommittedLength}.
     *
     * <p>The caller is responsible for ensuring every layer has been written
     * for the newly committed positions before invoking this method.</p>
     */
    public void commit(int newCommittedLength) {
        if (newCommittedLength < committedLength
                || newCommittedLength > layout().maxSequenceLength()) {
            throw new IllegalArgumentException(
                    "Committed length must be between current length and cache capacity");
        }
        committedLength = newCommittedLength;
    }

    /**
     * Discards the suffix beginning at {@code newLength}, clearing its storage
     * across all layers and reducing the visible sequence length.
     */
    public void truncate(int newLength) {
        if (newLength < 0 || newLength > committedLength) {
            throw new IllegalArgumentException(
                    "Truncation length must be within the committed sequence");
        }
        storage.truncate(newLength);
        committedLength = newLength;
    }

    /** Clears all storage and resets the visible sequence length. */
    public void reset() {
        storage.clear();
        committedLength = 0;
    }

    private void checkWritablePosition(int position) {
        if (position < 0 || position >= layout().maxSequenceLength()) {
            throw new IndexOutOfBoundsException("KV write position exceeds cache capacity");
        }
    }

    private void checkWritableRange(int firstPosition, int tokenCount) {
        if (firstPosition < 0 || tokenCount < 0
                || firstPosition > layout().maxSequenceLength()
                || tokenCount > layout().maxSequenceLength() - firstPosition) {
            throw new IndexOutOfBoundsException("KV write range exceeds cache capacity");
        }
    }

    private void checkReadableRange(int firstPosition, int tokenCount) {
        if (firstPosition < 0 || tokenCount < 0
                || firstPosition > committedLength
                || tokenCount > committedLength - firstPosition) {
            throw new IndexOutOfBoundsException("KV read range exceeds committed sequence length");
        }
    }
}
