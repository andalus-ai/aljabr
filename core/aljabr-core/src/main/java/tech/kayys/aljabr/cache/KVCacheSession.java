package tech.kayys.aljabr.cache;

import java.util.Arrays;
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
    private final boolean[] currentPositionWritten;
    private int committedLength;

    public KVCacheSession(KVCacheStorage storage) {
        this.storage = Objects.requireNonNull(storage, "storage");
        this.currentPositionWritten = new boolean[storage.layout().layerCount()];
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
        markCurrentPositionWritten(layer, position);
    }

    /** Writes a contiguous layer range, commonly used by prefill. */
    public void writeRange(int layer, int firstPosition, int tokenCount,
            float[] keys, int keysOffset, float[] values, int valuesOffset) {
        checkWritableRange(firstPosition, tokenCount);
        storage.writeRange(layer, firstPosition, tokenCount,
                keys, keysOffset, values, valuesOffset);
        if (tokenCount > 0 && firstPosition <= committedLength
                && committedLength < firstPosition + tokenCount) {
            currentPositionWritten[checkedLayer(layer)] = true;
        }
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
     * <p>The current position is visible only after this session has
     * successfully written that position for the requested layer. This
     * single-sequence decoder API must not expose partial writes to another
     * layer, sequence, or thread.</p>
     */
    public void readKeyRangeForAttention(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        checkAttentionReadRange(layer, firstPosition, tokenCount);
        storage.readKeyRange(layer, firstPosition, tokenCount, destination, destinationOffset);
    }

    /** See {@link #readKeyRangeForAttention(int, int, int, float[], int)}. */
    public void readValueRangeForAttention(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        checkAttentionReadRange(layer, firstPosition, tokenCount);
        storage.readValueRange(layer, firstPosition, tokenCount, destination, destinationOffset);
    }

    /**
     * Delivers the requested attention-visible range as contiguous K/V blocks.
     *
     * <p>Storage implementations with {@link KVCacheAttentionBlockReader}
     * support can decode their native blocks once and deliver both tensors
     * together. Other storage implementations use a single range read for K
     * and V as a compatibility fallback. The callback arrays are ephemeral
     * and read-only; copy them if they must outlive the callback.</p>
     */
    public void readBlocksForAttention(
            int layer,
            int firstPosition,
            int tokenCount,
            KVCacheAttentionBlockConsumer consumer) {
        Objects.requireNonNull(consumer, "consumer");
        checkAttentionReadRange(layer, firstPosition, tokenCount);
        if (storage instanceof KVCacheAttentionBlockReader blockReader) {
            blockReader.readAttentionBlocks(layer, firstPosition, tokenCount, consumer);
            return;
        }
        int elements = Math.multiplyExact(tokenCount, layout().layerStride(layer));
        float[] keys = new float[elements];
        float[] values = new float[elements];
        storage.readKeyRange(layer, firstPosition, tokenCount, keys, 0);
        storage.readValueRange(layer, firstPosition, tokenCount, values, 0);
        consumer.accept(firstPosition, tokenCount, keys, values, 0);
    }

    private void checkAttentionReadRange(int layer, int firstPosition, int tokenCount) {
        int checkedLayer = checkedLayer(layer);
        int visibleLimit = committedLength + 1;
        if (firstPosition < 0 || tokenCount < 0
                || firstPosition > visibleLimit
                || tokenCount > visibleLimit - firstPosition) {
            throw new IndexOutOfBoundsException(
                    "Attention read exceeds committed prefix plus current decode position");
        }
        if (firstPosition + tokenCount > committedLength
                && !currentPositionWritten[checkedLayer]) {
            throw new IllegalStateException(
                    "Current decode position has not been written for layer " + checkedLayer);
        }
    }

    private void markCurrentPositionWritten(int layer, int position) {
        if (position == committedLength) {
            currentPositionWritten[checkedLayer(layer)] = true;
        }
    }

    private int checkedLayer(int layer) {
        storage.layout().layerStride(layer);
        return layer;
    }

    private void clearCurrentPositionWrites() {
        Arrays.fill(currentPositionWritten, false);
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
        clearCurrentPositionWrites();
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
        clearCurrentPositionWrites();
    }

    /** Clears all storage and resets the visible sequence length. */
    public void reset() {
        storage.clear();
        committedLength = 0;
        clearCurrentPositionWrites();
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
