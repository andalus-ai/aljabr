package tech.kayys.aljabr.cache;

import java.util.Arrays;
import java.util.Objects;

/**
 * Heap-backed KV storage that grows its sequence capacity on demand.
 *
 * <p>The logical maximum is fixed by {@link KVCacheLayout}, but physical
 * storage is allocated only as positions are written. This avoids eagerly
 * reserving the full context window for short prompts while preserving the
 * same {@link KVCacheStorage} contract used by fixed-capacity providers.</p>
 *
 * <p>Storage is sequence-major within each layer. As with other storage
 * implementations, callers must externally synchronize concurrent access to
 * the same layer and position.</p>
 */
public final class GrowingHeapKVCacheStorage implements KVCacheStorage {
    private static final int DEFAULT_INITIAL_POSITIONS = 16;

    private final KVCacheLayout layout;
    private float[][] keys;
    private float[][] values;
    private int allocatedPositions;

    /** Creates storage with a small initial allocation, capped by the logical maximum. */
    public GrowingHeapKVCacheStorage(KVCacheLayout layout) {
        this(layout, DEFAULT_INITIAL_POSITIONS);
    }

    /**
     * Creates storage with the requested initial physical capacity.
     *
     * @param layout logical cache geometry and maximum sequence length
     * @param initialPositions initial physical sequence capacity, from zero to
     *                         {@code layout.maxSequenceLength()}
     */
    public GrowingHeapKVCacheStorage(KVCacheLayout layout, int initialPositions) {
        this.layout = Objects.requireNonNull(layout, "layout");
        if (initialPositions < 0 || initialPositions > layout.maxSequenceLength()) {
            throw new IllegalArgumentException("initialPositions must be within the logical cache capacity");
        }
        this.keys = new float[layout.layerCount()][];
        this.values = new float[layout.layerCount()][];
        this.allocatedPositions = initialPositions;
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            int elements = Math.multiplyExact(initialPositions, layout.layerStride(layer));
            keys[layer] = new float[elements];
            values[layer] = new float[elements];
        }
    }

    @Override
    public KVCacheLayout layout() {
        return layout;
    }

    /** Number of sequence positions currently backed by physical heap storage. */
    public int allocatedPositions() {
        ensureOpen();
        return allocatedPositions;
    }

    @Override
    public void write(int layer, int position,
            float[] key, int keyOffset,
            float[] value, int valueOffset) {
        ensureOpen();
        int stride = checkedLogicalPosition(layer, position);
        checkArrayRange(key, keyOffset, stride, "key");
        checkArrayRange(value, valueOffset, stride, "value");
        ensureCapacity(position + 1);
        int destinationOffset = Math.multiplyExact(position, stride);
        System.arraycopy(key, keyOffset, keys[layer], destinationOffset, stride);
        System.arraycopy(value, valueOffset, values[layer], destinationOffset, stride);
    }

    @Override
    public void readKey(int layer, int position, float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedAllocatedPosition(layer, position);
        checkArrayRange(destination, destinationOffset, stride, "destination");
        System.arraycopy(keys[layer], Math.multiplyExact(position, stride),
                destination, destinationOffset, stride);
    }

    @Override
    public void readValue(int layer, int position, float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedAllocatedPosition(layer, position);
        checkArrayRange(destination, destinationOffset, stride, "destination");
        System.arraycopy(values[layer], Math.multiplyExact(position, stride),
                destination, destinationOffset, stride);
    }

    @Override
    public void writeRange(int layer, int firstPosition, int tokenCount,
            float[] sourceKeys, int keysOffset, float[] sourceValues, int valuesOffset) {
        ensureOpen();
        int stride = checkedLogicalRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(sourceKeys, keysOffset, elements, "keys");
        checkArrayRange(sourceValues, valuesOffset, elements, "values");
        ensureCapacity(Math.addExact(firstPosition, tokenCount));
        int destinationOffset = Math.multiplyExact(firstPosition, stride);
        System.arraycopy(sourceKeys, keysOffset, keys[layer], destinationOffset, elements);
        System.arraycopy(sourceValues, valuesOffset, values[layer], destinationOffset, elements);
    }

    @Override
    public void readKeyRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedAllocatedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(destination, destinationOffset, elements, "destination");
        System.arraycopy(keys[layer], Math.multiplyExact(firstPosition, stride),
                destination, destinationOffset, elements);
    }

    @Override
    public void readValueRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedAllocatedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(destination, destinationOffset, elements, "destination");
        System.arraycopy(values[layer], Math.multiplyExact(firstPosition, stride),
                destination, destinationOffset, elements);
    }

    @Override
    public void clearLayer(int layer) {
        ensureOpen();
        checkLayer(layer);
        Arrays.fill(keys[layer], 0.0f);
        Arrays.fill(values[layer], 0.0f);
    }

    @Override
    public void clear() {
        ensureOpen();
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            clearLayer(layer);
        }
    }

    @Override
    public void truncate(int firstPosition) {
        ensureOpen();
        if (firstPosition < 0 || firstPosition > layout.maxSequenceLength()) {
            throw new IllegalArgumentException("firstPosition must be within cache capacity");
        }
        if (firstPosition >= allocatedPositions) {
            return;
        }
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            int offset = Math.multiplyExact(firstPosition, layout.layerStride(layer));
            Arrays.fill(keys[layer], offset, keys[layer].length, 0.0f);
            Arrays.fill(values[layer], offset, values[layer].length, 0.0f);
        }
    }

    @Override
    public void close() {
        keys = null;
        values = null;
        allocatedPositions = 0;
    }

    private void ensureCapacity(int requiredPositions) {
        if (requiredPositions > layout.maxSequenceLength()) {
            throw new IndexOutOfBoundsException("KV cache write exceeds logical sequence capacity");
        }
        if (requiredPositions <= allocatedPositions) {
            return;
        }
        int maximum = layout.maxSequenceLength();
        int doubled = allocatedPositions == 0 ? 1
                : allocatedPositions > maximum / 2 ? maximum : allocatedPositions * 2;
        int nextCapacity = Math.min(maximum, Math.max(requiredPositions, doubled));
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            int elements = Math.multiplyExact(nextCapacity, layout.layerStride(layer));
            keys[layer] = Arrays.copyOf(keys[layer], elements);
            values[layer] = Arrays.copyOf(values[layer], elements);
        }
        allocatedPositions = nextCapacity;
    }

    private int checkedLogicalPosition(int layer, int position) {
        checkLayer(layer);
        if (position < 0 || position >= layout.maxSequenceLength()) {
            throw new IndexOutOfBoundsException("KV cache position exceeds logical capacity: " + position);
        }
        return layout.layerStride(layer);
    }

    private int checkedAllocatedPosition(int layer, int position) {
        int stride = checkedLogicalPosition(layer, position);
        if (position >= allocatedPositions) {
            throw new IndexOutOfBoundsException("KV cache position has not been physically allocated: " + position);
        }
        return stride;
    }

    private int checkedLogicalRange(int layer, int firstPosition, int tokenCount) {
        checkLayer(layer);
        if (firstPosition < 0 || tokenCount < 0
                || firstPosition > layout.maxSequenceLength()
                || tokenCount > layout.maxSequenceLength() - firstPosition) {
            throw new IndexOutOfBoundsException("KV cache range exceeds logical capacity");
        }
        return layout.layerStride(layer);
    }

    private int checkedAllocatedRange(int layer, int firstPosition, int tokenCount) {
        int stride = checkedLogicalRange(layer, firstPosition, tokenCount);
        if (firstPosition > allocatedPositions || tokenCount > allocatedPositions - firstPosition) {
            throw new IndexOutOfBoundsException("KV cache range has not been physically allocated");
        }
        return stride;
    }

    private void checkLayer(int layer) {
        if (layer < 0 || layer >= layout.layerCount()) {
            throw new IndexOutOfBoundsException("Layer index out of range: " + layer);
        }
    }

    private static void checkArrayRange(float[] array, int offset, int length, String name) {
        Objects.requireNonNull(array, name);
        if (offset < 0 || length < 0 || offset > array.length - length) {
            throw new IndexOutOfBoundsException(name + " range is outside the supplied array");
        }
    }

    private void ensureOpen() {
        if (keys == null || values == null) {
            throw new IllegalStateException("KV cache storage is closed");
        }
    }
}
