package tech.kayys.aljabr.cache;

import java.util.Arrays;
import java.util.Objects;

/**
 * Heap-array implementation of {@link KVCacheStorage}.
 *
 * <p>Stores each layer as a contiguous sequence-major float array. This is a
 * straightforward portable implementation; optimized CPU/native/device
 * providers can implement the same contract without changing model formats.
 */
public final class HeapKVCacheStorage implements KVCacheStorage {
    private final KVCacheLayout layout;
    private float[][] keys;
    private float[][] values;

    public HeapKVCacheStorage(KVCacheLayout layout) {
        this.layout = Objects.requireNonNull(layout, "layout");
        this.keys = new float[layout.layerCount()][];
        this.values = new float[layout.layerCount()][];
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            int elements = Math.multiplyExact(
                    layout.maxSequenceLength(), layout.layerStride(layer));
            keys[layer] = new float[elements];
            values[layer] = new float[elements];
        }
    }

    @Override
    public KVCacheLayout layout() {
        return layout;
    }

    @Override
    public void write(int layer, int position,
            float[] key, int keyOffset,
            float[] value, int valueOffset) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkArrayRange(key, keyOffset, stride, "key");
        checkArrayRange(value, valueOffset, stride, "value");
        int cacheOffset = Math.multiplyExact(position, stride);

        // All arguments and ranges are validated before either cache is mutated.
        System.arraycopy(key, keyOffset, keys[layer], cacheOffset, stride);
        System.arraycopy(value, valueOffset, values[layer], cacheOffset, stride);
    }

    @Override
    public void readKey(int layer, int position, float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkArrayRange(destination, destinationOffset, stride, "destination");
        System.arraycopy(keys[layer], Math.multiplyExact(position, stride),
                destination, destinationOffset, stride);
    }

    @Override
    public void readValue(int layer, int position, float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkArrayRange(destination, destinationOffset, stride, "destination");
        System.arraycopy(values[layer], Math.multiplyExact(position, stride),
                destination, destinationOffset, stride);
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
            throw new IllegalArgumentException(
                    "firstPosition must be within cache capacity: " + firstPosition);
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
    }

    private int checkedPosition(int layer, int position) {
        checkLayer(layer);
        if (position < 0 || position >= layout.maxSequenceLength()) {
            throw new IndexOutOfBoundsException("Cache position out of range: " + position);
        }
        return layout.layerStride(layer);
    }

    private void checkLayer(int layer) {
        if (layer < 0 || layer >= layout.layerCount()) {
            throw new IndexOutOfBoundsException("Layer index out of range: " + layer);
        }
    }

    private static void checkArrayRange(float[] array, int offset, int length, String name) {
        Objects.requireNonNull(array, name);
        if (offset < 0 || offset > array.length - length) {
            throw new IndexOutOfBoundsException(name + " range is outside the supplied array");
        }
    }

    private void ensureOpen() {
        if (keys == null || values == null) {
            throw new IllegalStateException("KV cache storage is closed");
        }
    }
}
