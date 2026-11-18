package tech.kayys.aljabr.gguf.loader.inference;

import java.util.Arrays;
import java.util.Objects;

import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.cache.KVCacheStorage;

/**
 * Exposes an existing GGUF {@link KVCache} through Aljabr's format-neutral
 * storage contract without changing its backing arrays or sequence pointer.
 *
 * <p>This adapter is non-owning: closing it only closes the adapter and never
 * invalidates the wrapped cache. Storage operations deliberately do not commit
 * sequence positions; callers retain responsibility for sequence admission
 * and advancement through the GGUF cache API.
 */
public final class GgufKVCacheStorageAdapter implements KVCacheStorage {
    private final KVCache cache;
    private boolean closed;

    public GgufKVCacheStorageAdapter(KVCache cache) {
        this.cache = Objects.requireNonNull(cache, "cache");
    }

    @Override
    public KVCacheLayout layout() {
        return cache.layout();
    }

    @Override
    public void write(int layer, int position,
            float[] key, int keyOffset, float[] value, int valueOffset) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkArrayRange(key, keyOffset, stride, "key");
        checkArrayRange(value, valueOffset, stride, "value");
        int destinationOffset = Math.multiplyExact(position, stride);
        System.arraycopy(key, keyOffset, cache.kLayer(layer), destinationOffset, stride);
        System.arraycopy(value, valueOffset, cache.vLayer(layer), destinationOffset, stride);
    }

    @Override
    public void readKey(int layer, int position, float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkArrayRange(destination, destinationOffset, stride, "destination");
        System.arraycopy(cache.kLayer(layer), Math.multiplyExact(position, stride),
                destination, destinationOffset, stride);
    }

    @Override
    public void readValue(int layer, int position, float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkArrayRange(destination, destinationOffset, stride, "destination");
        System.arraycopy(cache.vLayer(layer), Math.multiplyExact(position, stride),
                destination, destinationOffset, stride);
    }

    @Override
    public void writeRange(int layer, int firstPosition, int tokenCount,
            float[] keys, int keysOffset, float[] values, int valuesOffset) {
        ensureOpen();
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(keys, keysOffset, elements, "keys");
        checkArrayRange(values, valuesOffset, elements, "values");
        cache.storeBatch(layer, firstPosition, keys, keysOffset, values, valuesOffset, elements);
    }

    @Override
    public void readKeyRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(destination, destinationOffset, elements, "destination");
        System.arraycopy(cache.kLayer(layer), Math.multiplyExact(firstPosition, stride),
                destination, destinationOffset, elements);
    }

    @Override
    public void readValueRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(destination, destinationOffset, elements, "destination");
        System.arraycopy(cache.vLayer(layer), Math.multiplyExact(firstPosition, stride),
                destination, destinationOffset, elements);
    }

    @Override
    public void clearLayer(int layer) {
        ensureOpen();
        Arrays.fill(cache.kLayer(layer), 0.0f);
        Arrays.fill(cache.vLayer(layer), 0.0f);
    }

    @Override
    public void clear() {
        ensureOpen();
        for (int layer = 0; layer < layout().layerCount(); layer++) {
            clearLayer(layer);
        }
    }

    @Override
    public void truncate(int firstPosition) {
        ensureOpen();
        if (firstPosition < 0 || firstPosition > layout().maxSequenceLength()) {
            throw new IllegalArgumentException(
                    "firstPosition must be within cache capacity: " + firstPosition);
        }
        for (int layer = 0; layer < layout().layerCount(); layer++) {
            int offset = Math.multiplyExact(firstPosition, layout().layerStride(layer));
            Arrays.fill(cache.kLayer(layer), offset, cache.kLayer(layer).length, 0.0f);
            Arrays.fill(cache.vLayer(layer), offset, cache.vLayer(layer).length, 0.0f);
        }
    }

    @Override
    public void close() {
        closed = true;
    }

    private int checkedPosition(int layer, int position) {
        int stride = layout().layerStride(layer);
        if (position < 0 || position >= layout().maxSequenceLength()) {
            throw new IndexOutOfBoundsException("Cache position out of range: " + position);
        }
        return stride;
    }

    private int checkedRange(int layer, int firstPosition, int tokenCount) {
        int stride = layout().layerStride(layer);
        if (firstPosition < 0 || tokenCount < 0
                || firstPosition > layout().maxSequenceLength()
                || tokenCount > layout().maxSequenceLength() - firstPosition) {
            throw new IndexOutOfBoundsException("KV cache range exceeds layer capacity");
        }
        return stride;
    }

    private static void checkArrayRange(float[] array, int offset, int length, String name) {
        if (array == null || offset < 0 || length < 0 || offset > array.length - length) {
            throw new IndexOutOfBoundsException(name + " range is outside the supplied array");
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("GGUF KV cache storage adapter is closed");
        }
    }
}
