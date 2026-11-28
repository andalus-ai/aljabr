package tech.kayys.aljabr.cache;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.nio.FloatBuffer;
import java.util.function.Function;

/**
 * Heap-array implementation of {@link KVCacheStorage}.
 *
 * <p>Stores each layer as a contiguous sequence-major float array. This is a
 * straightforward portable implementation; optimized CPU/native/device
 * providers can implement the same contract without changing model formats.
 */
public final class HeapKVCacheStorage implements KVCacheStorage, KVCacheBufferRegionProvider {
    private final KVCacheLayout layout;
    private float[][] keys;
    private float[][] values;
    private final List<HeapBufferLease> activeBufferLeases = new ArrayList<>();

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
    public synchronized void write(int layer, int position,
            float[] key, int keyOffset,
            float[] value, int valueOffset) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkArrayRange(key, keyOffset, stride, "key");
        checkArrayRange(value, valueOffset, stride, "value");
        checkBorrowedPrefixMutation(position, 1);
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
    public synchronized void writeRange(int layer, int firstPosition, int tokenCount,
            float[] sourceKeys, int keysOffset, float[] sourceValues, int valuesOffset) {
        ensureOpen();
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(sourceKeys, keysOffset, elements, "keys");
        checkArrayRange(sourceValues, valuesOffset, elements, "values");
        checkBorrowedPrefixMutation(firstPosition, tokenCount);
        int destinationOffset = Math.multiplyExact(firstPosition, stride);

        // Validate both source ranges before mutating either backing array.
        System.arraycopy(sourceKeys, keysOffset, keys[layer], destinationOffset, elements);
        System.arraycopy(sourceValues, valuesOffset, values[layer], destinationOffset, elements);
    }

    @Override
    public void readKeyRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(destination, destinationOffset, elements, "destination");
        int sourceOffset = Math.multiplyExact(firstPosition, stride);
        System.arraycopy(keys[layer], sourceOffset, destination, destinationOffset, elements);
    }

    @Override
    public void readValueRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(destination, destinationOffset, elements, "destination");
        int sourceOffset = Math.multiplyExact(firstPosition, stride);
        System.arraycopy(values[layer], sourceOffset, destination, destinationOffset, elements);
    }

    private int checkedRange(int layer, int firstPosition, int tokenCount) {
        checkLayer(layer);
        if (firstPosition < 0 || tokenCount < 0
                || firstPosition > layout.maxSequenceLength()
                || tokenCount > layout.maxSequenceLength() - firstPosition) {
            throw new IndexOutOfBoundsException("KV cache range exceeds layer capacity");
        }
        return layout.layerStride(layer);
    }

    @Override
    public synchronized void clearLayer(int layer) {
        ensureOpen();
        checkNoActiveBufferLeases("clear layer");
        checkLayer(layer);
        Arrays.fill(keys[layer], 0.0f);
        Arrays.fill(values[layer], 0.0f);
    }

    @Override
    public synchronized void clear() {
        ensureOpen();
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            clearLayer(layer);
        }
    }

    @Override
    public synchronized void truncate(int firstPosition) {
        ensureOpen();
        if (firstPosition < 0 || firstPosition > layout.maxSequenceLength()) {
            throw new IllegalArgumentException(
                    "firstPosition must be within cache capacity: " + firstPosition);
        }
        for (HeapBufferLease lease : activeBufferLeases) {
            if (!lease.closed && firstPosition < lease.visibleSequenceLength) {
                throw new IllegalStateException("Cannot truncate a borrowed KV prefix");
            }
        }
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            int offset = Math.multiplyExact(firstPosition, layout.layerStride(layer));
            Arrays.fill(keys[layer], offset, keys[layer].length, 0.0f);
            Arrays.fill(values[layer], offset, values[layer].length, 0.0f);
        }
    }

    @Override
    public synchronized KVCacheBufferReadView openBufferReadView(int visibleSequenceLength) {
        ensureOpen();
        if (visibleSequenceLength < 0 || visibleSequenceLength > layout.maxSequenceLength()) {
            throw new IllegalArgumentException("visibleSequenceLength exceeds cache capacity");
        }
        HeapBufferLease lease = new HeapBufferLease(visibleSequenceLength);
        activeBufferLeases.add(lease);
        return lease;
    }

    synchronized <T> T withReadOnlyBuffer(
            KVCacheBufferReadView view,
            KVCacheBufferRegion region,
            HeapKVCacheBufferHandle handle,
            Function<FloatBuffer, T> consumer) {
        ensureOpen();
        if (!(view instanceof HeapBufferLease lease) || !lease.owns(region)
                || region.handle() != handle) {
            throw new IllegalStateException("The active lease does not own this heap KV region");
        }
        lease.ensureLeaseOpen();
        int offset = Math.toIntExact(region.byteOffset() / Float.BYTES);
        int length = Math.toIntExact(region.elementCount());
        float[] allocation = handle.storage();
        if (offset < 0 || length < 0 || offset > allocation.length - length) {
            throw new IllegalArgumentException("Heap KV region exceeds its allocation");
        }
        FloatBuffer readOnly = FloatBuffer.wrap(allocation, offset, length)
                .slice()
                .asReadOnlyBuffer();
        return consumer.apply(readOnly);
    }

    private void checkBorrowedPrefixMutation(int firstPosition, int tokenCount) {
        if (tokenCount == 0) return;
        for (HeapBufferLease lease : activeBufferLeases) {
            if (!lease.closed && firstPosition < lease.visibleSequenceLength) {
                throw new IllegalStateException(
                        "Cannot mutate KV positions visible to an active buffer lease");
            }
        }
    }

    private void checkNoActiveBufferLeases(String operation) {
        for (HeapBufferLease lease : activeBufferLeases) {
            if (!lease.closed) {
                throw new IllegalStateException("Cannot " + operation
                        + " while a KV buffer lease is open");
            }
        }
    }

    private final class HeapBufferLease implements KVCacheBufferReadView {
        private final int visibleSequenceLength;
        private final KVCacheBufferRegion[] keyRegions;
        private final KVCacheBufferRegion[] valueRegions;
        private boolean closed;

        private HeapBufferLease(int visibleSequenceLength) {
            this.visibleSequenceLength = visibleSequenceLength;
            this.keyRegions = new KVCacheBufferRegion[layout.layerCount()];
            this.valueRegions = new KVCacheBufferRegion[layout.layerCount()];
            for (int layer = 0; layer < layout.layerCount(); layer++) {
                long byteLength = Math.multiplyExact(
                        Math.multiplyExact((long) visibleSequenceLength,
                                layout.layerStride(layer)), Float.BYTES);
                keyRegions[layer] = new KVCacheBufferRegion(
                        new HeapKVCacheBufferHandle(HeapKVCacheStorage.this, keys[layer]),
                        KVCacheMemoryDomain.HOST_HEAP, "cpu", "",
                        KVCacheElementType.FLOAT32, 0, byteLength);
                valueRegions[layer] = new KVCacheBufferRegion(
                        new HeapKVCacheBufferHandle(HeapKVCacheStorage.this, values[layer]),
                        KVCacheMemoryDomain.HOST_HEAP, "cpu", "",
                        KVCacheElementType.FLOAT32, 0, byteLength);
            }
        }

        private boolean owns(KVCacheBufferRegion region) {
            for (int layer = 0; layer < layout.layerCount(); layer++) {
                if (keyRegions[layer] == region || valueRegions[layer] == region) {
                    return true;
                }
            }
            return false;
        }

        private void ensureLeaseOpen() {
            if (closed) throw new IllegalStateException("KV buffer lease is closed");
            ensureOpen();
        }

        @Override
        public KVCacheLayout layout() {
            synchronized (HeapKVCacheStorage.this) {
                ensureLeaseOpen();
                return layout;
            }
        }

        @Override
        public int visibleSequenceLength() {
            synchronized (HeapKVCacheStorage.this) {
                ensureLeaseOpen();
                return visibleSequenceLength;
            }
        }

        @Override
        public KVCacheBufferRegion keyRegion(int layer) {
            synchronized (HeapKVCacheStorage.this) {
                ensureLeaseOpen();
                return keyRegions[checkedLayer(layer)];
            }
        }

        @Override
        public KVCacheBufferRegion valueRegion(int layer) {
            synchronized (HeapKVCacheStorage.this) {
                ensureLeaseOpen();
                return valueRegions[checkedLayer(layer)];
            }
        }

        @Override
        public java.util.concurrent.CompletionStage<Void> readiness() {
            synchronized (HeapKVCacheStorage.this) {
                ensureLeaseOpen();
                return CompletableFuture.completedFuture(null);
            }
        }

        @Override
        public String backendId() {
            synchronized (HeapKVCacheStorage.this) {
                ensureLeaseOpen();
                return "cpu";
            }
        }

        @Override
        public void close() {
            synchronized (HeapKVCacheStorage.this) {
                if (!closed) {
                    closed = true;
                    activeBufferLeases.remove(this);
                }
            }
        }
    }

    @Override
    public synchronized void close() {
        if (!activeBufferLeases.isEmpty()) {
            throw new IllegalStateException("Cannot close KV storage while buffer leases are open");
        }
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

    private int checkedLayer(int layer) {
        checkLayer(layer);
        return layer;
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
