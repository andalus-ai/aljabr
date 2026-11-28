package tech.kayys.aljabr.cache;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;

/** CPU KV storage backed by shared native memory allocated through the JDK FFM API. */
public final class NativeKVCacheStorage implements KVCacheStorage, KVCacheBufferRegionProvider {
    private final KVCacheLayout layout;
    private final Arena arena;
    private MemorySegment[] keys;
    private MemorySegment[] values;
    private final List<NativeBufferLease> activeBufferLeases = new ArrayList<>();
    private boolean closed;

    public NativeKVCacheStorage(KVCacheLayout layout) {
        this.layout = Objects.requireNonNull(layout, "layout");
        this.arena = Arena.ofShared();
        this.keys = new MemorySegment[layout.layerCount()];
        this.values = new MemorySegment[layout.layerCount()];
        try {
            for (int layer = 0; layer < layout.layerCount(); layer++) {
                long bytes = byteSize(layout.maxSequenceLength(), layout.layerStride(layer));
                keys[layer] = arena.allocate(bytes, Float.BYTES);
                values[layer] = arena.allocate(bytes, Float.BYTES);
            }
        } catch (RuntimeException | Error failure) {
            arena.close();
            closed = true;
            throw failure;
        }
    }

    @Override public KVCacheLayout layout() { return layout; }

    @Override
    public synchronized void write(int layer, int position, float[] key, int keyOffset,
            float[] value, int valueOffset) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkArrayRange(key, keyOffset, stride, "key");
        checkArrayRange(value, valueOffset, stride, "value");
        checkBorrowedPrefixMutation(position, 1);
        long offset = byteOffset(position, stride);
        copyFromArray(key, keyOffset, keys[layer], offset, stride);
        copyFromArray(value, valueOffset, values[layer], offset, stride);
    }

    @Override
    public synchronized void readKey(int layer, int position, float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkArrayRange(destination, destinationOffset, stride, "destination");
        copyToArray(keys[layer], byteOffset(position, stride), destination, destinationOffset, stride);
    }

    @Override
    public synchronized void readValue(int layer, int position, float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkArrayRange(destination, destinationOffset, stride, "destination");
        copyToArray(values[layer], byteOffset(position, stride), destination, destinationOffset, stride);
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
        long offset = byteOffset(firstPosition, stride);
        copyFromArray(sourceKeys, keysOffset, keys[layer], offset, elements);
        copyFromArray(sourceValues, valuesOffset, values[layer], offset, elements);
    }

    @Override
    public synchronized void readKeyRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(destination, destinationOffset, elements, "destination");
        copyToArray(keys[layer], byteOffset(firstPosition, stride), destination, destinationOffset, elements);
    }

    @Override
    public synchronized void readValueRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(destination, destinationOffset, elements, "destination");
        copyToArray(values[layer], byteOffset(firstPosition, stride), destination, destinationOffset, elements);
    }

    @Override
    public synchronized void clearLayer(int layer) {
        ensureOpen();
        checkNoActiveBufferLeases("clear layer");
        checkLayer(layer);
        keys[layer].fill((byte) 0);
        values[layer].fill((byte) 0);
    }

    @Override
    public synchronized void clear() {
        ensureOpen();
        checkNoActiveBufferLeases("clear storage");
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            keys[layer].fill((byte) 0);
            values[layer].fill((byte) 0);
        }
    }

    @Override
    public synchronized void truncate(int firstPosition) {
        ensureOpen();
        if (firstPosition < 0 || firstPosition > layout.maxSequenceLength()) {
            throw new IllegalArgumentException("firstPosition must be within cache capacity: " + firstPosition);
        }
        for (NativeBufferLease lease : activeBufferLeases) {
            if (!lease.closed && firstPosition < lease.visibleSequenceLength) {
                throw new IllegalStateException("Cannot truncate a borrowed KV prefix");
            }
        }
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            long offset = byteOffset(firstPosition, layout.layerStride(layer));
            keys[layer].asSlice(offset).fill((byte) 0);
            values[layer].asSlice(offset).fill((byte) 0);
        }
    }

    @Override
    public synchronized KVCacheBufferReadView openBufferReadView(int visibleSequenceLength) {
        ensureOpen();
        if (visibleSequenceLength < 0 || visibleSequenceLength > layout.maxSequenceLength()) {
            throw new IllegalArgumentException("visibleSequenceLength exceeds cache capacity");
        }
        NativeBufferLease lease = new NativeBufferLease(visibleSequenceLength);
        activeBufferLeases.add(lease);
        return lease;
    }

    synchronized <T> T withReadOnlySegment(KVCacheBufferReadView view, KVCacheBufferRegion region,
            NativeKVCacheBufferHandle handle, java.util.function.Function<MemorySegment, T> consumer) {
        ensureOpen();
        if (!(view instanceof NativeBufferLease lease) || !lease.owns(region) || region.handle() != handle) {
            throw new IllegalStateException("The active lease does not own this native KV region");
        }
        lease.ensureLeaseOpen();
        MemorySegment allocation = handle.segment();
        long offset = region.byteOffset();
        long length = region.byteLength();
        if (offset > allocation.byteSize() || length > allocation.byteSize() - offset) {
            throw new IllegalArgumentException("Native KV region exceeds its allocation");
        }
        return consumer.apply(allocation.asSlice(offset, length).asReadOnly());
    }

    private void checkBorrowedPrefixMutation(int firstPosition, int tokenCount) {
        if (tokenCount == 0) return;
        for (NativeBufferLease lease : activeBufferLeases) {
            if (!lease.closed && firstPosition < lease.visibleSequenceLength) {
                throw new IllegalStateException("Cannot mutate KV positions visible to an active buffer lease");
            }
        }
    }

    private void checkNoActiveBufferLeases(String operation) {
        for (NativeBufferLease lease : activeBufferLeases) {
            if (!lease.closed) throw new IllegalStateException("Cannot " + operation + " while a KV buffer lease is open");
        }
    }

    private final class NativeBufferLease implements KVCacheBufferReadView {
        private final int visibleSequenceLength;
        private final KVCacheBufferRegion[] keyRegions;
        private final KVCacheBufferRegion[] valueRegions;
        private boolean closed;

        private NativeBufferLease(int visibleSequenceLength) {
            this.visibleSequenceLength = visibleSequenceLength;
            this.keyRegions = new KVCacheBufferRegion[layout.layerCount()];
            this.valueRegions = new KVCacheBufferRegion[layout.layerCount()];
            for (int layer = 0; layer < layout.layerCount(); layer++) {
                long byteLength = byteSize(visibleSequenceLength, layout.layerStride(layer));
                keyRegions[layer] = region(keys[layer], byteLength);
                valueRegions[layer] = region(values[layer], byteLength);
            }
        }

        private KVCacheBufferRegion region(MemorySegment segment, long byteLength) {
            return new KVCacheBufferRegion(new NativeKVCacheBufferHandle(NativeKVCacheStorage.this, segment),
                    KVCacheMemoryDomain.HOST_NATIVE, "cpu", "", KVCacheElementType.FLOAT32, 0, byteLength);
        }

        private boolean owns(KVCacheBufferRegion region) {
            for (int layer = 0; layer < layout.layerCount(); layer++) {
                if (keyRegions[layer] == region || valueRegions[layer] == region) return true;
            }
            return false;
        }

        private void ensureLeaseOpen() {
            if (closed) throw new IllegalStateException("KV buffer lease is closed");
            ensureOpen();
        }

        @Override public KVCacheLayout layout() {
            synchronized (NativeKVCacheStorage.this) { ensureLeaseOpen(); return layout; }
        }
        @Override public int visibleSequenceLength() {
            synchronized (NativeKVCacheStorage.this) { ensureLeaseOpen(); return visibleSequenceLength; }
        }
        @Override public KVCacheBufferRegion keyRegion(int layer) {
            synchronized (NativeKVCacheStorage.this) { ensureLeaseOpen(); return keyRegions[checkedLayer(layer)]; }
        }
        @Override public KVCacheBufferRegion valueRegion(int layer) {
            synchronized (NativeKVCacheStorage.this) { ensureLeaseOpen(); return valueRegions[checkedLayer(layer)]; }
        }
        @Override public KVCacheBufferLayout keyLayout(int layer) {
            synchronized (NativeKVCacheStorage.this) {
                ensureLeaseOpen();
                int checkedLayer = checkedLayer(layer);
                return KVCacheBufferLayout.sequenceMajor(
                        visibleSequenceLength, layout.kvHeads(checkedLayer), layout.headDimension(checkedLayer));
            }
        }
        @Override public KVCacheBufferLayout valueLayout(int layer) {
            synchronized (NativeKVCacheStorage.this) {
                ensureLeaseOpen();
                int checkedLayer = checkedLayer(layer);
                return KVCacheBufferLayout.sequenceMajor(
                        visibleSequenceLength, layout.kvHeads(checkedLayer), layout.headDimension(checkedLayer));
            }
        }
        @Override public CompletionStage<Void> readiness() {
            synchronized (NativeKVCacheStorage.this) {
                ensureLeaseOpen();
                return CompletableFuture.completedFuture(null);
            }
        }
        @Override public String backendId() {
            synchronized (NativeKVCacheStorage.this) { ensureLeaseOpen(); return "cpu"; }
        }
        @Override public void close() {
            synchronized (NativeKVCacheStorage.this) {
                if (!closed) { closed = true; activeBufferLeases.remove(this); }
            }
        }
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        checkNoActiveBufferLeases("close storage");
        closed = true;
        keys = null;
        values = null;
        arena.close();
    }

    private int checkedPosition(int layer, int position) {
        checkLayer(layer);
        if (position < 0 || position >= layout.maxSequenceLength()) {
            throw new IndexOutOfBoundsException("Cache position out of range: " + position);
        }
        return layout.layerStride(layer);
    }

    private int checkedRange(int layer, int firstPosition, int tokenCount) {
        checkLayer(layer);
        if (firstPosition < 0 || tokenCount < 0 || firstPosition > layout.maxSequenceLength()
                || tokenCount > layout.maxSequenceLength() - firstPosition) {
            throw new IndexOutOfBoundsException("KV cache range exceeds layer capacity");
        }
        return layout.layerStride(layer);
    }

    private int checkedLayer(int layer) { checkLayer(layer); return layer; }
    private void checkLayer(int layer) {
        if (layer < 0 || layer >= layout.layerCount()) throw new IndexOutOfBoundsException("Layer index out of range: " + layer);
    }

    private void ensureOpen() {
        if (closed || keys == null || values == null || !arena.scope().isAlive()) {
            throw new IllegalStateException("KV cache storage is closed");
        }
    }

    private static long byteSize(long positions, long stride) {
        return Math.multiplyExact(Math.multiplyExact(positions, stride), Float.BYTES);
    }
    private static long byteOffset(long position, long stride) { return byteSize(position, stride); }

    private static void copyFromArray(float[] source, int sourceOffset, MemorySegment destination,
            long destinationByteOffset, int elementCount) {
        if (elementCount == 0) return;
        long bytes = byteSize(elementCount, 1);
        MemorySegment sourceSlice = MemorySegment.ofArray(source).asSlice(byteSize(sourceOffset, 1), bytes);
        destination.asSlice(destinationByteOffset, bytes).copyFrom(sourceSlice);
    }

    private static void copyToArray(MemorySegment source, long sourceByteOffset,
            float[] destination, int destinationOffset, int elementCount) {
        if (elementCount == 0) return;
        long bytes = byteSize(elementCount, 1);
        MemorySegment destinationSlice = MemorySegment.ofArray(destination).asSlice(byteSize(destinationOffset, 1), bytes);
        destinationSlice.copyFrom(source.asSlice(sourceByteOffset, bytes));
    }

    private static void checkArrayRange(float[] array, int offset, int length, String name) {
        Objects.requireNonNull(array, name);
        if (length < 0 || offset < 0 || offset > array.length - length) {
            throw new IndexOutOfBoundsException(name + " range is outside the supplied array");
        }
    }
}
