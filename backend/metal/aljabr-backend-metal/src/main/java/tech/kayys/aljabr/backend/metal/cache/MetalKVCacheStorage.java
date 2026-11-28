package tech.kayys.aljabr.backend.metal.cache;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;

import tech.kayys.aljabr.cache.KVCacheBufferReadView;
import tech.kayys.aljabr.cache.KVCacheBufferRegion;
import tech.kayys.aljabr.cache.KVCacheBufferRegionProvider;
import tech.kayys.aljabr.cache.KVCacheElementType;
import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.cache.KVCacheMemoryDomain;
import tech.kayys.aljabr.cache.KVCacheStorage;
import tech.kayys.aljabr.metal.binding.MetalBinding;

/**
 * Explicit Apple Metal KV provider backed by retained shared-storage MTLBuffers.
 *
 * <p>Each layer owns distinct K and V buffers. The Metal bridge retains each
 * Objective-C buffer until this storage closes. CPU writes target the shared
 * buffer contents directly; compatible Metal kernels can consume the opaque
 * MTLBuffer handle without staging a second cache allocation.</p>
 *
 * <p>This provider requires an initialized Metal runtime with unified memory.
 * It is not selected automatically and is not a fallback for non-Apple hosts.</p>
 */
public final class MetalKVCacheStorage
        implements KVCacheStorage, KVCacheBufferRegionProvider {

    private final KVCacheLayout layout;
    private final MetalBinding binding;
    private final String deviceId;
    private final MetalKVCacheBufferHandle[] keys;
    private final MetalKVCacheBufferHandle[] values;
    private final List<MetalBufferLease> activeLeases = new ArrayList<>();
    private boolean closed;

    public MetalKVCacheStorage(KVCacheLayout layout) {
        this(layout, MetalBinding.getInstance());
    }

    MetalKVCacheStorage(KVCacheLayout layout, MetalBinding binding) {
        this.layout = Objects.requireNonNull(layout, "layout");
        this.binding = Objects.requireNonNull(binding, "binding");
        if (!binding.isRuntimeActive() || !binding.supportsKVBufferAllocation()) {
            throw new UnsupportedOperationException(
                    "Metal KV storage requires an active unified-memory Metal runtime and KV buffer bridge");
        }
        this.deviceId = binding.deviceName();
        if (deviceId == null || deviceId.isBlank()) {
            throw new IllegalStateException("Metal device identity is unavailable");
        }
        this.keys = new MetalKVCacheBufferHandle[layout.layerCount()];
        this.values = new MetalKVCacheBufferHandle[layout.layerCount()];
        try {
            for (int layer = 0; layer < layout.layerCount(); layer++) {
                long bytes = byteSize(layout.maxSequenceLength(), layout.layerStride(layer));
                keys[layer] = allocate(bytes);
                values[layer] = allocate(bytes);
            }
        } catch (RuntimeException | Error failure) {
            releaseAllocatedBuffers();
            closed = true;
            throw failure;
        }
    }

    private MetalKVCacheBufferHandle allocate(long bytes) {
        MemorySegment objectHandle = binding.createKVBuffer(bytes);
        if (objectHandle == null || objectHandle.equals(MemorySegment.NULL)) {
            throw new IllegalStateException("Metal failed to allocate a shared KV buffer (" + bytes + " bytes)");
        }
        try {
            MemorySegment contentsPointer = binding.kvBufferContents(objectHandle);
            if (contentsPointer == null || contentsPointer.equals(MemorySegment.NULL)) {
                throw new IllegalStateException("Metal shared KV buffer has no CPU-visible contents");
            }
            MemorySegment contents = MemorySegment.ofAddress(contentsPointer.address()).reinterpret(bytes);
            return new MetalKVCacheBufferHandle(this, objectHandle, contents, bytes);
        } catch (RuntimeException | Error failure) {
            binding.releaseKVBuffer(objectHandle);
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
        copyFromArray(key, keyOffset, keys[layer].contents(), offset, stride);
        copyFromArray(value, valueOffset, values[layer].contents(), offset, stride);
    }

    @Override
    public synchronized void readKey(int layer, int position, float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkArrayRange(destination, destinationOffset, stride, "destination");
        copyToArray(keys[layer].contents(), byteOffset(position, stride), destination, destinationOffset, stride);
    }

    @Override
    public synchronized void readValue(int layer, int position, float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkArrayRange(destination, destinationOffset, stride, "destination");
        copyToArray(values[layer].contents(), byteOffset(position, stride), destination, destinationOffset, stride);
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
        copyFromArray(sourceKeys, keysOffset, keys[layer].contents(), offset, elements);
        copyFromArray(sourceValues, valuesOffset, values[layer].contents(), offset, elements);
    }

    @Override
    public synchronized void readKeyRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(destination, destinationOffset, elements, "destination");
        copyToArray(keys[layer].contents(), byteOffset(firstPosition, stride), destination, destinationOffset, elements);
    }

    @Override
    public synchronized void readValueRange(int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        ensureOpen();
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkArrayRange(destination, destinationOffset, elements, "destination");
        copyToArray(values[layer].contents(), byteOffset(firstPosition, stride), destination, destinationOffset, elements);
    }

    @Override
    public synchronized void clearLayer(int layer) {
        ensureOpen();
        checkNoActiveLeases("clear a layer");
        checkLayer(layer);
        keys[layer].contents().fill((byte) 0);
        values[layer].contents().fill((byte) 0);
    }

    @Override
    public synchronized void clear() {
        ensureOpen();
        checkNoActiveLeases("clear storage");
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            keys[layer].contents().fill((byte) 0);
            values[layer].contents().fill((byte) 0);
        }
    }

    @Override
    public synchronized void truncate(int firstPosition) {
        ensureOpen();
        if (firstPosition < 0 || firstPosition > layout.maxSequenceLength()) {
            throw new IllegalArgumentException("firstPosition exceeds KV cache capacity");
        }
        for (MetalBufferLease lease : activeLeases) {
            if (!lease.closed && firstPosition < lease.visibleSequenceLength) {
                throw new IllegalStateException("Cannot truncate a borrowed Metal KV prefix");
            }
        }
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            long offset = byteOffset(firstPosition, layout.layerStride(layer));
            keys[layer].contents().asSlice(offset).fill((byte) 0);
            values[layer].contents().asSlice(offset).fill((byte) 0);
        }
    }

    @Override
    public synchronized KVCacheBufferReadView openBufferReadView(int visibleSequenceLength) {
        ensureOpen();
        if (visibleSequenceLength < 0 || visibleSequenceLength > layout.maxSequenceLength()) {
            throw new IllegalArgumentException("visibleSequenceLength exceeds cache capacity");
        }
        MetalBufferLease lease = new MetalBufferLease(visibleSequenceLength);
        activeLeases.add(lease);
        return lease;
    }

    synchronized <T> T withMetalBuffer(KVCacheBufferReadView view, KVCacheBufferRegion region,
            MetalKVCacheBufferHandle handle,
            java.util.function.Function<MetalKVCacheBufferAccess.View, T> consumer) {
        ensureOpen();
        if (!(view instanceof MetalBufferLease lease) || !lease.owns(region)
                || region.handle() != handle) {
            throw new IllegalStateException("The active lease does not own this Metal KV region");
        }
        lease.ensureLeaseOpen();
        // Bind both pointer views to a confined callback arena so retaining
        // either segment cannot accidentally extend the lease's access window.
        try (Arena callbackArena = Arena.ofConfined()) {
            MemorySegment nativeBuffer = MemorySegment.ofAddress(handle.objectHandle().address())
                    .reinterpret(ValueLayout.ADDRESS.byteSize(), callbackArena, null);
            MemorySegment readOnlyContents = MemorySegment.ofAddress(
                    handle.contents().address())
                    .reinterpret(region.byteLength(), callbackArena, null)
                    .asReadOnly();
            return consumer.apply(new MetalKVCacheBufferAccess.View(
                    nativeBuffer, readOnlyContents, region.byteLength(), deviceId));
        }
    }

    private void checkBorrowedPrefixMutation(int firstPosition, int tokenCount) {
        if (tokenCount == 0) return;
        for (MetalBufferLease lease : activeLeases) {
            if (!lease.closed && firstPosition < lease.visibleSequenceLength) {
                throw new IllegalStateException("Cannot mutate positions visible to an active Metal KV lease");
            }
        }
    }

    private void checkNoActiveLeases(String operation) {
        for (MetalBufferLease lease : activeLeases) {
            if (!lease.closed) throw new IllegalStateException(
                    "Cannot " + operation + " while a Metal KV buffer lease is open");
        }
    }

    private final class MetalBufferLease implements KVCacheBufferReadView {
        private final int visibleSequenceLength;
        private final KVCacheBufferRegion[] keyRegions;
        private final KVCacheBufferRegion[] valueRegions;
        private boolean closed;

        private MetalBufferLease(int visibleSequenceLength) {
            this.visibleSequenceLength = visibleSequenceLength;
            this.keyRegions = new KVCacheBufferRegion[layout.layerCount()];
            this.valueRegions = new KVCacheBufferRegion[layout.layerCount()];
            for (int layer = 0; layer < layout.layerCount(); layer++) {
                long bytes = byteSize(visibleSequenceLength, layout.layerStride(layer));
                keyRegions[layer] = region(keys[layer], bytes);
                valueRegions[layer] = region(values[layer], bytes);
            }
        }

        private KVCacheBufferRegion region(MetalKVCacheBufferHandle handle, long bytes) {
            return new KVCacheBufferRegion(handle, KVCacheMemoryDomain.DEVICE,
                    "metal", deviceId, KVCacheElementType.FLOAT32, 0, bytes);
        }

        private boolean owns(KVCacheBufferRegion region) {
            for (int layer = 0; layer < layout.layerCount(); layer++) {
                if (keyRegions[layer] == region || valueRegions[layer] == region) return true;
            }
            return false;
        }

        private void ensureLeaseOpen() {
            if (closed) throw new IllegalStateException("Metal KV buffer lease is closed");
            ensureOpen();
        }

        @Override public KVCacheLayout layout() {
            synchronized (MetalKVCacheStorage.this) { ensureLeaseOpen(); return layout; }
        }
        @Override public int visibleSequenceLength() {
            synchronized (MetalKVCacheStorage.this) { ensureLeaseOpen(); return visibleSequenceLength; }
        }
        @Override public KVCacheBufferRegion keyRegion(int layer) {
            synchronized (MetalKVCacheStorage.this) { ensureLeaseOpen(); return keyRegions[checkedLayer(layer)]; }
        }
        @Override public KVCacheBufferRegion valueRegion(int layer) {
            synchronized (MetalKVCacheStorage.this) { ensureLeaseOpen(); return valueRegions[checkedLayer(layer)]; }
        }
        @Override public CompletionStage<Void> readiness() {
            synchronized (MetalKVCacheStorage.this) {
                ensureLeaseOpen();
                // Storage writes are synchronous CPU writes into MTLStorageModeShared.
                // The existing Metal dispatch path submits and waits for command completion.
                return CompletableFuture.completedFuture(null);
            }
        }
        @Override public String backendId() {
            synchronized (MetalKVCacheStorage.this) { ensureLeaseOpen(); return "metal"; }
        }
        @Override public void close() {
            synchronized (MetalKVCacheStorage.this) {
                if (!closed) { closed = true; activeLeases.remove(this); }
            }
        }
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        checkNoActiveLeases("close storage");
        closed = true;
        releaseAllocatedBuffers();
    }

    private void releaseAllocatedBuffers() {
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            if (keys[layer] != null) {
                binding.releaseKVBuffer(keys[layer].objectHandle());
                keys[layer] = null;
            }
            if (values[layer] != null) {
                binding.releaseKVBuffer(values[layer].objectHandle());
                values[layer] = null;
            }
        }
    }

    private int checkedPosition(int layer, int position) {
        checkLayer(layer);
        if (position < 0 || position >= layout.maxSequenceLength()) {
            throw new IndexOutOfBoundsException("KV position exceeds cache capacity: " + position);
        }
        return layout.layerStride(layer);
    }

    private int checkedRange(int layer, int firstPosition, int tokenCount) {
        checkLayer(layer);
        if (firstPosition < 0 || tokenCount < 0 || firstPosition > layout.maxSequenceLength()
                || tokenCount > layout.maxSequenceLength() - firstPosition) {
            throw new IndexOutOfBoundsException("KV range exceeds cache capacity");
        }
        return layout.layerStride(layer);
    }

    private int checkedLayer(int layer) { checkLayer(layer); return layer; }
    private void checkLayer(int layer) {
        if (layer < 0 || layer >= layout.layerCount()) throw new IndexOutOfBoundsException("Invalid KV layer: " + layer);
    }
    private void ensureOpen() {
        if (closed) throw new IllegalStateException("Metal KV cache storage is closed");
    }
    private static long byteSize(long positions, long stride) {
        return Math.multiplyExact(Math.multiplyExact(positions, stride), Float.BYTES);
    }
    private static long byteOffset(long position, long stride) { return byteSize(position, stride); }

    private static void copyFromArray(float[] source, int sourceOffset, MemorySegment destination,
            long destinationOffset, int elementCount) {
        if (elementCount == 0) return;
        long bytes = byteSize(elementCount, 1);
        MemorySegment sourceSlice = MemorySegment.ofArray(source).asSlice(byteSize(sourceOffset, 1), bytes);
        destination.asSlice(destinationOffset, bytes).copyFrom(sourceSlice);
    }

    private static void copyToArray(MemorySegment source, long sourceOffset,
            float[] destination, int destinationOffset, int elementCount) {
        if (elementCount == 0) return;
        long bytes = byteSize(elementCount, 1);
        MemorySegment destinationSlice = MemorySegment.ofArray(destination).asSlice(byteSize(destinationOffset, 1), bytes);
        destinationSlice.copyFrom(source.asSlice(sourceOffset, bytes));
    }

    private static void checkArrayRange(float[] array, int offset, int length, String name) {
        Objects.requireNonNull(array, name);
        if (length < 0 || offset < 0 || offset > array.length - length) {
            throw new IndexOutOfBoundsException(name + " range is outside the supplied array");
        }
    }
}
