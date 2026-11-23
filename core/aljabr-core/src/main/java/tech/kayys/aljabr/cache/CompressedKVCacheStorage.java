package tech.kayys.aljabr.cache;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Reference {@link KVCacheStorage} that persists each token position as
 * independently encoded K and V blocks.
 *
 * <p>The storage is codec-neutral: algorithms are supplied through
 * {@link KVCacheCompressionCodec}, while this class owns only encoded-block
 * placement, geometry, and storage lifecycle. One position is the atomic
 * publication unit. Both K and V blocks are encoded successfully before an
 * existing position is replaced.</p>
 *
 * <p>Reads decode on demand into caller-owned arrays. Unwritten positions read
 * as zero, matching the heap storage implementations. This deliberately
 * favors simple lifecycle semantics over block-level batching; per-position
 * metadata and decode overhead can be significant for short vectors or
 * high-throughput workloads.</p>
 *
 * <p>The storage does not own or close the supplied codec. As with other
 * storage implementations, callers must externally synchronize concurrent
 * access to the same layer and position.</p>
 */
public final class CompressedKVCacheStorage implements KVCacheStorage {

    private final KVCacheLayout layout;
    private final KVCacheCompressionCodec codec;
    private final List<Map<Integer, EncodedPosition>> layers;
    private boolean closed;

    public CompressedKVCacheStorage(KVCacheLayout layout, KVCacheCompressionCodec codec) {
        this.layout = Objects.requireNonNull(layout, "layout");
        this.codec = Objects.requireNonNull(codec, "codec");
        this.layers = new ArrayList<>(layout.layerCount());
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            layers.add(new HashMap<>());
        }
    }

    @Override
    public KVCacheLayout layout() {
        return layout;
    }

    @Override
    public void write(
            int layer,
            int position,
            float[] key,
            int keyOffset,
            float[] value,
            int valueOffset) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkArrayRange(key, keyOffset, stride, "key");
        checkArrayRange(value, valueOffset, stride, "value");

        KVCacheCompressionContext keyContext = context(layer, position, KVCacheTensorKind.KEY);
        KVCacheCompressionContext valueContext = context(layer, position, KVCacheTensorKind.VALUE);

        // Encode both tensors before publishing either one. A failure leaves
        // the previously stored position (if any) intact.
        KVCacheEncodedBlock encodedKey = codec.encode(key, keyOffset, keyContext);
        KVCacheEncodedBlock encodedValue = codec.encode(value, valueOffset, valueContext);
        layers.get(layer).put(position, new EncodedPosition(encodedKey, encodedValue));
    }

    @Override
    public void readKey(int layer, int position, float[] destination, int destinationOffset) {
        ensureOpen();
        checkedPosition(layer, position);
        int stride = layout.layerStride(layer);
        checkArrayRange(destination, destinationOffset, stride, "destination");
        EncodedPosition encoded = layers.get(layer).get(position);
        if (encoded == null) {
            java.util.Arrays.fill(destination, destinationOffset, destinationOffset + stride, 0.0f);
            return;
        }
        codec.decode(encoded.key(), destination, destinationOffset,
                context(layer, position, KVCacheTensorKind.KEY));
    }

    @Override
    public void readValue(int layer, int position, float[] destination, int destinationOffset) {
        ensureOpen();
        checkedPosition(layer, position);
        int stride = layout.layerStride(layer);
        checkArrayRange(destination, destinationOffset, stride, "destination");
        EncodedPosition encoded = layers.get(layer).get(position);
        if (encoded == null) {
            java.util.Arrays.fill(destination, destinationOffset, destinationOffset + stride, 0.0f);
            return;
        }
        codec.decode(encoded.value(), destination, destinationOffset,
                context(layer, position, KVCacheTensorKind.VALUE));
    }

    @Override
    public void clearLayer(int layer) {
        ensureOpen();
        checkedLayer(layer);
        layers.get(layer).clear();
    }

    @Override
    public void clear() {
        ensureOpen();
        for (Map<Integer, EncodedPosition> layer : layers) {
            layer.clear();
        }
    }

    @Override
    public void truncate(int firstPosition) {
        ensureOpen();
        if (firstPosition < 0 || firstPosition > layout.maxSequenceLength()) {
            throw new IllegalArgumentException(
                    "firstPosition must be within cache capacity: " + firstPosition);
        }
        for (Map<Integer, EncodedPosition> layer : layers) {
            layer.keySet().removeIf(position -> position >= firstPosition);
        }
    }

    /**
     * Number of positions for which both encoded K and V blocks are stored.
     */
    public int storedPositionCount() {
        ensureOpen();
        int count = 0;
        for (Map<Integer, EncodedPosition> layer : layers) {
            count = Math.addExact(count, layer.size());
        }
        return count;
    }

    /** Total encoded payload bytes, excluding decoder metadata. */
    public long encodedPayloadBytes() {
        ensureOpen();
        long bytes = 0L;
        for (Map<Integer, EncodedPosition> layer : layers) {
            for (EncodedPosition position : layer.values()) {
                bytes = Math.addExact(bytes,
                        (long) position.key().payloadSizeBytes() + position.value().payloadSizeBytes());
            }
        }
        return bytes;
    }

    /** Total decoder metadata bytes. */
    public long encodedMetadataBytes() {
        ensureOpen();
        long bytes = 0L;
        for (Map<Integer, EncodedPosition> layer : layers) {
            for (EncodedPosition position : layer.values()) {
                bytes = Math.addExact(bytes,
                        (long) position.key().metadataSizeBytes() + position.value().metadataSizeBytes());
            }
        }
        return bytes;
    }

    @Override
    public void close() {
        if (!closed) {
            clearStoredBlocks();
            closed = true;
        }
    }

    private KVCacheCompressionContext context(int layer, int position, KVCacheTensorKind kind) {
        return new KVCacheCompressionContext(
                layer,
                position,
                1,
                layout.kvHeads(layer),
                layout.headDimension(layer),
                kind);
    }

    private int checkedPosition(int layer, int position) {
        checkedLayer(layer);
        if (position < 0 || position >= layout.maxSequenceLength()) {
            throw new IndexOutOfBoundsException("Cache position out of range: " + position);
        }
        return layout.layerStride(layer);
    }

    private void checkedLayer(int layer) {
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
        if (closed) {
            throw new IllegalStateException("Compressed KV cache storage is closed");
        }
    }

    private void clearStoredBlocks() {
        for (Map<Integer, EncodedPosition> layer : layers) {
            layer.clear();
        }
    }

    private record EncodedPosition(KVCacheEncodedBlock key, KVCacheEncodedBlock value) {
        private EncodedPosition {
            Objects.requireNonNull(key, "key");
            Objects.requireNonNull(value, "value");
        }
    }
}
