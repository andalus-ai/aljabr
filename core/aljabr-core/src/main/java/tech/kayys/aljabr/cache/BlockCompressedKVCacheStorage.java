package tech.kayys.aljabr.cache;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Block-oriented reference storage for compressed transformer KV tensors.
 *
 * <p>Each layer stores contiguous position blocks. A codec receives the whole
 * block in one call, allowing per-block metadata (codebooks, norms, transform
 * parameters) to be amortized across several positions. The storage remains
 * codec-neutral and does not own the codec.</p>
 *
 * <p>Writes are atomic at block granularity: K and V are both encoded before
 * the replacement block is published. A range spanning multiple blocks may
 * publish earlier complete blocks if a later block fails. Unwritten positions
 * inside an allocated block remain logically zero.</p>
 *
 * <p>This implementation favors correctness and simple ownership over
 * allocation minimization. It decodes and re-encodes a block for an update;
 * it is not yet a fused or zero-copy execution path. Callers must externally
 * synchronize concurrent access.</p>
 */
public final class BlockCompressedKVCacheStorage implements KVCacheStorage {

    private final KVCacheLayout layout;
    private final KVCacheCompressionCodec codec;
    private final int blockSize;
    private final List<Map<Integer, StoredBlock>> layers;
    private boolean closed;

    /**
     * @param layout logical cache geometry
     * @param codec codec used for each K/V block
     * @param blockSize number of sequence positions per block
     */
    public BlockCompressedKVCacheStorage(
            KVCacheLayout layout, KVCacheCompressionCodec codec, int blockSize) {
        this.layout = Objects.requireNonNull(layout, "layout");
        this.codec = Objects.requireNonNull(codec, "codec");
        if (blockSize <= 0) {
            throw new IllegalArgumentException("blockSize must be positive");
        }
        this.blockSize = blockSize;
        this.layers = new ArrayList<>(layout.layerCount());
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            layers.add(new HashMap<>());
        }
    }

    @Override
    public KVCacheLayout layout() {
        return layout;
    }

    /** Number of sequence positions represented by each full block. */
    public int blockSize() {
        return blockSize;
    }

    @Override
    public void write(
            int layer, int position,
            float[] key, int keyOffset,
            float[] value, int valueOffset) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkRange(key, keyOffset, stride, "key");
        checkRange(value, valueOffset, stride, "value");
        int blockIndex = position / blockSize;
        int firstPosition = blockStart(blockIndex);
        int tokens = blockTokenCount(firstPosition);
        int localPosition = position - firstPosition;

        StoredBlock previous = layers.get(layer).get(blockIndex);
        float[] keys = new float[Math.multiplyExact(tokens, stride)];
        float[] values = new float[keys.length];
        BitSet written = new BitSet(tokens);
        if (previous != null) {
            decode(previous, layer, firstPosition, tokens, keys, values);
            written.or(previous.written());
        }
        System.arraycopy(key, keyOffset, keys, localPosition * stride, stride);
        System.arraycopy(value, valueOffset, values, localPosition * stride, stride);
        written.set(localPosition);

        StoredBlock replacement = encode(layer, firstPosition, tokens, keys, values, written);
        layers.get(layer).put(blockIndex, replacement);
    }

    @Override
    public void writeRange(
            int layer, int firstPosition, int tokenCount,
            float[] keys, int keysOffset, float[] values, int valuesOffset) {
        ensureOpen();
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkRange(keys, keysOffset, elements, "keys");
        checkRange(values, valuesOffset, elements, "values");
        int completed = 0;
        while (completed < tokenCount) {
            int position = firstPosition + completed;
            int blockRemaining = blockSize - position % blockSize;
            int count = Math.min(blockRemaining, tokenCount - completed);
            int blockIndex = position / blockSize;
            int blockFirst = blockStart(blockIndex);
            int blockTokens = blockTokenCount(blockFirst);
            StoredBlock previous = layers.get(layer).get(blockIndex);
            float[] blockKeys = new float[Math.multiplyExact(blockTokens, stride)];
            float[] blockValues = new float[blockKeys.length];
            BitSet written = new BitSet(blockTokens);
            if (previous != null) {
                decode(previous, layer, blockFirst, blockTokens, blockKeys, blockValues);
                written.or(previous.written());
            }
            int localOffset = (position - blockFirst) * stride;
            int sourceOffset = keysOffset + completed * stride;
            System.arraycopy(keys, sourceOffset, blockKeys, localOffset, count * stride);
            System.arraycopy(values, valuesOffset + completed * stride,
                    blockValues, localOffset, count * stride);
            written.set(position - blockFirst, position - blockFirst + count);
            StoredBlock replacement = encode(
                    layer, blockFirst, blockTokens, blockKeys, blockValues, written);
            layers.get(layer).put(blockIndex, replacement);
            completed += count;
        }
    }

    @Override
    public void readKey(int layer, int position, float[] destination, int destinationOffset) {
        readOne(layer, position, destination, destinationOffset, true);
    }

    @Override
    public void readValue(int layer, int position, float[] destination, int destinationOffset) {
        readOne(layer, position, destination, destinationOffset, false);
    }

    private void readOne(
            int layer, int position, float[] destination, int destinationOffset, boolean key) {
        ensureOpen();
        int stride = checkedPosition(layer, position);
        checkRange(destination, destinationOffset, stride, "destination");
        int blockIndex = position / blockSize;
        StoredBlock stored = layers.get(layer).get(blockIndex);
        if (stored == null || !stored.written().get(position - blockStart(blockIndex))) {
            java.util.Arrays.fill(destination, destinationOffset, destinationOffset + stride, 0.0f);
            return;
        }
        int first = blockStart(blockIndex);
        int tokens = blockTokenCount(first);
        float[] decoded = new float[Math.multiplyExact(tokens, stride)];
        decodeOne(stored, layer, first, tokens, decoded, key);
        System.arraycopy(decoded, (position - first) * stride,
                destination, destinationOffset, stride);
    }

    @Override
    public void readKeyRange(
            int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        readRange(layer, firstPosition, tokenCount, destination, destinationOffset, true);
    }

    @Override
    public void readValueRange(
            int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset) {
        readRange(layer, firstPosition, tokenCount, destination, destinationOffset, false);
    }

    private void readRange(
            int layer, int firstPosition, int tokenCount,
            float[] destination, int destinationOffset, boolean key) {
        ensureOpen();
        int stride = checkedRange(layer, firstPosition, tokenCount);
        int elements = Math.multiplyExact(tokenCount, stride);
        checkRange(destination, destinationOffset, elements, "destination");
        int completed = 0;
        while (completed < tokenCount) {
            int position = firstPosition + completed;
            int blockIndex = position / blockSize;
            int blockFirst = blockStart(blockIndex);
            int blockTokens = blockTokenCount(blockFirst);
            int count = Math.min(blockTokens - (position - blockFirst), tokenCount - completed);
            StoredBlock stored = layers.get(layer).get(blockIndex);
            int targetOffset = destinationOffset + completed * stride;
            if (stored == null) {
                java.util.Arrays.fill(destination, targetOffset, targetOffset + count * stride, 0.0f);
            } else {
                float[] decoded = new float[Math.multiplyExact(blockTokens, stride)];
                decodeOne(stored, layer, blockFirst, blockTokens, decoded, key);
                for (int token = 0; token < count; token++) {
                    int local = position - blockFirst + token;
                    int target = targetOffset + token * stride;
                    if (stored.written().get(local)) {
                        System.arraycopy(decoded, local * stride, destination, target, stride);
                    } else {
                        java.util.Arrays.fill(destination, target, target + stride, 0.0f);
                    }
                }
            }
            completed += count;
        }
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
        layers.forEach(Map::clear);
    }

    @Override
    public void truncate(int firstPosition) {
        ensureOpen();
        if (firstPosition < 0 || firstPosition > layout.maxSequenceLength()) {
            throw new IllegalArgumentException(
                    "firstPosition must be within cache capacity: " + firstPosition);
        }
        int firstRemovedBlock = (firstPosition + blockSize - 1) / blockSize;
        for (int layer = 0; layer < layout.layerCount(); layer++) {
            Map<Integer, StoredBlock> blocks = layers.get(layer);
            blocks.keySet().removeIf(index -> index >= firstRemovedBlock);
            int partialBlockIndex = firstPosition / blockSize;
            int localCut = firstPosition % blockSize;
            if (localCut != 0) {
                StoredBlock previous = blocks.get(partialBlockIndex);
                if (previous != null) {
                    int first = blockStart(partialBlockIndex);
                    int tokens = blockTokenCount(first);
                    int stride = layout.layerStride(layer);
                    float[] keys = new float[Math.multiplyExact(tokens, stride)];
                    float[] values = new float[keys.length];
                    decode(previous, layer, first, tokens, keys, values);
                    BitSet written = previous.written();
                    written.clear(localCut, tokens);
                    java.util.Arrays.fill(keys, localCut * stride, keys.length, 0.0f);
                    java.util.Arrays.fill(values, localCut * stride, values.length, 0.0f);
                    if (written.isEmpty()) {
                        blocks.remove(partialBlockIndex);
                    } else {
                        blocks.put(partialBlockIndex,
                                encode(layer, first, tokens, keys, values, written));
                    }
                }
            }
        }
    }

    /** Number of token positions explicitly written across all layers. */
    public long storedPositionCount() {
        ensureOpen();
        long count = 0;
        for (Map<Integer, StoredBlock> layer : layers) {
            for (StoredBlock block : layer.values()) {
                count = Math.addExact(count, block.written().cardinality());
            }
        }
        return count;
    }

    public long encodedPayloadBytes() {
        ensureOpen();
        long bytes = 0;
        for (Map<Integer, StoredBlock> layer : layers) {
            for (StoredBlock block : layer.values()) {
                bytes = Math.addExact(bytes,
                        (long) block.key().payloadSizeBytes() + block.value().payloadSizeBytes());
            }
        }
        return bytes;
    }

    public long encodedMetadataBytes() {
        ensureOpen();
        long bytes = 0;
        for (Map<Integer, StoredBlock> layer : layers) {
            for (StoredBlock block : layer.values()) {
                bytes = Math.addExact(bytes,
                        (long) block.key().metadataSizeBytes() + block.value().metadataSizeBytes());
            }
        }
        return bytes;
    }

    public int storedBlockCount() {
        ensureOpen();
        int count = 0;
        for (Map<Integer, StoredBlock> layer : layers) {
            count = Math.addExact(count, layer.size());
        }
        return count;
    }

    @Override
    public void close() {
        if (!closed) {
            layers.forEach(Map::clear);
            closed = true;
        }
    }

    private StoredBlock encode(
            int layer, int firstPosition, int tokens,
            float[] keys, float[] values, BitSet written) {
        KVCacheEncodedBlock encodedKey = codec.encode(
                keys, 0, context(layer, firstPosition, tokens, KVCacheTensorKind.KEY));
        KVCacheEncodedBlock encodedValue = codec.encode(
                values, 0, context(layer, firstPosition, tokens, KVCacheTensorKind.VALUE));
        return new StoredBlock(encodedKey, encodedValue, (BitSet) written.clone());
    }

    private void decode(
            StoredBlock stored, int layer, int firstPosition, int tokens,
            float[] keys, float[] values) {
        codec.decode(stored.key(), keys, 0,
                context(layer, firstPosition, tokens, KVCacheTensorKind.KEY));
        codec.decode(stored.value(), values, 0,
                context(layer, firstPosition, tokens, KVCacheTensorKind.VALUE));
    }

    private void decodeOne(
            StoredBlock stored, int layer, int firstPosition, int tokens,
            float[] destination, boolean key) {
        codec.decode(key ? stored.key() : stored.value(), destination, 0,
                context(layer, firstPosition, tokens,
                        key ? KVCacheTensorKind.KEY : KVCacheTensorKind.VALUE));
    }

    private KVCacheCompressionContext context(
            int layer, int firstPosition, int tokens, KVCacheTensorKind kind) {
        return new KVCacheCompressionContext(
                layer, firstPosition, tokens,
                layout.kvHeads(layer), layout.headDimension(layer), kind);
    }

    private int checkedPosition(int layer, int position) {
        checkedLayer(layer);
        if (position < 0 || position >= layout.maxSequenceLength()) {
            throw new IndexOutOfBoundsException("Cache position out of range: " + position);
        }
        return layout.layerStride(layer);
    }

    private int checkedRange(int layer, int firstPosition, int tokenCount) {
        checkedLayer(layer);
        if (firstPosition < 0 || tokenCount < 0
                || firstPosition > layout.maxSequenceLength()
                || tokenCount > layout.maxSequenceLength() - firstPosition) {
            throw new IndexOutOfBoundsException("KV cache range exceeds layer capacity");
        }
        return layout.layerStride(layer);
    }

    private void checkedLayer(int layer) {
        if (layer < 0 || layer >= layout.layerCount()) {
            throw new IndexOutOfBoundsException("Layer index out of range: " + layer);
        }
    }

    private int blockStart(int blockIndex) {
        return Math.multiplyExact(blockIndex, blockSize);
    }

    private int blockTokenCount(int firstPosition) {
        return Math.min(blockSize, layout.maxSequenceLength() - firstPosition);
    }

    private static void checkRange(float[] array, int offset, int length, String name) {
        Objects.requireNonNull(array, name);
        if (offset < 0 || length < 0 || offset > array.length - length) {
            throw new IndexOutOfBoundsException(name + " range is outside the supplied array");
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Block-compressed KV cache storage is closed");
        }
    }

    private record StoredBlock(KVCacheEncodedBlock key, KVCacheEncodedBlock value, BitSet written) {
        private StoredBlock {
            Objects.requireNonNull(key, "key");
            Objects.requireNonNull(value, "value");
            written = (BitSet) Objects.requireNonNull(written, "written").clone();
        }

        @Override
        public BitSet written() {
            return (BitSet) written.clone();
        }
    }
}
