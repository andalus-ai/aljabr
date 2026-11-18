package tech.kayys.aljabr.cache;

import java.util.Arrays;

/**
 * Immutable, format-neutral description of the logical geometry of a
 * transformer key/value cache.
 *
 * <p>The layout describes per-layer KV head counts and head dimensions without
 * prescribing how or where K/V data is stored. A backend may use heap arrays,
 * direct memory, native memory, or device allocations while sharing this
 * logical description.
 */
public final class KVCacheLayout {
    private final int maxSequenceLength;
    private final int[] kvHeadsPerLayer;
    private final int[] headDimensionsPerLayer;
    private final int[] layerStrides;

    /**
     * Creates a layout with potentially different KV geometry for each layer.
     *
     * @param maxSequenceLength maximum number of cached sequence positions
     * @param kvHeadsPerLayer number of KV heads for each layer
     * @param headDimensionsPerLayer K/V head dimension for each layer
     */
    public KVCacheLayout(
            int maxSequenceLength,
            int[] kvHeadsPerLayer,
            int[] headDimensionsPerLayer) {
        if (maxSequenceLength <= 0) {
            throw new IllegalArgumentException("maxSequenceLength must be positive");
        }
        if (kvHeadsPerLayer == null || headDimensionsPerLayer == null
                || kvHeadsPerLayer.length == 0
                || kvHeadsPerLayer.length != headDimensionsPerLayer.length) {
            throw new IllegalArgumentException(
                    "KV head and head-dimension arrays must be non-empty and have equal lengths");
        }

        this.maxSequenceLength = maxSequenceLength;
        this.kvHeadsPerLayer = kvHeadsPerLayer.clone();
        this.headDimensionsPerLayer = headDimensionsPerLayer.clone();
        this.layerStrides = new int[kvHeadsPerLayer.length];

        for (int layer = 0; layer < this.kvHeadsPerLayer.length; layer++) {
            int heads = this.kvHeadsPerLayer[layer];
            int headDimension = this.headDimensionsPerLayer[layer];
            if (heads <= 0 || headDimension <= 0) {
                throw new IllegalArgumentException(
                        "KV heads and head dimension must be positive for layer " + layer);
            }
            try {
                layerStrides[layer] = Math.multiplyExact(heads, headDimension);
                Math.multiplyExact(maxSequenceLength, layerStrides[layer]);
            } catch (ArithmeticException overflow) {
                throw new IllegalArgumentException(
                        "KV cache geometry exceeds supported array indexing at layer " + layer, overflow);
            }
        }
    }

    /** Creates a uniform layout shared by every layer. */
    public static KVCacheLayout uniform(
            int layerCount, int maxSequenceLength, int kvHeads, int headDimension) {
        if (layerCount <= 0) {
            throw new IllegalArgumentException("layerCount must be positive");
        }
        int[] heads = new int[layerCount];
        int[] dimensions = new int[layerCount];
        Arrays.fill(heads, kvHeads);
        Arrays.fill(dimensions, headDimension);
        return new KVCacheLayout(maxSequenceLength, heads, dimensions);
    }

    public int layerCount() {
        return kvHeadsPerLayer.length;
    }

    public int maxSequenceLength() {
        return maxSequenceLength;
    }

    public int kvHeads(int layer) {
        return kvHeadsPerLayer[checkedLayer(layer)];
    }

    public int headDimension(int layer) {
        return headDimensionsPerLayer[checkedLayer(layer)];
    }

    /** Number of floats in one token position for the specified layer. */
    public int layerStride(int layer) {
        return layerStrides[checkedLayer(layer)];
    }

    /** Returns a defensive copy of the per-layer KV head counts. */
    public int[] kvHeadsPerLayer() {
        return kvHeadsPerLayer.clone();
    }

    /** Returns a defensive copy of the per-layer head dimensions. */
    public int[] headDimensionsPerLayer() {
        return headDimensionsPerLayer.clone();
    }

    private int checkedLayer(int layer) {
        if (layer < 0 || layer >= layerCount()) {
            throw new IndexOutOfBoundsException("Layer index out of range: " + layer);
        }
        return layer;
    }
}
