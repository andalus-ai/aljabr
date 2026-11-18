package tech.kayys.aljabr.gguf.loader.inference;

/**
 * Per-layer key-value cache for auto-regressive transformer decoding.
 *
 * <p>Supports models where individual layers have different numbers of KV heads
 * and/or different head dimensions (e.g. Gemma 4 hybrid-attention: SWA layers
 * use 1 KV head × 256 dim, global layers use 8 KV heads × 512 dim).
 *
 * <p>Each layer has its own flat float array sized:
 * {@code maxSeq × nKVHeads[layer] × headDim[layer]}.
 * The per-layer stride is {@code nKVHeads[layer] × headDim[layer]}.
 *
 * <p>For uniform models (all layers share the same nKVHeads and headDim) use
 * {@link #KVCache(int, int, int, int)}; it delegates to the per-layer
 * constructor with identical arrays.
 */
public final class KVCache {
    private final float[][] k;          // k[layer][pos * layerStride[layer] + ...]
    private final float[][] v;          // v[layer][...]
    private final int[] layerStride;    // nKVHeads[layer] * headDim[layer]
    private final int[] layerNKVH;      // nKVHeads per layer
    private final int[] layerHDim;      // headDim per layer
    private final int maxSeq;
    private int seqLen = 0;

    /**
     * Per-layer constructor. {@code nKVHeadsPerLayer} and {@code headDimPerLayer}
     * must have length {@code nLayers}.
     */
    public KVCache(int nLayers, int maxSeq, int[] nKVHeadsPerLayer, int[] headDimPerLayer) {
        validateDimensions(nLayers, maxSeq);
        if (nKVHeadsPerLayer == null || headDimPerLayer == null
                || nKVHeadsPerLayer.length != nLayers || headDimPerLayer.length != nLayers) {
            throw new IllegalArgumentException("Per-layer KV head and head-dimension arrays must match nLayers");
        }
        this.maxSeq      = maxSeq;
        this.layerNKVH   = nKVHeadsPerLayer.clone();
        this.layerHDim   = headDimPerLayer.clone();
        this.layerStride = new int[nLayers];
        this.k = new float[nLayers][];
        this.v = new float[nLayers][];
        for (int i = 0; i < nLayers; i++) {
            if (nKVHeadsPerLayer[i] <= 0 || headDimPerLayer[i] <= 0) {
                throw new IllegalArgumentException("KV heads and head dimension must be positive for layer " + i);
            }
            int stride = checkedProduct(nKVHeadsPerLayer[i], headDimPerLayer[i], "layer stride");
            layerStride[i] = stride;
            k[i] = new float[checkedProduct(maxSeq, stride, "layer cache size")];
            v[i] = new float[checkedProduct(maxSeq, stride, "layer cache size")];
        }
    }

    /**
     * Uniform constructor — all layers share the same {@code nKVHeads} and
     * {@code headDim}. Equivalent to calling the per-layer constructor with
     * identical arrays.
     */
    public KVCache(int nLayers, int maxSeq, int nKVHeads, int headDim) {
        validateDimensions(nLayers, maxSeq);
        if (nKVHeads <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("KV heads and head dimension must be positive");
        }
        this.maxSeq      = maxSeq;
        this.layerNKVH   = new int[nLayers];
        this.layerHDim   = new int[nLayers];
        this.layerStride = new int[nLayers];
        this.k = new float[nLayers][];
        this.v = new float[nLayers][];
        int stride = checkedProduct(nKVHeads, headDim, "layer stride");
        for (int i = 0; i < nLayers; i++) {
            layerNKVH[i]   = nKVHeads;
            layerHDim[i]   = headDim;
            layerStride[i] = stride;
            k[i] = new float[checkedProduct(maxSeq, stride, "layer cache size")];
            v[i] = new float[checkedProduct(maxSeq, stride, "layer cache size")];
        }
    }

    /**
     * Store K and V vectors for the current sequence position in {@code layer}.
     *
     * @param layer  layer index
     * @param kVec   K vector, length must be {@code nKVHeads[layer] * headDim[layer]}
     * @param vVec   V vector, same length requirement
     */
    public void store(int layer, float[] kVec, float[] vVec) {
        validateLayer(layer);
        if (seqLen >= maxSeq) {
            throw new IllegalStateException("KV cache capacity exceeded: " + seqLen + " >= " + maxSeq);
        }
        int base = seqLen * layerStride[layer];
        int len  = layerStride[layer];
        requireArrayLength(kVec, len, "kVec");
        requireArrayLength(vVec, len, "vVec");
        System.arraycopy(kVec, 0, k[layer], base, len);
        System.arraycopy(vVec, 0, v[layer], base, len);
    }

    /**
     * Store K and V vectors directly at an explicit sequence position (for batch prefill).
     *
     * @param layer   layer index
     * @param pos     explicit target position (0-based)
     * @param kVec    K vector
     * @param vVec    V vector
     * @param len     number of floats to copy ({@code nKVHeads[layer] * headDim[layer]})
     */
    public void storeDirect(int layer, int pos, float[] kVec, float[] vVec, int len) {
        validateLayer(layer);
        validateWriteRange(pos, len, layerStride[layer]);
        requireArrayLength(kVec, len, "kVec");
        requireArrayLength(vVec, len, "vVec");
        int base = pos * layerStride[layer];
        System.arraycopy(kVec, 0, k[layer], base, len);
        System.arraycopy(vVec, 0, v[layer], base, len);
    }

    /**
     * Store contiguous K and V vector sequences for a batch of tokens starting at an explicit position.
     *
     * @param layer     layer index
     * @param startPos  explicit target starting position (0-based)
     * @param kBatch    K vector batch buffer
     * @param kOff      offset into {@code kBatch}
     * @param vBatch    V vector batch buffer
     * @param vOff      offset into {@code vBatch}
     * @param count     number of floats to copy (e.g. {@code batchSize * layerStride[layer]})
     */
    public void storeBatch(int layer, int startPos, float[] kBatch, int kOff, float[] vBatch, int vOff, int count) {
        validateLayer(layer);
        if (count < 0 || count % layerStride[layer] != 0) {
            throw new IllegalArgumentException("count must be a non-negative multiple of the layer stride");
        }
        int tokenCount = count / layerStride[layer];
        if (startPos < 0 || startPos > maxSeq || tokenCount > maxSeq - startPos) {
            throw new IndexOutOfBoundsException("KV batch write exceeds cache capacity");
        }
        requireArrayRange(kBatch, kOff, count, "kBatch");
        requireArrayRange(vBatch, vOff, count, "vBatch");
        int base = startPos * layerStride[layer];
        System.arraycopy(kBatch, kOff, k[layer], base, count);
        System.arraycopy(vBatch, vOff, v[layer], base, count);
    }

    /** Advance the sequence pointer by one position (call after all layers are stored). */
    public void advance() { advance(1); }

    /** Advance the sequence pointer by multiple positions. */
    public void advance(int n) {
        if (n < 0 || n > maxSeq - seqLen) {
            throw new IllegalArgumentException(
                    "Advance of " + n + " exceeds remaining KV cache capacity " + (maxSeq - seqLen));
        }
        seqLen += n;
    }

    /**
     * Copy one KV-head's K vector at sequence position {@code pos} into {@code dst}.
     *
     * @param layer   layer index
     * @param pos     sequence position (0 … seqLen-1)
     * @param kvHead  KV head index (0 … nKVHeads[layer]-1)
     * @param dst     destination buffer; must have at least {@code dstOff + headDim[layer]} floats
     * @param dstOff  offset into {@code dst}
     */
    public void readK(int layer, int pos, int kvHead, float[] dst, int dstOff) {
        validateRead(layer, pos, kvHead, dst, dstOff);
        int hd  = layerHDim[layer];
        int src = pos * layerStride[layer] + kvHead * hd;
        System.arraycopy(k[layer], src, dst, dstOff, hd);
    }

    /**
     * Copy one KV-head's V vector at sequence position {@code pos} into {@code dst}.
     */
    public void readV(int layer, int pos, int kvHead, float[] dst, int dstOff) {
        validateRead(layer, pos, kvHead, dst, dstOff);
        int hd  = layerHDim[layer];
        int src = pos * layerStride[layer] + kvHead * hd;
        System.arraycopy(v[layer], src, dst, dstOff, hd);
    }

    /** Raw K backing array for layer (allows direct indexed access in the attention loop). */
    public float[] kLayer(int layer) {
        validateLayer(layer);
        return k[layer];
    }

    /** Raw V backing array for layer. */
    public float[] vLayer(int layer) {
        validateLayer(layer);
        return v[layer];
    }

    /** Per-layer stride in floats ({@code nKVHeads[layer] * headDim[layer]}). */
    public int headStride(int layer)  {
        validateLayer(layer);
        return layerStride[layer];
    }

    /** Number of KV heads for {@code layer}. */
    public int nKVHeads(int layer)    {
        validateLayer(layer);
        return layerNKVH[layer];
    }

    /** Head dimension (K/V) for {@code layer}. */
    public int headDim(int layer)     {
        validateLayer(layer);
        return layerHDim[layer];
    }

    /** Current filled sequence length (same for all layers). */
    public int seqLen() { return seqLen; }

    /**
     * Legacy accessor — returns the stride of layer 0 (equivalent to the old
     * uniform {@code headStride()} field). Prefer {@link #headStride(int)}.
     *
     * @deprecated Use {@link #headStride(int layer)} for correctness with
     *             hybrid-attention models.
     */
    @Deprecated
    public int headStride() { return layerStride[0]; }

    private static void validateDimensions(int nLayers, int maxSeq) {
        if (nLayers <= 0 || maxSeq <= 0) {
            throw new IllegalArgumentException("nLayers and maxSeq must be positive");
        }
    }

    private static int checkedProduct(int left, int right, String description) {
        try {
            return Math.multiplyExact(left, right);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException(description + " exceeds supported array size", e);
        }
    }

    private void validateRead(int layer, int position, int kvHead, float[] destination, int destinationOffset) {
        validateLayer(layer);
        if (position < 0 || position >= seqLen) {
            throw new IndexOutOfBoundsException("KV read position is outside committed sequence length");
        }
        if (kvHead < 0 || kvHead >= layerNKVH[layer]) {
            throw new IndexOutOfBoundsException("KV head index out of range: " + kvHead);
        }
        requireArrayRange(destination, destinationOffset, layerHDim[layer], "destination");
    }

    private void validateLayer(int layer) {
        if (layer < 0 || layer >= k.length) {
            throw new IndexOutOfBoundsException("Layer index out of range: " + layer);
        }
    }

    private void validateWriteRange(int position, int length, int stride) {
        if (position < 0 || position >= maxSeq || length < 0 || length > stride) {
            throw new IndexOutOfBoundsException("KV write range is outside layer capacity");
        }
    }

    private static void requireArrayLength(float[] values, int length, String name) {
        if (values == null || values.length < length) {
            throw new IllegalArgumentException(name + " must contain at least " + length + " values");
        }
    }

    private static void requireArrayRange(float[] values, int offset, int length, String name) {
        if (values == null || offset < 0 || length < 0 || offset > values.length - length) {
            throw new IndexOutOfBoundsException(name + " range is outside the source array");
        }
    }

    /** Zero-fill and reset the sequence pointer. */
    public void reset() {
        seqLen = 0;
        for (float[] layer : k) java.util.Arrays.fill(layer, 0f);
        for (float[] layer : v) java.util.Arrays.fill(layer, 0f);
    }

    /** Reset the sequence pointer without zeroing out the underlying arrays. */
    public void resetFast() {
        this.seqLen = 0;
    }

    /**
     * Copies sequence positions 0 to numTokens from another cache into this cache.
     *
     * @param src source KVCache
     * @param numTokens number of sequence positions to copy
     */
    public void copyFrom(KVCache src, int numTokens) {
        if (src == null) {
            throw new IllegalArgumentException("Source KV cache must not be null");
        }
        if (numTokens < 0 || numTokens > maxSeq || numTokens > src.seqLen || src.k.length != k.length) {
            throw new IllegalArgumentException("Requested KV copy length is incompatible with source or destination");
        }
        int nLayers = k.length;
        // Validate every layer before copying any data. A late geometry mismatch
        // must not leave earlier layers overwritten under the destination's
        // previously committed sequence length.
        for (int l = 0; l < nLayers; l++) {
            if (src.layerStride[l] != layerStride[l]
                    || src.layerNKVH[l] != layerNKVH[l]
                    || src.layerHDim[l] != layerHDim[l]) {
                throw new IllegalArgumentException("KV cache layer geometry differs at layer " + l);
            }
        }
        for (int l = 0; l < nLayers; l++) {
            int count = numTokens * layerStride[l];
            System.arraycopy(src.k[l], 0, this.k[l], 0, count);
            System.arraycopy(src.v[l], 0, this.v[l], 0, count);
        }
        this.seqLen = numTokens;
    }
}
