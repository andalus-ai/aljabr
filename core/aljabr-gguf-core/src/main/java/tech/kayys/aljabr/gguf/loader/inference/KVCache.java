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
        this.maxSeq      = maxSeq;
        this.layerNKVH   = nKVHeadsPerLayer.clone();
        this.layerHDim   = headDimPerLayer.clone();
        this.layerStride = new int[nLayers];
        this.k = new float[nLayers][];
        this.v = new float[nLayers][];
        for (int i = 0; i < nLayers; i++) {
            int stride = nKVHeadsPerLayer[i] * headDimPerLayer[i];
            layerStride[i] = stride;
            k[i] = new float[maxSeq * stride];
            v[i] = new float[maxSeq * stride];
        }
    }

    /**
     * Uniform constructor — all layers share the same {@code nKVHeads} and
     * {@code headDim}. Equivalent to calling the per-layer constructor with
     * identical arrays.
     */
    public KVCache(int nLayers, int maxSeq, int nKVHeads, int headDim) {
        this.maxSeq      = maxSeq;
        this.layerNKVH   = new int[nLayers];
        this.layerHDim   = new int[nLayers];
        this.layerStride = new int[nLayers];
        this.k = new float[nLayers][];
        this.v = new float[nLayers][];
        int stride = nKVHeads * headDim;
        for (int i = 0; i < nLayers; i++) {
            layerNKVH[i]   = nKVHeads;
            layerHDim[i]   = headDim;
            layerStride[i] = stride;
            k[i] = new float[maxSeq * stride];
            v[i] = new float[maxSeq * stride];
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
        int base = seqLen * layerStride[layer];
        int len  = layerStride[layer];
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
        int base = pos * layerStride[layer];
        System.arraycopy(kVec, 0, k[layer], base, len);
        System.arraycopy(vVec, 0, v[layer], base, len);
    }

    /** Advance the sequence pointer by one position (call after all layers are stored). */
    public void advance() { seqLen++; }

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
        int hd  = layerHDim[layer];
        int src = pos * layerStride[layer] + kvHead * hd;
        System.arraycopy(k[layer], src, dst, dstOff, hd);
    }

    /**
     * Copy one KV-head's V vector at sequence position {@code pos} into {@code dst}.
     */
    public void readV(int layer, int pos, int kvHead, float[] dst, int dstOff) {
        int hd  = layerHDim[layer];
        int src = pos * layerStride[layer] + kvHead * hd;
        System.arraycopy(v[layer], src, dst, dstOff, hd);
    }

    /** Raw K backing array for layer (allows direct indexed access in the attention loop). */
    public float[] kLayer(int layer) { return k[layer]; }

    /** Raw V backing array for layer. */
    public float[] vLayer(int layer) { return v[layer]; }

    /** Per-layer stride in floats ({@code nKVHeads[layer] * headDim[layer]}). */
    public int headStride(int layer)  { return layerStride[layer]; }

    /** Number of KV heads for {@code layer}. */
    public int nKVHeads(int layer)    { return layerNKVH[layer]; }

    /** Head dimension (K/V) for {@code layer}. */
    public int headDim(int layer)     { return layerHDim[layer]; }

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

    /** Zero-fill and reset the sequence pointer. */
    public void reset() {
        seqLen = 0;
        for (float[] layer : k) java.util.Arrays.fill(layer, 0f);
        for (float[] layer : v) java.util.Arrays.fill(layer, 0f);
    }
}
