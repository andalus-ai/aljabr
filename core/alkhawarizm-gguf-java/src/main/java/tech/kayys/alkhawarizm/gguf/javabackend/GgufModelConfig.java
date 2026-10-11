package tech.kayys.alkhawarizm.gguf.javabackend;

import tech.kayys.alkhawarizm.gguf.loader.GGUFModel;

import java.util.List;
import java.util.Map;

/**
 * Transformer hyperparameters read straight out of {@link GGUFModel#metadata()}.
 *
 * <p>This intentionally does NOT reuse either of the two {@code ModelConfig}
 * classes already in the codebase:</p>
 * <ul>
 *   <li>{@code tech.kayys.alkhawarizm.gguf.model.ModelConfig} (a record) is
 *       only ever constructed from a {@code GGUFFile}, a different loaded-file
 *       representation than the {@code GGUFModel} this runner actually
 *       produces — there is no {@code fromGGUFModel} path for it.</li>
 *   <li>The {@code ModelConfig} that {@code LlamaForward}/{@code LlamaWeights}
 *       expect (bare, unimported, in their own {@code gguf.loader.model}
 *       package) doesn't resolve to any class in this codebase at all — see
 *       the migration notes for the full account.</li>
 * </ul>
 *
 * <p>Also fixes a real correctness gap in both of those: neither reads keys
 * prefixed by the model's actual {@code general.architecture} (they either
 * hardcode {@code "llama."} or don't prefix at all), which silently breaks
 * on Qwen/Mistral/Gemma-family GGUF files where hyperparameters live under
 * e.g. {@code qwen2.*} instead of {@code llama.*}.</p>
 *
 * <p><b>Gemma 4 hybrid-attention support:</b> Gemma 4 stores
 * {@code gemma4.attention.head_count_kv} as a per-layer array (not a scalar)
 * and has two different head dimensions: {@code key_length=512} for global
 * attention layers and {@code key_length_swa=256} for sliding-window layers.
 * {@link #nKVHeads()} returns the <em>maximum</em> across all layers (for
 * worst-case buffer allocation) and {@link #nKVHeadsPerLayer()} exposes the
 * raw per-layer array when available.</p>
 */
record GgufModelConfig(
        String architecture,
        int contextLength,
        int embeddingDim,
        int nLayers,
        int ffnDim,
        int nHeads,
        int nKVHeads,           // max across layers; equals uniform value for non-Gemma4
        int headDim,            // Q head dim (key_length for global, or key_length_swa for SWA)
        int headDimKV,          // KV head dim — may differ from headDim on Gemma 4
        int vocabSize,
        float rmsNormEps,
        float ropeFreqBase,
        int bosTokenId,
        int eosTokenId,
        int[] nKVHeadsPerLayer, // null when uniform; non-null for Gemma 4 (48 entries)
        int[] headDimPerLayer,  // null when uniform; non-null for Gemma 4 (48 entries)
        boolean isGemma4,
        float finalLogitSoftcap // Gemma-specific; 0 means disabled
) {

    static GgufModelConfig from(GGUFModel model) {
        Map<String, Object> meta = model.metadata();
        String arch = String.valueOf(meta.getOrDefault("general.architecture", "llama"));
        boolean isGemma4 = "gemma4".equals(arch) || "gemma3".equals(arch);

        int embeddingDim = intMeta(meta, arch, "embedding_length", 4096);
        int nHeads        = intMeta(meta, arch, "attention.head_count", 32);

        // head_count_kv may be a List<Integer> on Gemma 4 — take max for buffer sizing
        int nKVHeads = intOrListMaxMeta(meta, arch, "attention.head_count_kv", nHeads);

        // head dim: Q uses key_length; KV may differ (Gemma 4 value_length)
        int defaultHeadDim = nHeads == 0 ? embeddingDim : embeddingDim / nHeads;
        int headDim   = intMeta(meta, arch, "attention.key_length",   defaultHeadDim);
        int headDimKV = intMeta(meta, arch, "attention.value_length", headDim);

        // Per-layer arrays for Gemma 4
        int[] nKVHPerLayer  = null;
        int[] headDimPerLyr = null;
        if (isGemma4) {
            nKVHPerLayer  = intListMeta(meta, arch, "attention.head_count_kv");
            // Sliding window pattern: 1=SWA (key_length_swa), 0=global (key_length)
            int[] swaPat = intListMeta(meta, arch, "attention.sliding_window_pattern");
            if (swaPat != null) {
                int headDimGlobal = intMeta(meta, arch, "attention.key_length",     defaultHeadDim);
                int headDimSwa    = intMeta(meta, arch, "attention.key_length_swa", headDimGlobal);
                headDimPerLyr = new int[swaPat.length];
                for (int i = 0; i < swaPat.length; i++) {
                    headDimPerLyr[i] = (swaPat[i] == 1) ? headDimSwa : headDimGlobal;
                }
                // headDim for global buffers: use the maximum
                headDim   = Math.max(headDimGlobal, headDimSwa);
                headDimKV = headDim;
            }
        }

        float finalLogitSoftcap = floatMeta(meta, arch, "final_logit_softcapping", 0f);
        // Gemma 4 rope freq base differs per layer type; use global (1_000_000) as default
        float ropeFreqBase = isGemma4
                ? floatMeta(meta, arch, "rope.freq_base", 1_000_000f)
                : floatMeta(meta, arch, "rope.freq_base", 10_000f);

        int rawContextLength = intMeta(meta, arch, "context_length", 4096);
        // Cap context length for pure Java execution (default 1024 max) to fit comfortably within heap
        int maxJavaCtx = Integer.getInteger("alkhawarizm.gguf.java.max_context", 1024);
        int contextLength = Math.min(rawContextLength, maxJavaCtx);

        return new GgufModelConfig(
                arch,
                contextLength,
                embeddingDim,
                intMeta(meta, arch, "block_count", 32),
                intMeta(meta, arch, "feed_forward_length", embeddingDim * 4),
                nHeads,
                nKVHeads,
                headDim,
                headDimKV,
                vocabSizeOf(meta),
                floatMeta(meta, arch, "attention.layer_norm_rms_epsilon", 1e-5f),
                ropeFreqBase,
                intOf(meta, "tokenizer.ggml.bos_token_id", 1),
                intOf(meta, "tokenizer.ggml.eos_token_id", 2),
                nKVHPerLayer,
                headDimPerLyr,
                isGemma4,
                finalLogitSoftcap);
    }

    /** Return the nKVHeads for a specific layer index (falls back to {@link #nKVHeads()}). */
    int nKVHeadsForLayer(int layer) {
        if (nKVHeadsPerLayer != null && layer >= 0 && layer < nKVHeadsPerLayer.length) {
            return nKVHeadsPerLayer[layer];
        }
        return nKVHeads;
    }

    /** Return the head dimension for a specific layer (falls back to {@link #headDim()}). */
    int headDimForLayer(int layer) {
        if (headDimPerLayer != null && layer >= 0 && layer < headDimPerLayer.length) {
            return headDimPerLayer[layer];
        }
        return headDim;
    }

    // ── helpers ─────────────────────────────────────────────────────────────

    private static int vocabSizeOf(Map<String, Object> meta) {
        Object tokens = meta.getOrDefault("tokenizer.ggml.tokens", meta.get("tokenizer.tokens"));
        return tokens instanceof List<?> list ? list.size() : 32000;
    }

    /** Reads a scalar int OR a List<Number> (returns max of the list). */
    private static int intOrListMaxMeta(Map<String, Object> meta, String arch, String suffix, int fallback) {
        Object v = meta.get(arch + "." + suffix);
        if (v instanceof Number n) return n.intValue();
        if (v instanceof List<?> list && !list.isEmpty()) {
            int max = fallback;
            for (Object item : list) {
                if (item instanceof Number n) max = Math.max(max, n.intValue());
            }
            return max;
        }
        return fallback;
    }

    /** Reads a per-layer int array from GGUF metadata. Returns null if absent or not a list. */
    private static int[] intListMeta(Map<String, Object> meta, String arch, String suffix) {
        Object v = meta.get(arch + "." + suffix);
        if (v instanceof List<?> list && !list.isEmpty()) {
            int[] arr = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                Object item = list.get(i);
                if (item instanceof Number n) arr[i] = n.intValue();
                else if (item instanceof Boolean b) arr[i] = b ? 1 : 0;
            }
            return arr;
        }
        if (v instanceof Number n) {
            // Scalar — not per-layer, return null to signal "uniform"
            return null;
        }
        return null;
    }

    private static int intMeta(Map<String, Object> meta, String arch, String suffix, int fallback) {
        return intOf(meta, arch + "." + suffix, fallback);
    }

    private static float floatMeta(Map<String, Object> meta, String arch, String suffix, float fallback) {
        return floatOf(meta, arch + "." + suffix, fallback);
    }

    private static int intOf(Map<String, Object> meta, String key, int fallback) {
        Object v = meta.get(key);
        return v instanceof Number n ? n.intValue() : fallback;
    }

    private static float floatOf(Map<String, Object> meta, String key, float fallback) {
        Object v = meta.get(key);
        return v instanceof Number n ? n.floatValue() : fallback;
    }
}
