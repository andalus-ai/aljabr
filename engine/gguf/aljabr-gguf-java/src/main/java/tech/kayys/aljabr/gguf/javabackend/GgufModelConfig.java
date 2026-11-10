package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFModel;

import java.util.List;
import java.util.Map;

/**
 * Transformer hyperparameters read straight out of {@link GGUFModel#metadata()}.
 */
record GgufModelConfig(
        String architecture,
        int contextLength,
        int embeddingDim,
        int nLayers,
        int ffnDim,
        int nHeads,
        int nKVHeads,           // max across layers; equals uniform value when uniform
        int headDim,            // max Q head dim
        int headDimKV,          // max KV head dim
        int vocabSize,
        float rmsNormEps,
        float ropeFreqBase,     // global rope freq base (for global layers)
        float ropeFreqBaseSwa,  // SWA rope freq base (e.g. 10_000 for SWA)
        int bosTokenId,
        int eosTokenId,
        int[] nKVHeadsPerLayer, // null when uniform; non-null when heterogeneous
        int[] headDimPerLayer,  // null when uniform; non-null when heterogeneous
        int ropeDimensionCount, // rotary dimensions for global/full-attention layers; 0 means architecture default
        int ropeDimensionCountSwa, // rotary dimensions for SWA layers; 0 means architecture default
        int[] swaPattern,       // 1=SWA, 0=global per layer
        int slidingWindow,      // sliding window size (e.g. 1024 / 512, 0 if disabled)
        int nEmbdPerLayer,      // per-layer token embedding dimension (0 if absent)
        float finalLogitSoftcap,// softcapping factor (0 if disabled)
        float attentionScale,   // explicit attention scale factor (<= 0 for 1/sqrt(headDim))
        float embeddingScale,   // token embedding multiplier (e.g. sqrt(dim) or 1.0)
        boolean hasUnitVNorm,   // whether V has unscaled/unit RMSNorm (no learnable scale tensor)
        boolean isGeGLU,        // whether FFN uses GeGLU vs SwiGLU
        boolean isGemma4        // backward compatibility convenience flag
) {

    static GgufModelConfig from(GGUFModel model) {
        Map<String, Object> meta = model.metadata();
        String arch = String.valueOf(meta.getOrDefault("general.architecture", "llama")).toLowerCase(java.util.Locale.ROOT);
        boolean isGemma4 = "gemma4".equals(arch) || "gemma3".equals(arch);
        boolean isGemmaFamily = arch.startsWith("gemma");

        int embeddingDim = intMeta(meta, arch, "embedding_length", 4096);
        int nHeads        = intMeta(meta, arch, "attention.head_count", 32);

        int nKVHeads = intOrListMaxMeta(meta, arch, "attention.head_count_kv", nHeads);

        int defaultHeadDim = nHeads == 0 ? embeddingDim : embeddingDim / nHeads;
        int headDim   = intMeta(meta, arch, "attention.key_length",   defaultHeadDim);
        int headDimKV = intMeta(meta, arch, "attention.value_length", headDim);

        int[] nKVHPerLayer  = intListMeta(meta, arch, "attention.head_count_kv");
        int[] swaPattern    = intListMeta(meta, arch, "attention.sliding_window_pattern");
        int ropeDimensionCount = intMeta(meta, arch, "rope.dimension_count", 0);
        int ropeDimensionCountSwa = intMeta(meta, arch, "rope.dimension_count_swa", 0);
        int slidingWindow   = intMeta(meta, arch, "attention.sliding_window", 0);
        int nEmbdPerLayer   = intMeta(meta, arch, "embedding_length_per_layer", 0);
        int[] headDimPerLyr = null;

        if (swaPattern != null) {
            int headDimGlobal = intMeta(meta, arch, "attention.key_length",     defaultHeadDim);
            int headDimSwa    = intMeta(meta, arch, "attention.key_length_swa", headDimGlobal);
            headDimPerLyr = new int[swaPattern.length];
            for (int i = 0; i < swaPattern.length; i++) {
                headDimPerLyr[i] = (swaPattern[i] == 1) ? headDimSwa : headDimGlobal;
            }
            headDim   = Math.max(headDimGlobal, headDimSwa);
            headDimKV = headDim;
        }

        float finalLogitSoftcap = floatMeta(meta, arch, "final_logit_softcapping", 0f);

        // Attention scale: prefer metadata key, then query_pre_attn_scalar, else 0 (→ use 1/sqrt(headDim) per layer)
        float attentionScale = floatMeta(meta, arch, "attention.scale",
                floatMeta(meta, arch, "attention.query_pre_attn_scalar", 0f));

        // Embedding scale: prefer metadata key, then scale_embedding flag, then architecture default
        float embeddingScale = floatMeta(meta, arch, "embedding_scale", 0f);
        if (embeddingScale <= 0f) {
            boolean scaleEmbMeta = boolMeta(meta, arch, "scale_embedding", false);
            if (scaleEmbMeta || isGemmaFamily) {
                embeddingScale = (float) Math.sqrt(embeddingDim);
            } else {
                embeddingScale = 1.0f;
            }
        }

        // Unit V norm: whether V is normalized with unit weights (no learnable scale tensor)
        boolean hasUnitVNorm = boolMeta(meta, arch, "attention.v_norm_unit",
                boolMeta(meta, arch, "attention.unit_v_norm", isGemma4));

        // Activation: GeGLU vs SwiGLU
        String gatingFunc = stringMeta(meta, arch, "feed_forward_gating_func",
                stringMeta(meta, arch, "feed_forward_activation", null));
        boolean isGeGLU = gatingFunc != null
                ? gatingFunc.toLowerCase(java.util.Locale.ROOT).contains("gelu")
                : isGemmaFamily;

        float ropeFreqBase = floatMeta(meta, arch, "rope.freq_base", isGemmaFamily ? 1_000_000f : 10_000f);
        float ropeFreqBaseSwa = floatMeta(meta, arch, "rope.freq_base_swa", 10_000f);

        int rawContextLength = intMeta(meta, arch, "context_length", 4096);
        int maxJavaCtx = Integer.getInteger("aljabr.gguf.java.max_context", 1024);
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
                ropeFreqBaseSwa,
                intMeta(meta, arch, "tokenizer.bos_token_id", intOf(meta, "tokenizer.ggml.bos_token_id", 1)),
                intMeta(meta, arch, "tokenizer.eos_token_id", intOf(meta, "tokenizer.ggml.eos_token_id", 2)),
                nKVHPerLayer,
                headDimPerLyr,
                ropeDimensionCount,
                ropeDimensionCountSwa,
                swaPattern,
                slidingWindow,
                nEmbdPerLayer,
                finalLogitSoftcap,
                attentionScale,
                embeddingScale,
                hasUnitVNorm,
                isGeGLU,
                isGemma4);
    }

    int nKVHeadsForLayer(int layer) {
        if (nKVHeadsPerLayer != null && layer >= 0 && layer < nKVHeadsPerLayer.length) {
            return nKVHeadsPerLayer[layer];
        }
        return nKVHeads;
    }

    int headDimForLayer(int layer) {
        if (headDimPerLayer != null && layer >= 0 && layer < headDimPerLayer.length) {
            return headDimPerLayer[layer];
        }
        return headDim;
    }

    boolean isSwaLayer(int layer) {
        if (swaPattern != null && layer >= 0 && layer < swaPattern.length) {
            return swaPattern[layer] == 1;
        }
        return false;
    }

    int rotaryDimForLayer(int layer) {
        int layerHeadDim = headDimForLayer(layer);
        return resolveRotaryDimension(
                architecture,
                isSwaLayer(layer),
                layerHeadDim,
                isSwaLayer(layer) ? ropeDimensionCountSwa : ropeDimensionCount);
    }

    /**
     * Resolves the number of head dimensions that participate in RoPE.
     *
     * GGUF metadata takes precedence. Gemma 4's architecture default uses full
     * rotary dimensions for sliding-window layers and one quarter of the head
     * dimension for global-attention layers (proportional RoPE).
     */
    static int resolveRotaryDimension(
            String architecture, boolean swaLayer, int headDim, int configuredDimension) {
        if (headDim <= 0) {
            throw new IllegalArgumentException("headDim must be positive");
        }

        int resolved;
        if (configuredDimension > 0) {
            resolved = Math.min(headDim, configuredDimension);
        } else if ("gemma4".equalsIgnoreCase(architecture)) {
            resolved = swaLayer ? headDim : Math.max(2, (headDim / 4) & ~1);
        } else {
            resolved = headDim;
        }

        // RoPE rotates pairs of dimensions, so an odd configuration cannot be
        // applied as-is. Keep the nearest lower even dimension.
        if ((resolved & 1) != 0) {
            resolved--;
        }
        if (resolved < 2) {
            throw new IllegalArgumentException("rotary dimension must be at least 2");
        }
        return resolved;
    }

    float ropeFreqBaseForLayer(int layer) {
        if (isSwaLayer(layer)) {
            return ropeFreqBaseSwa;
        }
        return ropeFreqBase;
    }

    private static int vocabSizeOf(Map<String, Object> meta) {
        Object tokens = meta.getOrDefault("tokenizer.ggml.tokens", meta.get("tokenizer.tokens"));
        return tokens instanceof List<?> list ? list.size() : 32000;
    }

    private static int intOrListMaxMeta(Map<String, Object> meta, String arch, String suffix, int fallback) {
        Object v = meta.get(arch + "." + suffix);
        if (v instanceof Number n) return n.intValue();
        if (v instanceof List<?> list && !list.isEmpty()) {
            int max = 0;
            for (Object item : list) {
                if (item instanceof Number n) max = Math.max(max, n.intValue());
            }
            return max > 0 ? max : fallback;
        }
        return fallback;
    }

    private static int[] intListMeta(Map<String, Object> meta, String arch, String suffix) {
        Object v = meta.get(arch + "." + suffix);
        if (v instanceof List<?> list && !list.isEmpty()) {
            int[] arr = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                if (list.get(i) instanceof Number n) arr[i] = n.intValue();
                else if (list.get(i) instanceof Boolean b) arr[i] = b ? 1 : 0;
            }
            return arr;
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

    private static boolean boolMeta(Map<String, Object> meta, String arch, String suffix, boolean fallback) {
        Object v = meta.get(arch + "." + suffix);
        if (v == null) v = meta.get(suffix);
        if (v instanceof Boolean b) return b;
        if (v instanceof Number n) return n.intValue() != 0;
        if (v instanceof String s) return Boolean.parseBoolean(s);
        return fallback;
    }

    private static String stringMeta(Map<String, Object> meta, String arch, String suffix, String fallback) {
        Object v = meta.get(arch + "." + suffix);
        if (v == null) v = meta.get(suffix);
        return v != null ? String.valueOf(v) : fallback;
    }
}
