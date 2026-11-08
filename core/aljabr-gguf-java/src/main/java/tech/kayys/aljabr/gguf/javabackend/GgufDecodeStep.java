package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;
import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

import java.util.Locale;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * One decoder forward pass: {@code token, position -> logits}, given a
 * running {@link KVCache}.
 *
 * <p>
 * Supports standard LLaMA/Mistral/Qwen models as well as Gemma family models
 * (Gemma 2, Gemma 3, Gemma 4) with embedding scaling, per-head Q/K/V RMSNorm,
 * post-attention and post-FFN RMSNorm, GeGLU activations, per-layer SWA / RoPE,
 * rope_freqs proportional rotary embedding, per-layer token embeddings, layer
 * output scaling, and final logit softcapping.
 * </p>
 *
 * <p>
 * Gemma 4 SWA layers use <em>bidirectional</em> attention during prefill
 * (all tokens in the prompt attend to each other within the sliding window),
 * matching the {@code LLAMA_NON_CAUSAL_TYPE_SWA_ONLY} behaviour in llama.cpp.
 * Use {@link #prefill(int[], KVCache)} for prompt processing and
 * {@link #forward(int, int, KVCache, boolean)} for single-token decode.
 * </p>
 */
final class GgufDecodeStep {
    private static final Logger LOG = Logger.getLogger(GgufDecodeStep.class.getName());

    public static boolean isVerbose() {
        return Boolean.getBoolean("aljabr.metal.verbose")
                || Boolean.getBoolean("aqli.verbose")
                || System.getenv("ALJABR_METAL_VERBOSE") != null
                || System.getenv("ALJABR_DEBUG") != null
                || System.getenv("AQLI_DEBUG") != null
                || LOG.isLoggable(Level.FINE);
    }

    private final GGUFModel model;
    private final GgufExecutionDispatcher execution;
    private final GgufModelConfig cfg;

    private final GgufLayerWeights layerWeights;
    private final GgufTensorResolver tensorResolver;
    private final GGUFTensorInfo tokenEmbed;
    private final float[] outputNorm;
    private final GGUFTensorInfo outputWeight;

    // Per-layer token embeddings (Gemma 4)
    private final GGUFTensorInfo perLayerTokEmbd;
    private final GGUFTensorInfo perLayerModelProj;
    private final float[] perLayerProjNorm;

    // Coordinators
    private final GgufMetalDecodeState metalDecodeState = new GgufMetalDecodeState();
    private final GgufDecodePass decodePass;
    private final GgufPrefillPass prefillPass;

    /**
     * A key projection means this layer owns its KV entries. Gemma 4 can omit
     * {@code attn_v.weight}; in that case the projected key is reused as the
     * value input, but the layer still owns its KV cache. Do not misclassify
     * such layers as cross-layer KV-sharing layers during prefill.
     */

    /**
     * Explains why the native full-GPU decode path cannot be used before any
     * native buffers are registered. Kept pure so the eligibility policy can
     * be regression-tested without a GPU or a model file.
     */
    /**
     * Returns whether a model's KV projection layout is not safe for the
     * current native full-GPU decoder. Independent K/V projections, K-as-V
     * layers, and shared-KV owners with matching geometry are supported. A
     * V-only layer is not representable by the native decode path.
     *
     * <p>
     * Array entries are per-layer; a shared-KV layer with both projections
     * absent uses the preceding one layer for global attention or two layers
     * back for SWA, matching the current registration convention.
     * </p>
     */

    /**
     * Classifies a failed native decode call. Falling back is safe only before
     * the native path has successfully advanced any generated-token KV entries,
     * because the host KV cache is not populated by successful native steps.
     */

    /**
     * Prompt KV synchronization happens before generation and before the native
     * decoder has committed any token. If any layer fails to sync, the populated
     * host cache remains the safe source of truth for per-op decoding.
     */

    GgufDecodeStep(GGUFModel model, GgufModelConfig cfg) {
        this.model = model;
        this.execution = new GgufExecutionDispatcher(model);
        this.cfg = cfg;
        this.tensorResolver = new GgufTensorResolver(model, cfg.architecture());

        int dim = cfg.embeddingDim();
        int ffnDim = cfg.ffnDim();
        int nH = cfg.nHeads();
        int hDim = cfg.headDim();
        int layers = cfg.nLayers();
        int nEmbdPerLayer = cfg.nEmbdPerLayer();

        // Compute true max KV heads across all layers (ignore nKVHeads which may be
        // wrong)
        int maxKVH = 1;
        if (cfg.nKVHeadsPerLayer() != null) {
            for (int kvh : cfg.nKVHeadsPerLayer())
                maxKVH = Math.max(maxKVH, kvh);
        } else {
            maxKVH = cfg.nKVHeads();
        }
        this.tokenEmbed = tensorResolver.requireTensor("token_embd.weight");
        this.outputNorm = tensorResolver.dequantizeVector(tensorResolver.requireTensor("output_norm.weight"), dim);
        this.outputWeight = tensorResolver.findTensor("output.weight").orElse(tokenEmbed); // tied embeddings

        // Per-layer token embedding tensors
        this.perLayerTokEmbd = tensorResolver.findTensor("per_layer_token_embd.weight").orElse(null);
        this.perLayerModelProj = tensorResolver.findTensor("per_layer_model_proj.weight").orElse(null);
        this.perLayerProjNorm = tensorResolver.findTensor("per_layer_proj_norm.weight")
                .map(t -> tensorResolver.dequantizeVector(t, nEmbdPerLayer > 0 ? nEmbdPerLayer : 1))
                .orElse(null);

        this.layerWeights = GgufLayerWeightResolver.resolve(
                tensorResolver, cfg, dim, nEmbdPerLayer, perLayerTokEmbd);

        // Opt-in layer-shape diagnostics verify architecture-specific RoPE dimensions
        // on the actual loaded GGUF, especially for Gemma 4's local/global attention
        // mix.
        if (System.getenv("ALJABR_DEBUG_LAYERS") != null) {
            int previousHeadDim = -1;
            int previousRotaryDim = -1;
            boolean previousSwa = false;
            for (int layer = 0; layer < layers; layer++) {
                int layerHeadDim = cfg.headDimForLayer(layer);
                int rotaryDim = cfg.rotaryDimForLayer(layer);
                boolean swa = cfg.isSwaLayer(layer);
                if (isVerbose() && (layer == 0 || layerHeadDim != previousHeadDim
                        || rotaryDim != previousRotaryDim || swa != previousSwa)) {
                    System.out.printf(
                            "[JAVA ROPE CONFIG] layer=%d attention=%s head_dim=%d rotary_dim=%d rope_base=%.1f freq_factors=%d%n",
                            layer, swa ? "swa" : "global", layerHeadDim, rotaryDim,
                            cfg.ropeFreqBaseForLayer(layer),
                            layerWeights.ropeFreqs[layer] == null ? 0 : layerWeights.ropeFreqs[layer].length);
                }
                previousHeadDim = layerHeadDim;
                previousRotaryDim = rotaryDim;
                previousSwa = swa;
            }
        }

        // Register static layer norms on GPU bridge once to enable zero-copy decode
        // slices
        if (GgufMetalBridge.isAvailable()) {
            GgufMetalBridge.registerLayerNorms(layers, dim, hDim, layerWeights.postAttnNorm, layerWeights.ffnNorm,
                    layerWeights.postFfnNorm, layerWeights.attnNorm,
                    layerWeights.attnQNorm, layerWeights.attnKNorm);
        }

        if (GgufMetalBridge.isAvailable()) {
            GgufMetalBridge.ensureModelRegistered(model);
        }

        GgufMetalDecodeRegistration.Result metalRegistration = GgufMetalDecodeRegistration.register(model, cfg,
                layerWeights, tokenEmbed, perLayerTokEmbd,
                outputNorm, outputWeight, dim, ffnDim, nH, layers, nEmbdPerLayer);
        boolean canUseMetalDecodeStep = metalRegistration.decodeActive();
        boolean canUseMetalTokenEmbd = metalRegistration.tokenEmbeddingRegistered();

        this.decodePass = new GgufDecodePass(
                model, cfg, layerWeights, execution, tokenEmbed,
                perLayerTokEmbd, perLayerModelProj, perLayerProjNorm,
                outputWeight, outputNorm, metalDecodeState,
                canUseMetalDecodeStep, canUseMetalTokenEmbd, maxKVH,
                this::disableMetalDecodeAfterRuntimeFallback);
        this.prefillPass = new GgufPrefillPass(
                model, tokenEmbed, perLayerTokEmbd, perLayerModelProj, perLayerProjNorm,
                decodePass.scratchPeGate(), decodePass.scratchPeOut(), outputWeight, outputNorm,
                cfg, layerWeights, execution,
                metalDecodeState, canUseMetalDecodeStep, maxKVH,
                this::disableMetalDecodeAfterRuntimeFallback);

        if (isVerbose()) {
            if (metalRegistration.decodeActive()) {
                System.out.println(
                        "[GgufDecodeStep] Full-GPU single command buffer decode step ENABLED (" + layers + " layers).");
            } else {
                String reason = metalRegistration.skipReason() == null
                        ? "unknown eligibility or registration failure"
                        : metalRegistration.skipReason();
                System.out.println("[GgufDecodeStep] Full-GPU single command buffer decode step UNAVAILABLE: "
                        + reason + "; using the existing per-op decode path.");
            }
        }
    }

    /**
     * Batch-prefill all prompt tokens through the network in one shot.
     *
     * <p>
     * For Gemma 4 SWA layers this uses <em>bidirectional</em> attention
     * within the sliding window (matching llama.cpp's
     * {@code LLAMA_NON_CAUSAL_TYPE_SWA_ONLY}). Global (dense) layers remain
     * strictly causal. All KV pairs are deposited into {@code cache} so that
     * subsequent single-token decode calls can build on the prefill.
     *
     * @param tokenIds prompt token IDs, length T
     * @param cache    KV cache to populate (must be empty / positioned at 0)
     * @return logits array for the last token
     */
    float[] prefill(int[] tokenIds, KVCache cache) {
        return prefillPass.execute(tokenIds, cache);
    }

    void syncMetalCache(int tokenCount, KVCache cache) {
        prefillPass.syncMetalCache(tokenCount, cache);
    }

    void disableMetalDecodeAfterRuntimeFallback(int pos, String reason) {
        metalDecodeState.disableBeforeCommit();
        System.err.printf(
                "[GgufDecodeStep] Full-GPU decode disabled after position %d: %s; "
                        + "falling back to per-op decode for this and subsequent tokens.%n",
                pos, reason);
    }

    public boolean canUseMetalGreedy() {
        return decodePass.canUseMetalGreedy();
    }

    int forwardGreedy(int tokenId, int pos, KVCache cache) {
        return decodePass.forwardGreedy(tokenId, pos, cache);
    }

    float[] forward(int tokenId, int pos, KVCache cache, boolean computeLogits) {
        return decodePass.forward(tokenId, pos, cache, computeLogits);
    }

    float[] forward(int tokenId, int pos, KVCache cache) {
        return decodePass.forward(tokenId, pos, cache);
    }
}
