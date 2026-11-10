package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

/**
 * Initializes and registers the native full-GPU GGUF decode path.
 *
 * <p>This owns native eligibility and buffer registration only; decode
 * orchestration and numerical execution remain in {@link GgufDecodeStep}.</p>
 */
final class GgufMetalDecodeRegistration {
    record Result(boolean decodeActive, boolean tokenEmbeddingRegistered, String skipReason) {}

    private GgufMetalDecodeRegistration() {}

    static Result register(
            GGUFModel model, GgufModelConfig cfg, GgufLayerWeights layerWeights,
            GGUFTensorInfo tokenEmbed, GGUFTensorInfo perLayerTokEmbd, float[] outputNorm, GGUFTensorInfo outputWeight,
            int dim, int ffnDim, int nH, int layers, int nEmbdPerLayer) {
                boolean decodeDisabled = Boolean.getBoolean("aljabr.metal.decode.disable");
                boolean nativeDecodeAvailable = !decodeDisabled && GgufMetalBridge.isDecodeStepAvailable();
                boolean usesPerLayerTokenEmbeddings = perLayerTokEmbd != null && nEmbdPerLayer > 0;
                boolean usesPartialRotaryDimensions = false;
                boolean[] hasKeyProjection = new boolean[layers];
                boolean[] hasValueProjection = new boolean[layers];
                boolean[] swaLayers = new boolean[layers];
                int[] layerHeadDims = new int[layers];
                int[] layerKvHeads = new int[layers];
                for (int layer = 0; layer < layers; layer++) {
                    if (cfg.rotaryDimForLayer(layer) < cfg.headDimForLayer(layer)) {
                        usesPartialRotaryDimensions = true;
                    }
                    hasKeyProjection[layer] = layerWeights.wK[layer] != null;
                    hasValueProjection[layer] = layerWeights.wV[layer] != null;
                    swaLayers[layer] = cfg.isSwaLayer(layer);
                    layerHeadDims[layer] = cfg.headDimForLayer(layer);
                    layerKvHeads[layer] = cfg.nKVHeadsForLayer(layer);
                }
                boolean usesUnsupportedNativeKvLayout = GgufDecodeEligibility.hasUnsupportedNativeKvLayout(
                        hasKeyProjection, hasValueProjection, swaLayers, layerHeadDims, layerKvHeads);
                String metalDecodeSkipReason = GgufDecodeEligibility.fullGpuDecodeEligibilityFailure(
                        decodeDisabled,
                        nativeDecodeAvailable,
                        usesPerLayerTokenEmbeddings,
                        usesPartialRotaryDimensions,
                        usesUnsupportedNativeKvLayout);
        
                for (int l = 0; l < layers && metalDecodeSkipReason == null; l++) {
                    if (layerWeights.wQ[l] == null || layerWeights.wO[l] == null || layerWeights.ffnUp[l] == null || layerWeights.ffnDown[l] == null) {
                        metalDecodeSkipReason = "layer " + l + " is missing a required Q/O/FFN weight";
                    }
                }
        
                boolean metalDecodeActive = false;
                if (metalDecodeSkipReason == null) {
                    int[] strides = new int[layers];
                    for (int i = 0; i < layers; i++) {
                        int layerNKVH = cfg.nKVHeadsForLayer(i);
                        int hD = cfg.headDimForLayer(i);
                        strides[i] = layerNKVH * hD;
                    }
                    if (GgufMetalBridge.initKVCache(layers, cfg.contextLength(), strides)) {
                        boolean allLayersRegistered = true;
                        int failedRegistrationLayer = -1;
                        for (int l = 0; l < layers && allLayersRegistered; l++) {
                            int layerNKVH = cfg.nKVHeadsForLayer(l);
                            int layerHDim = cfg.headDimForLayer(l);
                            float layerRopeFreq = cfg.ropeFreqBaseForLayer(l);
                            boolean isSwa = cfg.isSwaLayer(l);
                            int n0 = nH * layerHDim;
                            int n1 = layerNKVH * layerHDim;
                            int n2 = layerNKVH * layerHDim;
                            int wO_K = n0;
                            int wO_N = dim;
                            int ffnIntermediate = ffnDim;
                            int stride = strides[l];
        
                            float attnScale = cfg.attentionScale();
                            if (attnScale <= 0f) {
                                attnScale = (layerWeights.attnQNorm[l] != null || layerWeights.attnKNorm[l] != null) ? 1.0f
                                        : (1.0f / (float) Math.sqrt(layerHDim));
                            }
        
                            // A key projection owns the KV cache, including K-as-V layers.
                            int kvLayerIndex = layerWeights.wK[l] != null
                                    ? l
                                    : Math.max(0, l - (isSwa ? 2 : 1));
                            boolean reg = GgufMetalBridge.registerDecodeLayer(
                                    l, kvLayerIndex, model,
                                    layerWeights.wQ[l], layerWeights.wK[l], layerWeights.wV[l], layerWeights.wO[l],
                                    layerWeights.ffnGate[l], layerWeights.ffnUp[l], layerWeights.ffnDown[l],
                                    n0, n1, n2, wO_K, wO_N, ffnIntermediate,
                                    nH, layerNKVH, layerHDim, stride,
                                    layerRopeFreq, isSwa ? 1 : 0,
                                    layerWeights.outScale[l], attnScale,
                                    layerWeights.postAttnNorm[l] != null ? 1 : 0,
                                    layerWeights.postFfnNorm[l] != null ? 1 : 0,
                                    layerWeights.attnQNorm[l] != null ? 1 : 0,
                                    layerWeights.attnKNorm[l] != null ? 1 : 0,
                                    (cfg.hasUnitVNorm() || layerWeights.attnVNorm[l] != null) ? 1 : 0);
                            boolean ropeFreqsRegistered = layerWeights.ropeFreqs[l] == null;
                            if (reg && layerWeights.ropeFreqs[l] != null) {
                                ropeFreqsRegistered = GgufMetalBridge.registerRopeFreqs(l, layerWeights.ropeFreqs[l]);
                            }
                            if (!decodeLayerRegistrationComplete(reg, layerWeights.ropeFreqs[l], ropeFreqsRegistered)) {
                                if (reg && layerWeights.ropeFreqs[l] != null && !ropeFreqsRegistered) {
                                    System.err.println("[GgufDecodeStep] Failed to register proportional RoPE frequencies for layer "
                                            + l + "; disabling Metal decode to preserve correctness.");
                                }
                                allLayersRegistered = false;
                            }
                        }
                        if (allLayersRegistered) {
                            metalDecodeActive = GgufMetalBridge.registerOutput(
                                    model, outputNorm, outputWeight, dim, cfg.vocabSize(), cfg.finalLogitSoftcap());
                            if (!metalDecodeActive) {
                                metalDecodeSkipReason = "native output/logits registration failed";
                            }
                        } else {
                            metalDecodeSkipReason = "native decode-layer registration failed at layer "
                                    + failedRegistrationLayer;
                        }
                    } else {
                        metalDecodeSkipReason = "native KV-cache initialization failed";
                    }
                }
                boolean tokenEmbdRegistered = false;
                if (metalDecodeActive && tokenEmbed != null && !usesPerLayerTokenEmbeddings) {
                    tokenEmbdRegistered = GgufMetalBridge.registerTokenEmbd(
                            model, tokenEmbed, dim, cfg.embeddingScale());
                    if (tokenEmbdRegistered && GgufDecodeStep.isVerbose()) {
                        System.out.println("[GgufDecodeStep] GPU-resident token embedding registered.");
                    }
                }
        return new Result(metalDecodeActive, tokenEmbdRegistered, metalDecodeSkipReason);
    }

    static boolean decodeLayerRegistrationComplete(
            boolean layerRegistered, float[] layerRopeFreqs, boolean ropeFreqsRegistered) {
        return layerRegistered && (layerRopeFreqs == null || ropeFreqsRegistered);
    }
}
