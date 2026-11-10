package tech.kayys.aljabr.gguf.javabackend;

/**
 * Pure policy decisions for the optional native full-GPU GGUF decoder.
 * Kept separate from model execution so architecture and fallback rules can
 * be tested without loading a model or initializing a native backend.
 */
final class GgufDecodeEligibility {
    private GgufDecodeEligibility() { }

    static boolean hasOwnKvCache(boolean hasKeyProjection) {
        return hasKeyProjection;
    }

    static boolean hasUnsupportedNativeKvLayout(
            boolean[] hasKeyProjection,
            boolean[] hasValueProjection,
            boolean[] swaLayers,
            int[] headDims,
            int[] kvHeads) {
        if (hasKeyProjection == null || hasValueProjection == null || swaLayers == null
                || headDims == null || kvHeads == null) {
            throw new IllegalArgumentException("native KV layout arrays must not be null");
        }
        int layers = hasKeyProjection.length;
        if (hasValueProjection.length != layers || swaLayers.length != layers
                || headDims.length != layers || kvHeads.length != layers) {
            throw new IllegalArgumentException("native KV layout arrays must have equal lengths");
        }

        for (int layer = 0; layer < layers; layer++) {
            boolean hasKey = hasKeyProjection[layer];
            boolean hasValue = hasValueProjection[layer];

            // A V-only layer is unsupported. K-as-V is supported: the native
            // path copies the raw K projection into V before K normalization
            // and RoPE, and this layer owns its own KV cache.
            if (!hasKey && hasValue) {
                return true;
            }

            // If both projections are absent, this is a shared-KV layer.
            // The native decoder cannot safely use an owner with a different
            // head dimension or number of KV heads.
            if (!hasKey) {
                int ownerLayer = Math.max(0, layer - (swaLayers[layer] ? 2 : 1));
                if (headDims[layer] != headDims[ownerLayer]
                        || kvHeads[layer] != kvHeads[ownerLayer]) {
                    return true;
                }
            }
        }
        return false;
    }

    static String fullGpuDecodeEligibilityFailure(
            boolean decodeDisabled,
            boolean nativeDecodeAvailable,
            boolean usesPerLayerTokenEmbeddings) {
        return fullGpuDecodeEligibilityFailure(
                decodeDisabled, nativeDecodeAvailable, usesPerLayerTokenEmbeddings, false, false);
    }

    static String fullGpuDecodeRuntimeFailureReason(
            boolean nativeStepSucceeded, boolean nativeStepPreviouslyCommitted) {
        if (nativeStepSucceeded) {
            return null;
        }
        if (nativeStepPreviouslyCommitted) {
            return "native Metal decode-step failed after prior native tokens; the host KV cache is incomplete";
        }
        return "native Metal decode-step failed before any native token was committed; per-op fallback is safe";
    }

    static String fullGpuDecodeKvCacheSyncFailureReason(int failedLayer) {
        if (failedLayer < 0) {
            return null;
        }
        return "Metal KV-cache sync failed at layer " + failedLayer
                + " before generation; host KV cache is intact and per-op fallback is safe";
    }

    static String fullGpuDecodeEligibilityFailure(
            boolean decodeDisabled,
            boolean nativeDecodeAvailable,
            boolean usesPerLayerTokenEmbeddings,
            boolean usesPartialRotaryDimensions) {
        return fullGpuDecodeEligibilityFailure(
                decodeDisabled, nativeDecodeAvailable, usesPerLayerTokenEmbeddings,
                usesPartialRotaryDimensions, false);
    }

    static String fullGpuDecodeEligibilityFailure(
            boolean decodeDisabled,
            boolean nativeDecodeAvailable,
            boolean usesPerLayerTokenEmbeddings,
            boolean usesPartialRotaryDimensions,
            boolean usesUnsupportedNativeKvLayout) {
        if (decodeDisabled) {
            return "disabled by -Daljabr.metal.decode.disable=true";
        }
        if (!nativeDecodeAvailable) {
            return "native Metal decode-step entry point is unavailable";
        }
        if (usesPerLayerTokenEmbeddings) {
            return "model uses per-layer token embeddings, which the native full-GPU decode step does not implement";
        }
        if (usesUnsupportedNativeKvLayout) {
            return "model uses V-only projections or incompatible shared-KV geometry, which the native full-GPU decode step does not implement safely";
        }
        if (usesPartialRotaryDimensions) {
            return "model uses per-layer partial RoPE, which the native full-GPU decode step does not implement";
        }
        return null;
    }
}
