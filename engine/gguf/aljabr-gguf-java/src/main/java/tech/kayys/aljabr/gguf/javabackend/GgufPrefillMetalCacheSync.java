package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;

/** Synchronizes the host-populated prompt KV cache into native Metal decode buffers. */
final class GgufPrefillMetalCacheSync {
    private GgufPrefillMetalCacheSync() {}

    /**
     * Copies each layer's prompt KV data into its native buffer.
     *
     * @return the first layer that failed synchronization, or {@code -1} on success
     */
    static int sync(int tokenCount, GGUFTensorInfo[] keyWeights, GgufModelConfig config,
            KVCache cache, int layers) {
        for (int layer = 0; layer < layers; layer++) {
            int sourceLayer = sourceLayer(layer, keyWeights[layer] != null, config.isSwaLayer(layer));
            int stride = cache.headStride(sourceLayer);
            int count = tokenCount * stride;
            if (!GgufMetalBridge.syncKVCache(layer, tokenCount,
                    cache.kLayer(sourceLayer), cache.vLayer(sourceLayer), count)) {
                return layer;
            }
        }
        return -1;
    }

    static int sourceLayer(int layer, boolean hasOwnKeyProjection, boolean isSwaLayer) {
        return hasOwnKeyProjection ? layer : Math.max(0, layer - (isSwaLayer ? 2 : 1));
    }
}
