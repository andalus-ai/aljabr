package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

/** Prepares the flat, row-major token embedding buffer consumed by GGUF prefill. */
final class GgufPrefillEmbeddingPreparation {
    private GgufPrefillEmbeddingPreparation() {
    }

    /**
     * Dequantizes prompt token rows and applies the model embedding scale.
     *
     * @return a contiguous {@code tokenIds.length * dimension} hidden-state buffer
     */
    static float[] prepareTokenEmbeddings(
            GGUFModel model,
            GGUFTensorInfo tokenEmbedding,
            int[] tokenIds,
            int dimension,
            float embeddingScale) {
        float[] embeddings = new float[tokenIds.length * dimension];
        for (int token = 0; token < tokenIds.length; token++) {
            int offset = token * dimension;
            GgufTensorOps.dequantizeRow(model, tokenEmbedding, tokenIds[token], embeddings, offset);
            if (embeddingScale > 0.0f && embeddingScale != 1.0f) {
                GGUFVectorOps.scale(embeddings, offset, embeddingScale, dimension);
            }
        }
        return embeddings;
    }
}
