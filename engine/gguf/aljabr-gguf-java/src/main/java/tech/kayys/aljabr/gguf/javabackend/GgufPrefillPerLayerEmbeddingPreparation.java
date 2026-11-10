package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

/** Prepares optional per-layer token embeddings used by architectures that expose them. */
final class GgufPrefillPerLayerEmbeddingPreparation {
    private GgufPrefillPerLayerEmbeddingPreparation() {
    }

    /**
     * Builds the per-token, per-layer embedding inputs consumed by the prefill layer loop.
     *
     * @return {@code null} when the model has no per-layer token embeddings; otherwise
     *         a {@code [token][layer * perLayerDimension]} array
     */
    static float[][] prepare(
            GGUFModel model,
            GGUFTensorInfo perLayerTokenEmbedding,
            GGUFTensorInfo perLayerModelProjection,
            float[] perLayerProjectionNorm,
            int[] tokenIds,
            float[] tokenEmbeddings,
            int dimension,
            int perLayerDimension,
            int layerCount,
            float rmsNormEpsilon,
            GgufExecutionDispatcher execution) {
        if (perLayerTokenEmbedding == null || perLayerDimension <= 0) {
            return null;
        }

        int perTokenSize = perLayerDimension * layerCount;
        float[][] inputs = new float[tokenIds.length][perTokenSize];
        float tokenEmbeddingScale = (float) Math.sqrt(perLayerDimension);
        float projectionScale = 1.0f / (float) Math.sqrt(dimension);
        float inputScale = 1.0f / (float) Math.sqrt(2.0);
        float[] rawToken = new float[perTokenSize];
        float[] projected = new float[perTokenSize];
        float[] embeddingSlice = new float[dimension];

        for (int token = 0; token < tokenIds.length; token++) {
            GgufTensorOps.dequantizeRow(model, perLayerTokenEmbedding, tokenIds[token], rawToken);
            GGUFVectorOps.scale(rawToken, tokenEmbeddingScale, perTokenSize);
            System.arraycopy(tokenEmbeddings, token * dimension, embeddingSlice, 0, dimension);
            execution.dispatchMatVec(perLayerModelProjection, embeddingSlice, projected);
            GGUFVectorOps.scale(projected, projectionScale, perTokenSize);

            if (perLayerProjectionNorm != null) {
                for (int layer = 0; layer < layerCount; layer++) {
                    int offset = layer * perLayerDimension;
                    GGUFVectorOps.rmsNorm(projected, offset, perLayerProjectionNorm,
                            projected, offset, perLayerDimension, rmsNormEpsilon);
                }
            }

            for (int index = 0; index < perTokenSize; index++) {
                inputs[token][index] = (projected[index] + rawToken[index]) * inputScale;
            }
        }
        return inputs;
    }
}
