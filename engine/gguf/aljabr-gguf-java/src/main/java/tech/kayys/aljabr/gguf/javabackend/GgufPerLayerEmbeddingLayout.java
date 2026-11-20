package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

/**
 * Validates the optional per-layer token-embedding (PLE) tensor group before
 * the decoder registers or executes the associated projection.
 */
final class GgufPerLayerEmbeddingLayout {
    private GgufPerLayerEmbeddingLayout() {
    }

    static void validate(
            GGUFTensorInfo tokenEmbedding,
            GGUFTensorInfo modelProjection,
            GGUFTensorInfo projectionNorm,
            int embeddingDimension,
            int vocabularySize,
            int layerCount,
            int perLayerDimension) {
        boolean hasTokenEmbedding = tokenEmbedding != null;
        boolean hasModelProjection = modelProjection != null;
        if (hasTokenEmbedding != hasModelProjection) {
            throw new IllegalArgumentException(
                    "GGUF per-layer token embedding and model projection tensors must be present together");
        }
        if (!hasTokenEmbedding) {
            if (projectionNorm != null) {
                throw new IllegalArgumentException(
                        "GGUF per-layer projection norm requires per-layer token embedding and model projection tensors");
            }
            return;
        }
        if (embeddingDimension <= 0 || vocabularySize <= 0 || layerCount <= 0 || perLayerDimension <= 0) {
            throw new IllegalArgumentException(
                    "GGUF per-layer embedding dimensions, vocabulary size, and layer count must be positive");
        }

        int perTokenDimension;
        try {
            perTokenDimension = Math.multiplyExact(layerCount, perLayerDimension);
        } catch (ArithmeticException exception) {
            throw new IllegalArgumentException("GGUF per-layer embedding width exceeds Java array limits", exception);
        }

        validateMatrix(tokenEmbedding, "per_layer_token_embd.weight",
                perTokenDimension, vocabularySize);
        validateMatrix(modelProjection, "per_layer_model_proj.weight",
                embeddingDimension, perTokenDimension);
        if (projectionNorm != null) {
            validateVector(projectionNorm, "per_layer_proj_norm.weight", perLayerDimension);
        }
    }

    private static void validateMatrix(
            GGUFTensorInfo tensor, String tensorName, int expectedColumns, int expectedRows) {
        long[] shape = tensor.shape();
        if (shape == null || shape.length != 2) {
            throw new IllegalArgumentException("GGUF tensor '" + tensorName + "' must be rank 2");
        }
        long columns = GgufTensorOps.matrixColumns(tensor);
        long rows = GgufTensorOps.matrixRows(tensor);
        if (columns != expectedColumns || rows != expectedRows) {
            throw new IllegalArgumentException("GGUF tensor '" + tensorName + "' has matrix shape ["
                    + rows + ", " + columns + "]; expected [" + expectedRows + ", " + expectedColumns + "]");
        }
    }

    private static void validateVector(GGUFTensorInfo tensor, String tensorName, int expectedDimension) {
        long[] shape = tensor.shape();
        if (shape == null || shape.length != 1) {
            throw new IllegalArgumentException("GGUF tensor '" + tensorName + "' must be a rank-1 vector");
        }
        if (shape[0] != expectedDimension) {
            throw new IllegalArgumentException("GGUF tensor '" + tensorName + "' must contain "
                    + expectedDimension + " values, got " + shape[0]);
        }
    }
}
