package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

/**
 * Validates the vocabulary-facing tensors used by token lookup and logits projection.
 */
final class GgufOutputTensorLayout {
    private GgufOutputTensorLayout() {
    }

    static void validate(
            GGUFTensorInfo tokenEmbedding,
            GGUFTensorInfo outputNorm,
            GGUFTensorInfo outputWeight,
            int embeddingDimension,
            int vocabularySize) {
        if (embeddingDimension <= 0 || vocabularySize <= 0) {
            throw new IllegalArgumentException("embedding dimension and vocabulary size must be positive");
        }
        validateMatrix(tokenEmbedding, "token_embd.weight", embeddingDimension, vocabularySize);
        validateVector(outputNorm, "output_norm.weight", embeddingDimension);
        if (outputWeight != null) {
            validateMatrix(outputWeight, "output.weight", embeddingDimension, vocabularySize);
        }
    }

    private static void validateMatrix(
            GGUFTensorInfo tensor, String tensorName, int embeddingDimension, int vocabularySize) {
        if (tensor == null) {
            throw new IllegalArgumentException("GGUF required tensor '" + tensorName + "' is missing");
        }
        long[] shape = tensor.shape();
        if (shape == null || shape.length != 2) {
            throw new IllegalArgumentException("GGUF tensor '" + tensorName + "' must be rank 2");
        }
        long inputDimension = GgufTensorOps.matrixColumns(tensor);
        long outputDimension = GgufTensorOps.matrixRows(tensor);
        if (inputDimension != embeddingDimension || outputDimension != vocabularySize) {
            throw new IllegalArgumentException("GGUF tensor '" + tensorName + "' has matrix shape ["
                    + outputDimension + ", " + inputDimension + "]; expected ["
                    + vocabularySize + ", " + embeddingDimension + "]");
        }
    }

    private static void validateVector(GGUFTensorInfo tensor, String tensorName, int expectedDimension) {
        if (tensor == null) {
            throw new IllegalArgumentException("GGUF required tensor '" + tensorName + "' is missing");
        }
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
