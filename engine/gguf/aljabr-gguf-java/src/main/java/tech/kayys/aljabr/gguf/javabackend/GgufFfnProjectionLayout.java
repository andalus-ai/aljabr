package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

/**
 * Validates dense FFN gate, up, and down projection matrices against the
 * model embedding and intermediate dimensions.
 */
final class GgufFfnProjectionLayout {
    private GgufFfnProjectionLayout() {
    }

    static void validate(
            GGUFTensorInfo[] gateWeights,
            GGUFTensorInfo[] upWeights,
            GGUFTensorInfo[] downWeights,
            int embeddingDimension,
            int intermediateDimension) {
        if (upWeights == null || downWeights == null) {
            throw new IllegalArgumentException("FFN up/down projection arrays must not be null");
        }
        if (gateWeights != null && gateWeights.length != upWeights.length
                || downWeights.length != upWeights.length) {
            throw new IllegalArgumentException("FFN projection arrays must have equal layer counts");
        }
        if (embeddingDimension <= 0 || intermediateDimension <= 0) {
            throw new IllegalArgumentException("FFN embedding and intermediate dimensions must be positive");
        }

        for (int layer = 0; layer < upWeights.length; layer++) {
            validateProjection(upWeights[layer], "ffn_up", layer,
                    embeddingDimension, intermediateDimension);
            validateProjection(downWeights[layer], "ffn_down", layer,
                    intermediateDimension, embeddingDimension);
            if (gateWeights != null && gateWeights[layer] != null) {
                validateProjection(gateWeights[layer], "ffn_gate", layer,
                        embeddingDimension, intermediateDimension);
            }
        }
    }

    private static void validateProjection(
            GGUFTensorInfo tensor,
            String projection,
            int layer,
            int expectedInputDimension,
            int expectedOutputDimension) {
        if (tensor == null) {
            throw new IllegalArgumentException(
                    "Layer " + layer + " is missing required " + projection + " projection");
        }

        long actualInputDimension = GgufTensorOps.matrixColumns(tensor);
        long actualOutputDimension = GgufTensorOps.matrixRows(tensor);
        if (actualInputDimension != expectedInputDimension
                || actualOutputDimension != expectedOutputDimension) {
            throw new IllegalArgumentException("Layer " + layer
                    + " FFN " + projection + " tensor '" + tensor.name()
                    + "' has matrix shape [" + actualOutputDimension + ", "
                    + actualInputDimension + "]; expected [" + expectedOutputDimension
                    + ", " + expectedInputDimension + "]");
        }
    }
}
