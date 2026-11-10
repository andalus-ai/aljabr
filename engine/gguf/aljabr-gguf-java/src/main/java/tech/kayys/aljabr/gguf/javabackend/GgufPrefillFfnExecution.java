package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;

/**
 * Executes batched Feed-Forward Network (FFN) computation and residual addition
 * for prefill prompt tokens.
 */
final class GgufPrefillFfnExecution {
    private GgufPrefillFfnExecution() {
    }

    record Result(boolean fused, long elapsedNanos) {
    }

    /**
     * Executes the FFN block for all prompt tokens in a layer, adding the post-FFN
     * residual into {@code hiddenStates}.
     *
     * @return a {@link Result} indicating whether the fused Metal kernel ran and the elapsed nanoseconds
     */
    static Result execute(
            float[] hiddenStates,
            float[] ffnInput,
            float[] ffnOutput,
            float[] tokenBuffer,
            float[] gateBuffer,
            float[] upBuffer,
            float[] hiddenBuffer,
            int tokenCount,
            int layer,
            GgufLayerWeights layerWeights,
            GgufModelConfig config,
            GgufExecutionDispatcher execution) {
        int dimension = config.embeddingDim();
        int ffnDimension = config.ffnDim();
        float rmsNormEpsilon = config.rmsNormEps();
        boolean isGeGLU = config.isGeGLU();

        long startNanos = System.nanoTime();

        // 1. RMSNorm ffn_norm for all tokens into ffnInput
        for (int token = 0; token < tokenCount; token++) {
            GGUFVectorOps.rmsNorm(hiddenStates, token * dimension, layerWeights.ffnNorm[layer],
                    ffnInput, token * dimension, dimension, rmsNormEpsilon);
        }

        // 2. Fused batch dispatch on Metal if gate projection is present
        boolean fusedSuccess = false;
        if (layerWeights.ffnGate[layer] != null) {
            fusedSuccess = execution.dispatchFfnFusedBatch(
                    layerWeights.ffnGate[layer], layerWeights.ffnUp[layer], layerWeights.ffnDown[layer],
                    ffnInput, ffnOutput, isGeGLU, tokenCount);
        }

        // 3. Post-FFN norm and residual addition
        if (fusedSuccess) {
            for (int token = 0; token < tokenCount; token++) {
                System.arraycopy(ffnOutput, token * dimension, tokenBuffer, 0, dimension);
                if (layerWeights.postFfnNorm[layer] != null) {
                    GGUFVectorOps.rmsNorm(tokenBuffer, layerWeights.postFfnNorm[layer],
                            tokenBuffer, dimension, rmsNormEpsilon);
                }
                GGUFVectorOps.addInPlace(hiddenStates, token * dimension, tokenBuffer, 0, dimension);
            }
        } else {
            for (int token = 0; token < tokenCount; token++) {
                System.arraycopy(ffnInput, token * dimension, tokenBuffer, 0, dimension);
                if (layerWeights.ffnGate[layer] != null) {
                    execution.dispatchMatVecPair(
                            layerWeights.ffnGate[layer], gateBuffer,
                            layerWeights.ffnUp[layer], upBuffer, tokenBuffer);
                    if (isGeGLU) {
                        GGUFVectorOps.geGLU(gateBuffer, upBuffer, hiddenBuffer, ffnDimension);
                    } else {
                        GGUFVectorOps.swiGLU(gateBuffer, upBuffer, hiddenBuffer, ffnDimension);
                    }
                } else {
                    execution.dispatchMatVec(layerWeights.ffnUp[layer], tokenBuffer, upBuffer);
                    GGUFVectorOps.siluInPlace(upBuffer, ffnDimension);
                    System.arraycopy(upBuffer, 0, hiddenBuffer, 0, ffnDimension);
                }
                execution.dispatchMatVec(layerWeights.ffnDown[layer], hiddenBuffer, tokenBuffer);
                if (layerWeights.postFfnNorm[layer] != null) {
                    GGUFVectorOps.rmsNorm(tokenBuffer, layerWeights.postFfnNorm[layer],
                            tokenBuffer, dimension, rmsNormEpsilon);
                }
                GGUFVectorOps.addInPlace(hiddenStates, token * dimension, tokenBuffer, 0, dimension);
            }
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        return new Result(fusedSuccess, elapsedNanos);
    }
}
