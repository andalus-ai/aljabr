package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;

/**
 * Executes post-attention projections, residuals, feed-forward network,
 * optional per-layer embeddings, and layer output scaling for prefill batches.
 */
final class GgufPrefillResidualExecution {
    record Result(
            boolean fusedLayerDone,
            long elapsedNanos,
            boolean fusedSuccess,
            long tWoNanos,
            long tFfnNanos) {}

    private GgufPrefillResidualExecution() {}

    static Result execute(
            GGUFModel model,
            int layer,
            float[] X,
            float[] allAttn,
            float[] allWoOut,
            float[] allFfnIn,
            float[] allFfnOut,
            float[] xbL,
            float[] gateL,
            float[] upL,
            float[] hbL,
            float[] peGate,
            float[] peOut,
            float[][] perLayerInputsT,
            boolean hasPerLayer,
            int tokenCount,
            int dim,
            int qDim,
            int nEmbdPL,
            float eps,
            GgufLayerWeights layerWeights,
            GgufModelConfig cfg,
            GgufExecutionDispatcher execution,
            GgufMetalDecodeState metalDecodeState,
            boolean canUseMetalDecodeStep) {

        long t0 = System.nanoTime();
        boolean fusedLayerDone = false;
        long tWo = 0;
        long tFfn = 0;
        boolean fusedSuccess = false;

        if (metalDecodeState.canUse(canUseMetalDecodeStep) && !hasPerLayer) {
            fusedLayerDone = GgufMetalBridge.woFfnFusedLayerBatch(
                    model, layer, X, allAttn, dim, qDim, tokenCount, eps, cfg.isGeGLU());
        }

        if (!fusedLayerDone) {
            long tWo0 = System.nanoTime();
            execution.dispatchMatVecBatch(layerWeights.wO[layer], allAttn, allWoOut, tokenCount);

            for (int t = 0; t < tokenCount; t++) {
                System.arraycopy(allWoOut, t * dim, xbL, 0, dim);
                if (layerWeights.postAttnNorm[layer] != null) {
                    GGUFVectorOps.rmsNorm(xbL, layerWeights.postAttnNorm[layer], xbL, dim, eps);
                }
                GGUFVectorOps.addInPlace(X, t * dim, xbL, 0, dim);
            }
            tWo = System.nanoTime() - tWo0;

            // FFN execution
            GgufPrefillFfnExecution.Result ffnResult = GgufPrefillFfnExecution.execute(
                    X, allFfnIn, allFfnOut, xbL, gateL, upL, hbL, tokenCount, layer, layerWeights, cfg, execution);
            fusedSuccess = ffnResult.fused();
            tFfn = ffnResult.elapsedNanos();

            // Per-layer embedding + Layer output scale
            for (int t = 0; t < tokenCount; t++) {
                if (hasPerLayer && layerWeights.perLayerInpGate[layer] != null
                        && layerWeights.perLayerProj[layer] != null) {
                    float[] xSlice = new float[dim];
                    System.arraycopy(X, t * dim, xSlice, 0, dim);
                    execution.dispatchMatVec(layerWeights.perLayerInpGate[layer], xSlice, peGate);
                    float kGelu = 0.7978845608f;
                    int peSliceOffset = layer * nEmbdPL;
                    for (int j = 0; j < nEmbdPL; j++) {
                        float g = peGate[j];
                        float gelu = 0.5f * g * (1.0f + (float) Math.tanh(kGelu * (g + 0.044715f * g * g * g)));
                        peGate[j] = gelu * perLayerInputsT[t][peSliceOffset + j];
                    }
                    execution.dispatchMatVec(layerWeights.perLayerProj[layer], peGate, peOut);
                    if (layerWeights.perLayerPostNorm[layer] != null) {
                        GGUFVectorOps.rmsNorm(peOut, layerWeights.perLayerPostNorm[layer], peOut, dim, eps);
                    }
                    GGUFVectorOps.addInPlace(X, t * dim, peOut, 0, dim);
                }

                if (layerWeights.outScale[layer] != 1.0f) {
                    GGUFVectorOps.scale(X, t * dim, layerWeights.outScale[layer], dim);
                }
            }
        }

        long elapsedNanos = System.nanoTime() - t0;
        return new Result(fusedLayerDone, elapsedNanos, fusedSuccess, tWo, tFfn);
    }
}
