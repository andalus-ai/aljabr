package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

/** Per-layer model tensors and normalization metadata resolved from a GGUF model. */
final class GgufLayerWeights {
    final GGUFTensorInfo[] perLayerInpGate;
    final GGUFTensorInfo[] perLayerProj;
    final float[][] perLayerPostNorm;
    final float[][] attnNorm;
    final float[][] attnQNorm;
    final float[][] attnKNorm;
    final float[][] attnVNorm;
    final float[][] postAttnNorm;
    final float[][] ropeFreqs;
    final float[] outScale;
    final GGUFTensorInfo[] wQ, wK, wV, wO;
    final float[][] ffnNorm;
    final float[][] postFfnNorm;
    final GGUFTensorInfo[] ffnGate, ffnUp, ffnDown;

    GgufLayerWeights(int layers) {
        perLayerInpGate = new GGUFTensorInfo[layers];
        perLayerProj = new GGUFTensorInfo[layers];
        perLayerPostNorm = new float[layers][];
        attnNorm = new float[layers][];
        attnQNorm = new float[layers][];
        attnKNorm = new float[layers][];
        attnVNorm = new float[layers][];
        postAttnNorm = new float[layers][];
        ropeFreqs = new float[layers][];
        outScale = new float[layers];
        wQ = new GGUFTensorInfo[layers];
        wK = new GGUFTensorInfo[layers];
        wV = new GGUFTensorInfo[layers];
        wO = new GGUFTensorInfo[layers];
        ffnNorm = new float[layers][];
        postFfnNorm = new float[layers][];
        ffnGate = new GGUFTensorInfo[layers];
        ffnUp = new GGUFTensorInfo[layers];
        ffnDown = new GGUFTensorInfo[layers];
    }
}
