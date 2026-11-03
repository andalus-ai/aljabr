package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

/**
 * Executes GGUF matrix operations through the available Metal provider,
 * preserving the existing Java CPU fallback and operation accounting.
 *
 * <p>Decode orchestration and model-specific math remain in
 * {@link GgufDecodeStep}; this class owns only backend dispatch mechanics.</p>
 */
final class GgufExecutionDispatcher {
    private final GGUFModel model;

    GgufExecutionDispatcher(GGUFModel model) {
        this.model = model;
    }

    private static long countMetal = 0;
    private static long countCpu = 0;

    private void dispatchMatVec(GGUFTensorInfo tensor, float[] input, float[] output) {
        if (GgufMetalBridge.matVec(model, tensor, input, output)) {
            countMetal++;
        } else {
            countCpu++;
            if (countCpu <= 15) {
                System.err.println("[FALLBACK] tensor " + tensor.name() + " (typeId=" + tensor.typeId() + ", shape="
                        + java.util.Arrays.toString(tensor.shape()) + ") fell back to CPU!");
            }
            GgufTensorOps.matVecParallel(model, tensor, input, output);
        }
    }

    private void dispatchMatVecPair(GGUFTensorInfo t0, float[] out0, GGUFTensorInfo t1, float[] out1, float[] input) {
        if (GgufMetalBridge.matVecPair(model, t0, out0, t1, out1, input)) {
            countMetal += 2;
        } else {
            dispatchMatVec(t0, input, out0);
            dispatchMatVec(t1, input, out1);
        }
    }

    private void dispatchMatVecTriple(GGUFTensorInfo t0, float[] out0, GGUFTensorInfo t1, float[] out1,
            GGUFTensorInfo t2, float[] out2, float[] input) {
        if (GgufMetalBridge.matVecTriple(model, t0, out0, t1, out1, t2, out2, input)) {
            countMetal += 3;
        } else {
            dispatchMatVec(t0, input, out0);
            dispatchMatVec(t1, input, out1);
            dispatchMatVec(t2, input, out2);
        }
    }

    private void dispatchMatVecBatch(GGUFTensorInfo tensor, float[] inputBatch, float[] outputBatch, int batchSize) {
        if (GgufMetalBridge.matVecBatch(model, tensor, inputBatch, outputBatch, batchSize)) {
            countMetal += batchSize;
        } else {
            int K = (int) GgufTensorOps.matrixColumns(tensor);
            int N = (int) GgufTensorOps.matrixRows(tensor);
            float[] inSlice = new float[K];
            float[] outSlice = new float[N];
            for (int t = 0; t < batchSize; t++) {
                System.arraycopy(inputBatch, t * K, inSlice, 0, K);
                dispatchMatVec(tensor, inSlice, outSlice);
                System.arraycopy(outSlice, 0, outputBatch, t * N, N);
            }
        }
    }

    private void dispatchMatVecTripleBatch(GGUFTensorInfo t0, float[] out0,
            GGUFTensorInfo t1, float[] out1,
            GGUFTensorInfo t2, float[] out2,
            float[] inputBatch, int batchSize) {
        if (GgufMetalBridge.matVecTripleBatch(model, t0, out0, t1, out1, t2, out2, inputBatch, batchSize)) {
            countMetal += 3L * batchSize;
        } else {
            int K = (int) GgufTensorOps.matrixColumns(t0);
            int N0 = (int) GgufTensorOps.matrixRows(t0);
            int N1 = (int) GgufTensorOps.matrixRows(t1);
            int N2 = (int) GgufTensorOps.matrixRows(t2);
            float[] inSlice = new float[K];
            float[] s0 = new float[N0];
            float[] s1 = new float[N1];
            float[] s2 = new float[N2];
            for (int t = 0; t < batchSize; t++) {
                System.arraycopy(inputBatch, t * K, inSlice, 0, K);
                dispatchMatVecTriple(t0, s0, t1, s1, t2, s2, inSlice);
                System.arraycopy(s0, 0, out0, t * N0, N0);
                System.arraycopy(s1, 0, out1, t * N1, N1);
                System.arraycopy(s2, 0, out2, t * N2, N2);
            }
        }
    }

    private boolean dispatchFfnFused(GGUFTensorInfo gate, GGUFTensorInfo up, GGUFTensorInfo down,
            float[] inA, float[] outDown, boolean isGeGLU) {
        if (GgufMetalBridge.ffnGatedFused(model, gate, up, down, inA, outDown, isGeGLU)) {
            countMetal += 3;
            return true;
        }
        return false;
    }

    private boolean dispatchFfnFusedBatch(GGUFTensorInfo gate, GGUFTensorInfo up, GGUFTensorInfo down,
            float[] inABatch, float[] outDownBatch, boolean isGeGLU, int batchSize) {
        if (GgufMetalBridge.ffnGatedFusedBatch(model, gate, up, down, inABatch, outDownBatch, isGeGLU, batchSize)) {
            countMetal += 3L * batchSize;
            return true;
        }
        return false;
    }
}
