package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;

/**
 * Computes the final normalized output projection and optional logit softcap.
 *
 * <p>Backend selection remains delegated to the existing Metal bridge and
 * execution dispatcher; this class only owns the final-logits stage.</p>
 */
final class GgufLogitsProcessor {
    private GgufLogitsProcessor() {
    }

    static Timing compute(
            GGUFModel model,
            GGUFTensorInfo outputWeight,
            float[] outputNorm,
            float[] hiddenState,
            float[] normalizedState,
            float[] logits,
            int dimension,
            float epsilon,
            float softcap,
            GgufExecutionDispatcher execution) {
        long start = System.nanoTime();
        GGUFVectorOps.rmsNorm(hiddenState, outputNorm, normalizedState, dimension, epsilon);
        long normalizationEnd = System.nanoTime();

        if (softcap > 0.0f && GgufMetalBridge.matVecSoftcap(
                model, outputWeight, normalizedState, logits, softcap)) {
            long projectionEnd = System.nanoTime();
            return new Timing(normalizationEnd - start, projectionEnd - normalizationEnd, 0L);
        }

        execution.dispatchMatVec(outputWeight, normalizedState, logits);
        long projectionEnd = System.nanoTime();
        if (softcap > 0.0f) {
            // Prefer the GPU kernel; retain the numerically equivalent Java fallback.
            boolean gpuDone = GgufMetalBridge.softcapLogits(logits, softcap);
            if (!gpuDone) {
                float inverseSoftcap = 1.0f / softcap;
                for (int i = 0; i < logits.length; i++) {
                    logits[i] = softcap * (float) Math.tanh(logits[i] * inverseSoftcap);
                }
            }
        }
        long softcapEnd = System.nanoTime();
        return new Timing(
                normalizationEnd - start,
                projectionEnd - normalizationEnd,
                softcapEnd - projectionEnd);
    }

    record Timing(long normalizationNanos, long projectionNanos, long softcapNanos) {
        long totalNanos() {
            return normalizationNanos + projectionNanos + softcapNanos;
        }
    }
}
