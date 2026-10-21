package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q2KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q3KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q4KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q5KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q6KMatrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q8Matrix;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.Q32Matrix;

/**
 * Prepared matrix builders for Java-native GGUF mat-vec kernels.
 *
 * <p>These builders unpack compact GGUF rows into cache-friendly byte/scale
 * arrays while preserving the public matrix record types exposed by
 * {@link GgufTensorOps}.</p>
 * @author bhangun
 */
final class GgufBuild {
    private GgufBuild() {
    }

    static Q32Matrix q32Matrix(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufQ32Build.matrix(model, tensor);
    }

    static Q2KMatrix q2KMatrix(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufQ2KBuild.matrix(model, tensor);
    }

    static Q3KMatrix q3KMatrix(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufKPlainBuild.q3(model, tensor);
    }

    static Q4KMatrix q4KMatrix(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufQ45KBuild.q4(model, tensor);
    }

    static Q5KMatrix q5KMatrix(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufQ45KBuild.q5(model, tensor);
    }

    static Q6KMatrix q6KMatrix(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufKPlainBuild.q6(model, tensor);
    }

    static Q8Matrix q8Matrix(GGUFModel model, GGUFTensorInfo tensor) {
        return GgufQ8Build.matrix(model, tensor);
    }
}
