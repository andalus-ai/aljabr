package tech.kayys.aljabr.gguf.javabackend;

import java.util.Optional;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;

/**
 * Resolves GGUF tensor metadata and materializes small vector weights.
 *
 * <p>This class deliberately does not select execution backends or perform
 * matrix operations. Those remain responsibilities of the decode step.</p>
 */
final class GgufTensorResolver {
    private final GGUFModel model;
    private final String architecture;

    GgufTensorResolver(GGUFModel model, String architecture) {
        this.model = model;
        this.architecture = architecture;
    }

    GGUFTensorInfo requireTensor(String name) {
        return findTensor(name)
                .orElseThrow(() -> new IllegalStateException(
                        "GGUF model is missing required tensor '" + name + "' for architecture '"
                                + architecture + "'"));
    }

    Optional<GGUFTensorInfo> findTensor(String name) {
        try {
            return Optional.ofNullable(GgufTensorOps.findTensor(model, name));
        } catch (IllegalArgumentException e) {
            return Optional.empty();
        }
    }

    float[] dequantizeVector(GGUFTensorInfo tensor, int fallbackDim) {
        int dim = (tensor.shape() != null && tensor.shape().length > 0 && tensor.shape()[0] > 0)
                ? (int) tensor.shape()[0]
                : fallbackDim;
        float[] out = new float[dim];
        GgufTensorOps.dequantizeRow(model, tensor, 0L, out);
        return out;
    }
}
