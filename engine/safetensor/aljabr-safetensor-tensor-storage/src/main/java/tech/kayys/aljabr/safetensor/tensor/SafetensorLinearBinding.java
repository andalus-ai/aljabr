package tech.kayys.aljabr.safetensor.tensor;

import java.util.Objects;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;

/**
 * Binds a mapped Safetensors matrix directly as an Aljabr {@link Linear} layer.
 * The returned module owns the mapped weight tensor and its retained shard lease.
 */
public final class SafetensorLinearBinding {
    private SafetensorLinearBinding() {
    }

    /**
     * Opens a bias-free linear layer over a [outFeatures, inFeatures] checkpoint matrix.
     * Closing the returned module releases the mapped weight and its shard lease.
     */
    public static Linear openLinear(
            SafetensorShardSession session, String weightName, ComputeBackend backend) {
        Objects.requireNonNull(session, "session must not be null");
        Objects.requireNonNull(weightName, "weightName must not be null");
        Objects.requireNonNull(backend, "backend must not be null");

        Tensor weight = SafetensorTensorBinding.openTensor(session, weightName, backend);
        try {
            if (weight.shape().rank() != 2) {
                throw new IllegalArgumentException(
                        "linear weight must be rank two [outFeatures, inFeatures], got " + weight.shape());
            }
            return new Linear(weight, null);
        } catch (RuntimeException | Error failure) {
            weight.close();
            throw failure;
        }
    }
}
