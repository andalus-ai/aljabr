package tech.kayys.aljabr.safetensor.tensor;

import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;

/**
 * Creates ordinary Aljabr tensors directly over mapped Safetensors weights.
 *
 * <p>This integration-layer factory keeps the tensor core unaware of the
 * Safetensors format and the format loader unaware of compute backends. Binding
 * is zero-copy; the selected backend governs subsequent tensor operations.</p>
 */
public final class SafetensorTensorBinding {

    private SafetensorTensorBinding() {
    }

    /**
     * Opens a named checkpoint tensor as an eager Aljabr tensor without copying
     * its payload. Ownership of the storage reference is transferred to the
     * returned tensor, which releases the session lease when closed.
     *
     * @param session open Safetensors shard session
     * @param tensorName source tensor name
     * @param backend selected backend responsible for subsequent operations
     * @return an ordinary Aljabr tensor backed by mapped checkpoint bytes
     */
    public static Tensor openTensor(
            SafetensorShardSession session,
            String tensorName,
            ComputeBackend backend) {
        Objects.requireNonNull(backend, "backend must not be null");
        SafetensorTensorStorage storage = SafetensorTensorStorage.open(session, tensorName);
        try {
            return new DefaultTensor(
                    storage.shape(),
                    storage.dtype(),
                    backend.deviceType(),
                    storage,
                    backend,
                    storage.layout());
        } catch (RuntimeException | Error failure) {
            storage.release();
            throw failure;
        }
    }
}
