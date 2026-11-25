package tech.kayys.aljabr.safetensor.chronos2;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.models.chronos2.Chronos2Config;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;
import tech.kayys.aljabr.safetensor.tensor.SafetensorLinearBinding;
import tech.kayys.aljabr.safetensor.tensor.SafetensorTensorBinding;

/** Open Chronos-2 checkpoint backed by the shared mapped Safetensors loader. */
public final class Chronos2SafetensorCheckpoint implements AutoCloseable {
    private final SafetensorShardLoader.SafetensorShardSession session;
    private final Chronos2SafetensorBindings.Binding binding;
    private final AtomicBoolean closed = new AtomicBoolean();

    Chronos2SafetensorCheckpoint(SafetensorShardLoader.SafetensorShardSession session,
            Chronos2Config config, Chronos2SafetensorBindings binder) {
        this.session = Objects.requireNonNull(session, "session");
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(binder, "binder");
        try {
            Map<String, SafetensorTensor> tensors = new LinkedHashMap<>();
            for (String name : binder.expectedTensors(config).keySet()) tensors.put(name, session.tensor(name));
            this.binding = binder.bind(config, tensors, session.tensorCount());
        } catch (RuntimeException | Error failure) {
            session.close();
            throw failure;
        }
    }
    public SafetensorTensor tensor(String parameterName) {
        ensureOpen();
        return binding.tensorForParameter(parameterName);
    }
    /** Returned tensor retains its own lease and must be closed by its caller. */
    public Tensor openTensor(String parameterName, ComputeBackend backend) {
        ensureOpen();
        return SafetensorTensorBinding.openTensor(session, binding.checkpointNameForParameter(parameterName), backend);
    }
    /**
     * Opens a mapped rank-two checkpoint parameter as the reusable Aljabr Linear module.
     * The module owns its tensor and lease, may outlive this checkpoint, and must be closed.
     * Bias tensors remain independently addressable through {@link #openTensor}.
     */
    public Linear openLinear(String parameterName, ComputeBackend backend) {
        ensureOpen();
        return SafetensorLinearBinding.openLinear(
                session, binding.checkpointNameForParameter(parameterName), backend);
    }
    public Map<String, String> parameterBindings() {
        ensureOpen();
        return binding.mapping().parameterBindings();
    }
    public int expectedTensorCount() { ensureOpen(); return binding.mapping().expectedTensorCount(); }
    public int checkpointTensorCount() { ensureOpen(); return binding.mapping().checkpointTensorCount(); }
    public boolean isClosed() { return closed.get(); }
    @Override public void close() {
        if (closed.compareAndSet(false, true)) session.close();
    }
    private void ensureOpen() {
        if (closed.get()) throw new IllegalStateException("Chronos-2 Safetensors checkpoint is closed");
    }
}
