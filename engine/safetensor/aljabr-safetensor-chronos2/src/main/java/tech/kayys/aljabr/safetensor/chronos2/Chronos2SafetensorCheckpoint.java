package tech.kayys.aljabr.safetensor.chronos2;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import tech.kayys.aljabr.models.chronos2.Chronos2Config;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;

/**
 * Open Chronos-2 checkpoint whose tensor views are backed by the shared
 * Safetensors loader's mapped/off-heap storage.
 *
 * <p>The checkpoint owns the shard session. Do not retain tensor views beyond
 * this object's lifetime; closing it invalidates every view.</p>
 */
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
            for (String name : binder.expectedTensors(config).keySet()) {
                tensors.put(name, session.tensor(name));
            }
            this.binding = binder.bind(config, tensors, session.tensorCount());
        } catch (RuntimeException | Error failure) {
            session.close();
            throw failure;
        }
    }

    /** Original zero-copy view for a logical parameter path. */
    public SafetensorTensor tensor(String parameterName) {
        ensureOpen();
        return binding.tensorForParameter(parameterName);
    }

    /** Validated checkpoint name to logical parameter name bindings. */
    public Map<String, String> parameterBindings() {
        ensureOpen();
        return binding.mapping().parameterBindings();
    }

    /** Number of tensors required by this model family. */
    public int expectedTensorCount() {
        ensureOpen();
        return binding.mapping().expectedTensorCount();
    }

    /** Total tensor count in the source checkpoint, including unrelated tensors. */
    public int checkpointTensorCount() {
        ensureOpen();
        return binding.mapping().checkpointTensorCount();
    }

    public boolean isClosed() {
        return closed.get();
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            session.close();
        }
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("Chronos-2 Safetensors checkpoint is closed");
        }
    }
}
