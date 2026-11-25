package tech.kayys.aljabr.safetensor.whisper;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;

/**
 * Lifetime-safe view of a Whisper Safetensors checkpoint.
 *
 * <p>Tensor views are zero-copy and owned by the underlying shard session.
 * Callers must not retain a view after this checkpoint is closed.</p>
 *
 * <p>This is a storage adapter only: it does not imply that the Whisper
 * encoder/decoder graph or autoregressive inference is implemented.</p>
 */
public final class WhisperSafetensorCheckpoint implements AutoCloseable {
    private final SafetensorShardLoader.SafetensorShardSession session;
    private final AtomicBoolean closed = new AtomicBoolean();

    WhisperSafetensorCheckpoint(SafetensorShardLoader.SafetensorShardSession session) {
        this.session = Objects.requireNonNull(session, "session");
    }

    /** Returns the zero-copy tensor view under its Hugging Face checkpoint key. */
    public SafetensorTensor tensor(String checkpointName) {
        ensureOpen();
        if (checkpointName == null || checkpointName.isBlank()) {
            throw new IllegalArgumentException("checkpointName must not be blank");
        }
        return session.tensor(checkpointName);
    }

    /** Number of tensors declared by the checkpoint index/header. */
    public int tensorCount() {
        ensureOpen();
        return session.tensorCount();
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
            throw new IllegalStateException("Whisper Safetensors checkpoint is closed");
        }
    }
}
