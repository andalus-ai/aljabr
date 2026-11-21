package tech.kayys.aljabr.integration.kv;

import java.util.concurrent.atomic.AtomicBoolean;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Owns immutable K/V tensor snapshots copied from a committed cache range.
 *
 * <p>The tensors returned by {@link #keys()} and {@link #values()} are borrowed
 * views owned by this snapshot. Callers must not release them independently;
 * closing this snapshot releases both tensors.</p>
 */
public final class KVCacheTensorSnapshot implements AutoCloseable {
    private final Tensor keys;
    private final Tensor values;
    private final int layer;
    private final int firstPosition;
    private final int tokenCount;
    private final AtomicBoolean closed = new AtomicBoolean();

    KVCacheTensorSnapshot(
            Tensor keys, Tensor values, int layer, int firstPosition, int tokenCount) {
        this.keys = keys;
        this.values = values;
        this.layer = layer;
        this.firstPosition = firstPosition;
        this.tokenCount = tokenCount;
    }

    /** Returns the borrowed key tensor with shape {@code [tokens, kvHeads, headDimension]}. */
    public Tensor keys() {
        ensureOpen();
        return keys;
    }

    /** Returns the borrowed value tensor with shape {@code [tokens, kvHeads, headDimension]}. */
    public Tensor values() {
        ensureOpen();
        return values;
    }

    public int layer() {
        return layer;
    }

    public int firstPosition() {
        return firstPosition;
    }

    public int tokenCount() {
        return tokenCount;
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            RuntimeException failure = null;
            try {
                keys.release();
            } catch (RuntimeException releaseFailure) {
                failure = releaseFailure;
            }
            try {
                values.release();
            } catch (RuntimeException releaseFailure) {
                if (failure == null) {
                    failure = releaseFailure;
                } else {
                    failure.addSuppressed(releaseFailure);
                }
            }
            if (failure != null) {
                throw failure;
            }
        }
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("KV cache tensor snapshot is closed");
        }
    }
}
