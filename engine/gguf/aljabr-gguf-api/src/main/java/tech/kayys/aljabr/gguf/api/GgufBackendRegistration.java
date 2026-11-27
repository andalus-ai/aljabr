package tech.kayys.aljabr.gguf.api;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Owns a registered backend and defers provider cleanup until every operation
 * that acquired it has released its lease.
 */
final class GgufBackendRegistration {
    private final GgufBackend backend;
    private final Runnable closeAction;
    private int activeLeases;
    private boolean retired;
    private boolean closeStarted;

    GgufBackendRegistration(GgufBackend backend, Runnable closeAction) {
        this.backend = Objects.requireNonNull(backend, "backend");
        this.closeAction = Objects.requireNonNull(closeAction, "closeAction");
    }

    synchronized Lease acquire() {
        if (retired) {
            throw new IllegalStateException("GGUF backend registration is retiring");
        }
        activeLeases++;
        return new Lease(this, backend);
    }

    void retire() {
        boolean closeNow;
        synchronized (this) {
            retired = true;
            closeNow = beginCloseIfUnused();
        }
        if (closeNow) closeAction.run();
    }

    private void release() {
        boolean closeNow;
        synchronized (this) {
            if (activeLeases <= 0) throw new IllegalStateException("GGUF backend lease accounting underflow");
            activeLeases--;
            closeNow = beginCloseIfUnused();
        }
        if (closeNow) closeAction.run();
    }

    private boolean beginCloseIfUnused() {
        if (!retired || activeLeases != 0 || closeStarted) return false;
        closeStarted = true;
        return true;
    }

    static final class Lease implements AutoCloseable {
        private final GgufBackendRegistration owner;
        private final GgufBackend backend;
        private final AtomicBoolean released = new AtomicBoolean();

        private Lease(GgufBackendRegistration owner, GgufBackend backend) {
            this.owner = owner;
            this.backend = backend;
        }

        GgufBackend backend() { return backend; }

        @Override
        public void close() {
            if (released.compareAndSet(false, true)) owner.release();
        }
    }
}
