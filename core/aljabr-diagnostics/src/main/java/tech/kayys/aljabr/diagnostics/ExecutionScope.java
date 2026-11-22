package tech.kayys.aljabr.diagnostics;

import java.time.Instant;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Explicit, closeable lifecycle scope around one computation.
 *
 * <p>Scopes emit STARTED immediately, then exactly one terminal transition:
 * SUCCEEDED, FAILED, or ABANDONED (when closed without an explicit outcome).
 * Elapsed time uses {@link System#nanoTime()} and is therefore independent of
 * wall-clock adjustments. The caller supplies all context; no thread-local or
 * implicit asynchronous propagation is performed.</p>
 *
 * <p>Observer {@link RuntimeException}s are isolated and retained for
 * inspection through {@link #observerFailure()}. They do not alter the
 * computation's outcome. Fatal {@link Error}s are allowed to propagate.</p>
 */
public final class ExecutionScope implements AutoCloseable {

    private final DiagnosticContext context;
    private final ExecutionObserver observer;
    private final long startedNanos;
    private final AtomicBoolean completed = new AtomicBoolean();
    private final AtomicReference<RuntimeException> observerFailure = new AtomicReference<>();

    private ExecutionScope(DiagnosticContext context, ExecutionObserver observer) {
        this.context = requireExecutionContext(context);
        this.observer = Objects.requireNonNull(observer, "observer");
        this.startedNanos = observer == ExecutionObserver.NOOP ? 0L : System.nanoTime();
        if (observer != ExecutionObserver.NOOP) {
            emit(new ExecutionEvent(
                    ExecutionEvent.Type.STARTED, context, Instant.now(), 0, null));
        }
    }

    /**
     * Starts a scope and synchronously emits STARTED.
     *
     * @param context explicit context; correlation ID and operation are required
     * @param observer receiver, or {@link ExecutionObserver#NOOP}
     */
    public static ExecutionScope start(
            DiagnosticContext context,
            ExecutionObserver observer) {
        return new ExecutionScope(context, observer);
    }

    /** Starts a scope with instrumentation disabled. */
    public static ExecutionScope start(DiagnosticContext context) {
        return start(context, ExecutionObserver.NOOP);
    }

    /** Marks this execution successful and emits SUCCEEDED exactly once. */
    public void succeed() {
        finish(ExecutionEvent.Type.SUCCEEDED, null);
    }

    /**
     * Marks this execution failed and emits only the exception type.
     * Exception messages and stack traces are not exported.
     */
    public void fail(Throwable failure) {
        Objects.requireNonNull(failure, "failure");
        finish(ExecutionEvent.Type.FAILED, failure.getClass().getName());
    }

    /** Returns whether a terminal transition has already been emitted. */
    public boolean isCompleted() {
        return completed.get();
    }

    /**
     * Returns the first observer runtime failure, if any. This makes isolated
     * instrumentation failures visible without changing computation behavior.
     */
    public Optional<RuntimeException> observerFailure() {
        return Optional.ofNullable(observerFailure.get());
    }

    /**
     * Closes an unfinished scope as ABANDONED. Closing a completed scope is
     * harmless, which makes try-with-resources safe after succeed/fail.
     */
    @Override
    public void close() {
        if (completed.compareAndSet(false, true)) {
            emitTerminal(ExecutionEvent.Type.ABANDONED, null);
        }
    }

    private void finish(ExecutionEvent.Type type, String failureType) {
        if (!completed.compareAndSet(false, true)) {
            throw new IllegalStateException("execution scope is already completed");
        }
        emitTerminal(type, failureType);
    }

    private void emitTerminal(ExecutionEvent.Type type, String failureType) {
        if (observer == ExecutionObserver.NOOP) {
            return;
        }
        long elapsed = Math.max(0L, System.nanoTime() - startedNanos);
        emit(new ExecutionEvent(type, context, Instant.now(), elapsed, failureType));
    }

    private void emit(ExecutionEvent event) {
        try {
            observer.onEvent(event);
        } catch (RuntimeException failure) {
            observerFailure.compareAndSet(null, failure);
        }
    }

    private static DiagnosticContext requireExecutionContext(DiagnosticContext context) {
        Objects.requireNonNull(context, "context");
        if (context.correlationId() == null || context.operation() == null) {
            throw new IllegalArgumentException(
                    "execution context requires correlationId and operation");
        }
        return context;
    }
}
