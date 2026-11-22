package tech.kayys.aljabr.diagnostics;

/**
 * Synchronous receiver of execution lifecycle events.
 *
 * <p>Observers should be fast and non-blocking. Runtime exceptions thrown by
 * an observer are isolated by {@link ExecutionScope} and exposed through the
 * scope's {@code observerFailure()} accessor; they do not replace a
 * computation's result or failure. Fatal {@link Error}s are not intercepted.</p>
 */
@FunctionalInterface
public interface ExecutionObserver {

    /** Shared disabled observer; it performs no work per event. */
    ExecutionObserver NOOP = event -> {
        // Intentionally disabled.
    };

    /** Receives one lifecycle event. */
    void onEvent(ExecutionEvent event);

    /** Returns the shared disabled observer. */
    static ExecutionObserver noop() {
        return NOOP;
    }
}
