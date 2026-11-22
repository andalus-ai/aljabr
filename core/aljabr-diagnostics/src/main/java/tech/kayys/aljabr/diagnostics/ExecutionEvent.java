package tech.kayys.aljabr.diagnostics;

import java.time.Instant;
import java.util.Objects;

/**
 * Immutable lifecycle observation for one explicitly scoped computation.
 *
 * <p>Lifecycle events are observations, not diagnostics, audit records or
 * performance metrics. The event contains metadata only; failure messages
 * and stack traces are deliberately excluded.</p>
 *
 * @param type lifecycle transition
 * @param context explicit execution and correlation metadata
 * @param occurredAt wall-clock time at which the transition was observed
 * @param elapsedNanos monotonic elapsed time from scope start; zero for STARTED
 * @param failureType exception class name for FAILED, otherwise {@code null}
 */
public record ExecutionEvent(
        Type type,
        DiagnosticContext context,
        Instant occurredAt,
        long elapsedNanos,
        String failureType) {

    public ExecutionEvent {
        type = Objects.requireNonNull(type, "type");
        context = Objects.requireNonNull(context, "context");
        occurredAt = Objects.requireNonNull(occurredAt, "occurredAt");
        if (elapsedNanos < 0) {
            throw new IllegalArgumentException("elapsedNanos must not be negative");
        }
        if (type == Type.STARTED && elapsedNanos != 0) {
            throw new IllegalArgumentException("STARTED elapsedNanos must be zero");
        }
        if (type == Type.FAILED) {
            if (failureType == null || failureType.isBlank()) {
                throw new IllegalArgumentException("FAILED events require a failureType");
            }
            failureType = failureType.trim();
        } else if (failureType != null) {
            throw new IllegalArgumentException("failureType is only valid for FAILED events");
        }
    }

    /** Supported terminal and non-terminal execution transitions. */
    public enum Type {
        STARTED,
        SUCCEEDED,
        FAILED,
        ABANDONED
    }
}
