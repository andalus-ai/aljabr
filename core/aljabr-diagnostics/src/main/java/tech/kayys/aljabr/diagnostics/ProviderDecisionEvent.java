package tech.kayys.aljabr.diagnostics;

import java.time.Instant;
import java.util.Objects;

/**
 * Metadata-only explanation of a compute-provider selection decision.
 *
 * <p>This is an operational observation, not a diagnostic finding, audit
 * record or metric. It intentionally excludes model paths, request payloads,
 * exception messages and arbitrary provider metadata.</p>
 */
public record ProviderDecisionEvent(
        String providerId,
        Decision decision,
        String reason,
        int priority,
        SelectionMode selectionMode,
        Instant occurredAt) {

    public ProviderDecisionEvent {
        providerId = bounded(providerId, "providerId", 128);
        decision = Objects.requireNonNull(decision, "decision");
        reason = bounded(reason, "reason", 512);
        selectionMode = Objects.requireNonNull(selectionMode, "selectionMode");
        occurredAt = Objects.requireNonNull(occurredAt, "occurredAt");
    }

    private static String bounded(String value, String name, int max) {
        Objects.requireNonNull(value, name);
        String normalized = value.trim();
        if (normalized.isEmpty() || normalized.length() > max) {
            throw new IllegalArgumentException(name + " must contain 1.." + max + " characters");
        }
        return normalized;
    }

    public enum Decision {
        SELECTED,
        REJECTED,
        UNAVAILABLE,
        NOT_READY,
        PROBE_FAILED
    }

    public enum SelectionMode {
        AUTO,
        EXPLICIT
    }
}
