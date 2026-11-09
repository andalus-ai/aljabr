package tech.kayys.aljabr.backend.metal;

import java.util.Objects;

/** Describes the execution route selected for one Metal backend operation. */
public record MetalExecutionEvent(String operation, ExecutionPath executionPath, String reason) {
    public MetalExecutionEvent {
        if (operation == null || operation.isBlank()) {
            throw new IllegalArgumentException("operation must not be blank");
        }
        Objects.requireNonNull(executionPath, "executionPath");
        if (reason == null || reason.isBlank()) {
            throw new IllegalArgumentException("reason must not be blank");
        }
    }

    public enum ExecutionPath {
        METAL_NATIVE,
        CPU_FALLBACK,
        REJECTED
    }
}
