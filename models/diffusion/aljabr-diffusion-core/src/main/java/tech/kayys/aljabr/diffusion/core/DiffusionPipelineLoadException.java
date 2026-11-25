package tech.kayys.aljabr.diffusion.core;

import java.util.List;

/** Raised when a manifest cannot be materialized into a loaded component pipeline. */
public final class DiffusionPipelineLoadException extends RuntimeException {
    private final List<String> missingRequiredRoles;

    public DiffusionPipelineLoadException(String message, List<String> missingRequiredRoles) {
        super(message);
        this.missingRequiredRoles = List.copyOf(missingRequiredRoles);
    }

    public DiffusionPipelineLoadException(String message, Throwable cause) {
        super(message, cause);
        this.missingRequiredRoles = List.of();
    }

    public List<String> missingRequiredRoles() { return missingRequiredRoles; }
}
