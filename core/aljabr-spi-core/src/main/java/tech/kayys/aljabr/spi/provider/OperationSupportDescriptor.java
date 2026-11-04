package tech.kayys.aljabr.spi.provider;

import java.util.Objects;

/** Immutable support result for one operation under caller-supplied constraints. */
public record OperationSupportDescriptor(
        String operationId,
        OperationSupport support,
        String effectiveBackendId) {

    public OperationSupportDescriptor {
        if (operationId == null || operationId.isBlank())
            throw new IllegalArgumentException("operationId must not be blank");
        operationId = operationId.trim();
        Objects.requireNonNull(support, "support");
        effectiveBackendId = effectiveBackendId == null || effectiveBackendId.isBlank()
                ? null : effectiveBackendId.trim();
        if (support == OperationSupport.DELEGATED && effectiveBackendId == null)
            throw new IllegalArgumentException("DELEGATED support requires effectiveBackendId");
    }
}