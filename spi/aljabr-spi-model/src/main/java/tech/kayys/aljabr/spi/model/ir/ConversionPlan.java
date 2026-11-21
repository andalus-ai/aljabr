package tech.kayys.aljabr.spi.model.ir;

import java.util.Objects;

/**
 * Immutable result of format compatibility preflight.
 *
 * <p>A plan is descriptive only; it does not execute conversion or imply that
 * a writer has been invoked.
 */
public record ConversionPlan(
        ModelIdentity sourceIdentity,
        String sourceFormat,
        String targetFormat,
        CompatibilityReport compatibility) {

    public ConversionPlan {
        Objects.requireNonNull(sourceIdentity, "sourceIdentity must not be null");
        sourceFormat = requireText(sourceFormat, "sourceFormat");
        targetFormat = requireText(targetFormat, "targetFormat");
        Objects.requireNonNull(compatibility, "compatibility must not be null");
    }

    /** True when the target can accept the model, possibly with explicit losses. */
    public boolean isConvertible() {
        return compatibility.fidelity() != CompatibilityReport.ConversionFidelity.UNSUPPORTED;
    }

    /** True when callers must explicitly acknowledge a partial-fidelity plan. */
    public boolean requiresAcknowledgement() {
        return compatibility.fidelity() == CompatibilityReport.ConversionFidelity.PARTIAL;
    }

    /** True only when the target reports full semantic and metadata fidelity. */
    public boolean isLossless() {
        return compatibility.fidelity() == CompatibilityReport.ConversionFidelity.FULL;
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value.trim();
    }
}
