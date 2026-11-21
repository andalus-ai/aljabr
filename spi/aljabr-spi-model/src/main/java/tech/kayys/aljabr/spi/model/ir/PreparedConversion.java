package tech.kayys.aljabr.spi.model.ir;

import java.util.Objects;

/**
 * Immutable, preflighted conversion input. Preparing a conversion reads model
 * structure and assesses compatibility, but does not open tensor payloads or
 * create the target file.
 */
public record PreparedConversion(ModelDescriptor model, ConversionPlan plan) {

    public PreparedConversion {
        Objects.requireNonNull(model, "model must not be null");
        Objects.requireNonNull(plan, "plan must not be null");
        if (!model.identity().equals(plan.sourceIdentity())) {
            throw new IllegalArgumentException("plan identity does not match model descriptor");
        }
        if (!model.sourceFormat().equalsIgnoreCase(plan.sourceFormat())) {
            throw new IllegalArgumentException("plan source format does not match model descriptor");
        }
    }
}
