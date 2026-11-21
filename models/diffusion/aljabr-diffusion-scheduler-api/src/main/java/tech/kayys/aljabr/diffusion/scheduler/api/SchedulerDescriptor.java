package tech.kayys.aljabr.diffusion.scheduler.api;

import java.util.Objects;
import java.util.Set;

/** Immutable declaration of a scheduler's mathematical compatibility. */
public record SchedulerDescriptor(
        String id,
        ScheduleDomain domain,
        Set<PredictionType> supportedPredictionTypes,
        boolean supportsStochasticSampling) {

    public SchedulerDescriptor {
        Objects.requireNonNull(id, "id");
        if (id.isBlank()) {
            throw new IllegalArgumentException("id must not be blank");
        }
        Objects.requireNonNull(domain, "domain");
        supportedPredictionTypes = Set.copyOf(
                Objects.requireNonNull(supportedPredictionTypes, "supportedPredictionTypes"));
        if (supportedPredictionTypes.isEmpty()) {
            throw new IllegalArgumentException("at least one prediction type is required");
        }
    }

    public boolean supports(PredictionType predictionType) {
        return supportedPredictionTypes.contains(Objects.requireNonNull(predictionType, "predictionType"));
    }
}
