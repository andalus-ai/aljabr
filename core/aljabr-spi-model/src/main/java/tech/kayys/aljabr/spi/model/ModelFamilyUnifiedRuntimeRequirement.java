package tech.kayys.aljabr.spi.model;

import java.util.List;
import java.util.Map;

/**
 * Declares a native / non-standard runtime requirement for a specific model
 * type
 * within a model family (e.g. a detached MoE expert process).
 *
 * @param modelType   the model-type identifier this requirement applies to
 * @param modalities  required modality types (e.g. IMAGE, AUDIO)
 * @param detached    whether the runtime must run in a separate process
 * @param description human-readable reason for the requirement
 * @param metadata    additional key/value pairs for tooling
 * @author bhangun
 */
public record ModelFamilyUnifiedRuntimeRequirement(
        String modelType,
        List<String> requiredInputModalities,
        boolean productionReadyRequired,
        String reason,
        Map<String, String> metadata) {

    public ModelFamilyUnifiedRuntimeRequirement {
        modelType = modelType == null ? "" : modelType;
        requiredInputModalities = requiredInputModalities == null ? List.of() : List.copyOf(requiredInputModalities);
        reason = reason == null ? "" : reason;
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
}
