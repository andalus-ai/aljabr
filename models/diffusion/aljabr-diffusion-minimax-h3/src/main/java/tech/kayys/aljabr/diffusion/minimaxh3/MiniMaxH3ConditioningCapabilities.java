package tech.kayys.aljabr.diffusion.minimaxh3;

import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningRequirement;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Declares which generic conditioning input modalities are associated with
 * each MiniMax-H3 task, including minimum modality cardinalities.
 *
 * <p>This is descriptive capability metadata only. It does not decode inputs,
 * build embeddings, preprocess images, or execute a generation pipeline.</p>
 */
public final class MiniMaxH3ConditioningCapabilities {

    private MiniMaxH3ConditioningCapabilities() {
    }

    /**
     * Returns the request-input requirements for a task. The prompt is
     * required for all declared tasks; image/reference/frame requirements
     * follow the task's conditioning mode.
     */
    public static List<DiffusionConditioningRequirement> requirements(MiniMaxH3Task task) {
        Objects.requireNonNull(task, "task");
        List<DiffusionConditioningRequirement> requirements = new ArrayList<>();
        requirements.add(DiffusionConditioningRequirement.required(
                DiffusionConditioningInput.Modality.PROMPT));

        switch (task) {
            case TEXT_TO_VIDEO, TEXT_TO_IMAGE -> {
                // Prompt-only conditioning.
            }
            case IMAGE_TO_VIDEO, IMAGE_EDIT ->
                    requirements.add(DiffusionConditioningRequirement.required(
                            DiffusionConditioningInput.Modality.IMAGE));
            case REFERENCE_TO_VIDEO ->
                    requirements.add(DiffusionConditioningRequirement.required(
                            DiffusionConditioningInput.Modality.REFERENCE_IMAGE));
            case MULTIFRAME_REFERENCE_TO_VIDEO -> {
                requirements.add(DiffusionConditioningRequirement.required(
                        DiffusionConditioningInput.Modality.VIDEO_FRAME));
                requirements.add(DiffusionConditioningRequirement.optional(
                        DiffusionConditioningInput.Modality.REFERENCE_IMAGE));
            }
        }
        return List.copyOf(requirements);
    }

    public static boolean supports(
            MiniMaxH3Task task,
            DiffusionConditioningInput.Modality modality) {
        Objects.requireNonNull(modality, "modality");
        return requirements(task).stream()
                .anyMatch(requirement -> requirement.modality() == modality);
    }

    /**
     * Validates required, optional and unsupported input modalities for a
     * task. This does not impose prompt cardinality beyond one-or-more, frame
     * sampling, preprocessing, or tensor-layout policy.
     */
    public static void validate(
            MiniMaxH3Task task,
            DiffusionConditioningSet conditioning) {
        Objects.requireNonNull(task, "task");
        Objects.requireNonNull(conditioning, "conditioning");
        conditioning.validateAgainst(requirements(task));
    }
}
