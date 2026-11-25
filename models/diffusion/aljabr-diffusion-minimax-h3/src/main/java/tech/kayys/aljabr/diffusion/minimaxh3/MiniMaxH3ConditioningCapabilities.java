package tech.kayys.aljabr.diffusion.minimaxh3;

import tech.kayys.aljabr.diffusion.api.DiffusionConditioningInput;
import tech.kayys.aljabr.diffusion.api.DiffusionConditioningSet;

import java.util.Objects;

/**
 * Declares which generic conditioning input modalities are associated with
 * each MiniMax-H3 task.
 *
 * <p>This is descriptive capability metadata only. It does not decode inputs,
 * build embeddings, preprocess images, or execute a generation pipeline.</p>
 */
public final class MiniMaxH3ConditioningCapabilities {

    private MiniMaxH3ConditioningCapabilities() {
    }

    public static boolean supports(
            MiniMaxH3Task task,
            DiffusionConditioningInput.Modality modality) {
        Objects.requireNonNull(task, "task");
        Objects.requireNonNull(modality, "modality");

        if (modality == DiffusionConditioningInput.Modality.PROMPT) {
            return true;
        }

        return switch (task) {
            case TEXT_TO_VIDEO, TEXT_TO_IMAGE -> false;
            case IMAGE_TO_VIDEO, IMAGE_EDIT ->
                    modality == DiffusionConditioningInput.Modality.IMAGE;
            case REFERENCE_TO_VIDEO ->
                    modality == DiffusionConditioningInput.Modality.REFERENCE_IMAGE;
            case MULTIFRAME_REFERENCE_TO_VIDEO ->
                    modality == DiffusionConditioningInput.Modality.VIDEO_FRAME
                            || modality == DiffusionConditioningInput.Modality.REFERENCE_IMAGE;
        };
    }

    /**
     * Rejects input modalities not declared for the task. This intentionally
     * does not impose prompt cardinality or infer model-specific preprocessing.
     */
    public static void validate(
            MiniMaxH3Task task,
            DiffusionConditioningSet conditioning) {
        Objects.requireNonNull(conditioning, "conditioning");
        for (DiffusionConditioningInput input : conditioning.inputs()) {
            if (!supports(task, input.modality())) {
                throw new IllegalArgumentException(
                        "Conditioning modality " + input.modality()
                                + " is not declared for MiniMax-H3 task " + task.id());
            }
        }
    }
}
