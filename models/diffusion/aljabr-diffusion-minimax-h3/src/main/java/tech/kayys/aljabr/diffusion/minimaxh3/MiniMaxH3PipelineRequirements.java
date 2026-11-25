package tech.kayys.aljabr.diffusion.minimaxh3;

import tech.kayys.aljabr.diffusion.api.DiffusionComponentRequirement;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Builds the component requirements for a MiniMax-H3 task from the runtime
 * conditioning and output choices.
 *
 * <p>This is a family capability description, not a pipeline executor. The
 * generic diffusion core remains responsible for resolving, materializing,
 * loading and managing component resources.</p>
 */
public final class MiniMaxH3PipelineRequirements {

    private MiniMaxH3PipelineRequirements() {
    }

    /**
     * Returns the checkpoint components needed for the selected task path.
     *
     * <p>Input images/reference frames are request inputs and are deliberately
     * not represented as checkpoint artifacts by this profile.</p>
     */
    public static List<DiffusionComponentRequirement> forTask(
            MiniMaxH3Task task,
            Conditioning conditioning,
            Output output) {
        Objects.requireNonNull(task, "task");
        Objects.requireNonNull(conditioning, "conditioning");
        Objects.requireNonNull(output, "output");

        boolean videoTask = isVideoTask(task);
        if (videoTask != (output == Output.VIDEO)) {
            throw new IllegalArgumentException(
                    "Task " + task.id() + " requires " + (videoTask ? "VIDEO" : "IMAGE") + " output");
        }

        List<DiffusionComponentRequirement> requirements = new ArrayList<>();
        requirements.add(required(MiniMaxH3ComponentRole.DIFFUSION_MODEL));

        switch (conditioning) {
            case LOCAL_TEXT_ENCODER ->
                    requirements.add(required(MiniMaxH3ComponentRole.TEXT_ENCODER));
            case PRECOMPUTED_PROMPT_EMBEDDING ->
                    requirements.add(required(MiniMaxH3ComponentRole.PROMPT_EMBEDDING));
        }

        if (output == Output.VIDEO) {
            requirements.add(required(MiniMaxH3ComponentRole.VIDEO_AUTOENCODER));
        }
        return List.copyOf(requirements);
    }

    /**
     * Adds the audio decoder only for requests that actually produce audio.
     * Audio is intentionally not a mandatory dependency of silent generation.
     */
    public static List<DiffusionComponentRequirement> withAudioOutput(
            List<DiffusionComponentRequirement> requirements) {
        Objects.requireNonNull(requirements, "requirements");
        List<DiffusionComponentRequirement> result = new ArrayList<>(requirements);
        if (result.stream().noneMatch(r ->
                r.role().equals(MiniMaxH3ComponentRole.AUDIO_AUTOENCODER.roleId()))) {
            result.add(required(MiniMaxH3ComponentRole.AUDIO_AUTOENCODER));
        }
        return List.copyOf(result);
    }

    private static DiffusionComponentRequirement required(MiniMaxH3ComponentRole role) {
        return new DiffusionComponentRequirement(role.roleId(), true);
    }

    private static boolean isVideoTask(MiniMaxH3Task task) {
        return switch (task) {
            case TEXT_TO_VIDEO, IMAGE_TO_VIDEO, REFERENCE_TO_VIDEO, MULTIFRAME_REFERENCE_TO_VIDEO -> true;
            case TEXT_TO_IMAGE, IMAGE_EDIT -> false;
        };
    }

    public enum Conditioning {
        /** Encode text prompts using the checkpoint's local text encoder. */
        LOCAL_TEXT_ENCODER,
        /** Consume prompt embeddings produced by an upstream conditioning stage. */
        PRECOMPUTED_PROMPT_EMBEDDING
    }

    public enum Output {
        IMAGE,
        VIDEO
    }
}
