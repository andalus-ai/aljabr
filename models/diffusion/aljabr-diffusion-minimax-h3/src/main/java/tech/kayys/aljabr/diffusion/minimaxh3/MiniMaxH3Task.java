package tech.kayys.aljabr.diffusion.minimaxh3;

/**
 * Tasks represented by the Comfy-Org MiniMax-H3 model bundle.
 *
 * <p>These identifiers describe model-family capabilities; they do not imply
 * that an Aljabr execution pipeline for the task is available yet.</p>
 */
public enum MiniMaxH3Task {
    TEXT_TO_VIDEO("text-to-video"),
    IMAGE_TO_VIDEO("image-to-video"),
    REFERENCE_TO_VIDEO("reference-to-video"),
    MULTIFRAME_REFERENCE_TO_VIDEO("multiframe-reference-to-video"),
    TEXT_TO_IMAGE("text-to-image"),
    IMAGE_EDIT("image-edit");

    private final String id;

    MiniMaxH3Task(String id) {
        this.id = id;
    }

    public String id() {
        return id;
    }

    public static boolean supports(String id) {
        if (id == null || id.isBlank()) {
            return false;
        }
        for (MiniMaxH3Task task : values()) {
            if (task.id.equals(id)) {
                return true;
            }
        }
        return false;
    }
}
