package tech.kayys.aljabr.diffusion.minimaxh3;

/**
 * Logical component roles found in the Comfy-Org MiniMax-H3 model bundle.
 *
 * <p>Roles are independent of concrete filenames, quantization variants and
 * storage locations so loaders can resolve compatible artifacts separately.</p>
 */
public enum MiniMaxH3ComponentRole {
    DIFFUSION_MODEL,
    TEXT_ENCODER,
    VIDEO_AUTOENCODER,
    AUDIO_AUTOENCODER,
    ADAPTER,
    MODEL_PATCH,
    PROMPT_EMBEDDING

    /** Stable family-neutral role identifier used by diffusion manifests. */
    public String roleId() {
        return switch (this) {
            case DIFFUSION_MODEL -> "diffusion-model";
            case TEXT_ENCODER -> "text-encoder";
            case VIDEO_AUTOENCODER -> "video-autoencoder";
            case AUDIO_AUTOENCODER -> "audio-autoencoder";
            case ADAPTER -> "adapter";
            case MODEL_PATCH -> "model-patch";
            case PROMPT_EMBEDDING -> "prompt-embedding";
        };
    }
}
