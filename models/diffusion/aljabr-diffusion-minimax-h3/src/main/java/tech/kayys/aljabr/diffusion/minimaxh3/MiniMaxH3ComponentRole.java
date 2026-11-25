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
}
