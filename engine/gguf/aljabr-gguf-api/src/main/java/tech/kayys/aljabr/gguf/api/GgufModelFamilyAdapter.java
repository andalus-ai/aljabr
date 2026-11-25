package tech.kayys.aljabr.gguf.api;

/**
 * Service-provider contract for architecture-specific GGUF model inspection.
 *
 * <p>Implementations belong in independently installable model-family modules.
 * They may validate architecture metadata and artifact structure, but must not
 * claim executable modality support: that is established separately by a
 * compatible execution provider.</p>
 */
public interface GgufModelFamilyAdapter {

    /** Stable, namespaced identifier for this model family. */
    String familyId();

    /**
     * Inspects an already parsed artifact and returns a typed compatibility
     * result. Implementations must be deterministic and must not load tensors
     * or allocate native execution resources.
     */
    GgufModelFamilyInspection inspect(GgufModelDescriptor descriptor);
}
