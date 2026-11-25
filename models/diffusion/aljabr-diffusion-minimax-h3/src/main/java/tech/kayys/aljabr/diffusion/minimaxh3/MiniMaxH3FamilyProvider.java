package tech.kayys.aljabr.diffusion.minimaxh3;

import tech.kayys.aljabr.diffusion.api.DiffusionFamilyProvider;
import tech.kayys.aljabr.diffusion.api.DiffusionModelDescriptor;

/**
 * Identifies MiniMax-H3 descriptors for optional diffusion-family discovery.
 *
 * <p>This provider deliberately performs no checkpoint loading or pipeline
 * construction. Those contracts are not yet present in the family-neutral
 * diffusion API, and accepting a descriptor must not be mistaken for runtime
 * inference support.</p>
 */
public final class MiniMaxH3FamilyProvider implements DiffusionFamilyProvider {

    public static final String FAMILY_ID = "minimax-h3";
    public static final String ARCHITECTURE_ID = "minimax-h3";

    @Override
    public String familyId() {
        return FAMILY_ID;
    }

    @Override
    public boolean supports(DiffusionModelDescriptor descriptor) {
        if (descriptor == null
                || !FAMILY_ID.equals(descriptor.familyId())
                || !ARCHITECTURE_ID.equals(descriptor.architecture())) {
            return false;
        }

        String task = descriptor.attributes().get("task");
        return task == null || MiniMaxH3Task.supports(task);
    }
}
