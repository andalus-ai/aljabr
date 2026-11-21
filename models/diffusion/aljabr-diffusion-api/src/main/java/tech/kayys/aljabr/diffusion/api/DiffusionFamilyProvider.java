package tech.kayys.aljabr.diffusion.api;

/**
 * Optional family-module contribution used to identify supported model
 * descriptors. Implementations should be stateless and safe to share.
 *
 * <p>Providers can be supplied explicitly or discovered with
 * {@link java.util.ServiceLoader}; the API does not create a global registry.</p>
 */
public interface DiffusionFamilyProvider {

    /** Stable, lowercase family identifier, for example {@code stable-diffusion} or {@code flux}. */
    String familyId();

    /** Returns whether this provider understands the supplied descriptor. */
    boolean supports(DiffusionModelDescriptor descriptor);
}
