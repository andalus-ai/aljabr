package tech.kayys.aljabr.diffusion.api;

/**
 * Repository/storage-specific materializer for diffusion component artifacts.
 * Implementations may resolve local paths, Hugging Face repositories, object
 * stores, or other schemes without coupling model families to those providers.
 */
public interface DiffusionArtifactResolver {

    /** URI scheme handled by this resolver, for example {@code hf} or {@code file}. */
    String scheme();

    /** Whether this resolver accepts the artifact location. */
    boolean supports(DiffusionComponentArtifact artifact);

    /**
     * Resolve and, when necessary, materialize an artifact in the supplied
     * cache. Implementations must verify a declared checksum before returning.
     */
    ResolvedDiffusionArtifact resolve(
            DiffusionComponentArtifact artifact,
            DiffusionArtifactResolutionContext context)
            throws DiffusionArtifactResolutionException;
}
