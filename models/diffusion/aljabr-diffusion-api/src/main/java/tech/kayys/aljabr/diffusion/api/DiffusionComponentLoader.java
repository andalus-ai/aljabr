package tech.kayys.aljabr.diffusion.api;

/**
 * Format-specific adapter for opening a logical diffusion component artifact.
 *
 * <p>This is a loading SPI, not a model-family abstraction. Implementations
 * belong in format integration modules (for example SafeTensors or a future
 * native checkpoint adapter). They own location resolution, parsing and
 * resource lifetime; the diffusion API remains independent of storage and
 * tensor implementations.</p>
 *
 * @param <T> the format-specific, closeable loaded component/session
 */
public interface DiffusionComponentLoader<T extends AutoCloseable> {

    /** Stable format identifier understood by component manifests. */
    String format();

    /**
     * Whether this loader can open the artifact. The default implementation
     * matches the declared format; adapters may additionally inspect metadata.
     */
    default boolean supports(DiffusionComponentArtifact artifact) {
        return artifact != null && format().equalsIgnoreCase(artifact.format());
    }

    /**
     * Open an artifact and return an owned resource handle.
     *
     * <p>The caller owns the returned handle and must close it. Implementations
     * should not return tensor views whose backing memory can outlive that
     * handle.</p>
     *
     * @throws DiffusionComponentLoadException if the artifact cannot be opened
     */
    T load(DiffusionComponentArtifact artifact) throws DiffusionComponentLoadException;
}
