package tech.kayys.aljabr.diffusion.api;

/**
 * Format-specific adapter for opening a logical diffusion component artifact.
 *
 * <p>This loading SPI is independent of model families, storage providers and
 * compute backends. Implementations own parsing and the lifetime of the
 * returned format-specific resource.</p>
 *
 * @param <T> closeable loaded component/session
 */
public interface DiffusionComponentLoader<T extends AutoCloseable> {
    /** Stable format identifier used by component manifests. */
    String format();

    /** Match an artifact to this loader by its declared format. */
    default boolean supports(DiffusionComponentArtifact artifact) {
        return artifact != null && format().equalsIgnoreCase(artifact.format());
    }

    /**
     * Open an artifact and return a caller-owned resource handle.
     * Tensor views returned by the handle must not outlive that handle.
     */
    T load(DiffusionComponentArtifact artifact) throws DiffusionComponentLoadException;
}
