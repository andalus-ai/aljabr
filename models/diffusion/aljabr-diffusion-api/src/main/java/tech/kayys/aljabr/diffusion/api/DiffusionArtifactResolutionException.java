package tech.kayys.aljabr.diffusion.api;

/** Raised when a component artifact cannot be resolved or verified. */
public final class DiffusionArtifactResolutionException extends RuntimeException {
    private final String artifactId;
    private final String scheme;

    public DiffusionArtifactResolutionException(
            DiffusionComponentArtifact artifact, String message, Throwable cause) {
        super(message, cause);
        this.artifactId = artifact == null ? null : artifact.id();
        this.scheme = schemeOf(artifact);
    }

    public DiffusionArtifactResolutionException(
            DiffusionComponentArtifact artifact, String message) {
        this(artifact, message, null);
    }

    public String artifactId() { return artifactId; }
    public String scheme() { return scheme; }

    private static String schemeOf(DiffusionComponentArtifact artifact) {
        if (artifact == null || artifact.location() == null) return null;
        int separator = artifact.location().indexOf(':');
        return separator <= 0 ? "file" : artifact.location().substring(0, separator).toLowerCase();
    }
}
