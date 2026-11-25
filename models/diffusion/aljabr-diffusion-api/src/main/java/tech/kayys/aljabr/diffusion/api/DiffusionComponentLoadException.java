package tech.kayys.aljabr.diffusion.api;

/** Raised when a format adapter cannot open a diffusion component artifact. */
public final class DiffusionComponentLoadException extends RuntimeException {

    private final String artifactId;
    private final String format;

    public DiffusionComponentLoadException(
            DiffusionComponentArtifact artifact, String message, Throwable cause) {
        super(message, cause);
        this.artifactId = artifact == null ? null : artifact.id();
        this.format = artifact == null ? null : artifact.format();
    }

    public DiffusionComponentLoadException(
            DiffusionComponentArtifact artifact, String message) {
        this(artifact, message, null);
    }

    public String artifactId() {
        return artifactId;
    }

    public String format() {
        return format;
    }
}
