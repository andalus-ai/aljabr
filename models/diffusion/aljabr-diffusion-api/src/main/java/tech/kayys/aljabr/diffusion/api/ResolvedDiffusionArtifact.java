package tech.kayys.aljabr.diffusion.api;

import java.nio.file.Path;
import java.util.Objects;

/**
 * A repository-independent, locally materialized view of a manifest artifact.
 * The resolved file remains owned by the resolver/cache; closing a format
 * loader handle does not evict it.
 */
public record ResolvedDiffusionArtifact(
        DiffusionComponentArtifact source,
        Path localPath,
        String resolvedRevision,
        String sha256,
        boolean cacheHit) {

    public ResolvedDiffusionArtifact {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(localPath, "localPath");
        localPath = localPath.toAbsolutePath().normalize();
        resolvedRevision = normalize(resolvedRevision);
        sha256 = normalize(sha256);
        if (sha256 != null && !sha256.matches("[0-9a-fA-F]{64}")) {
            throw new IllegalArgumentException("sha256 must contain exactly 64 hexadecimal characters");
        }
    }

    /** Adapt the resolved local file back to the existing format-loader contract. */
    public DiffusionComponentArtifact asLocalArtifact() {
        return new DiffusionComponentArtifact(
                source.id(), source.role(), localPath.toUri().toString(), source.format(),
                source.variant(), sha256 == null ? source.sha256() : sha256,
                source.required(), source.metadata());
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? null : value;
    }
}
