package tech.kayys.aljabr.diffusion.core;

import tech.kayys.aljabr.diffusion.api.DiffusionArtifactResolutionContext;
import tech.kayys.aljabr.diffusion.api.DiffusionArtifactResolutionException;
import tech.kayys.aljabr.diffusion.api.DiffusionArtifactResolver;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentArtifact;
import tech.kayys.aljabr.diffusion.api.ResolvedDiffusionArtifact;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;

/** Resolves already-local paths and file URIs without copying or taking ownership. */
public final class LocalDiffusionArtifactResolver implements DiffusionArtifactResolver {
    @Override public String scheme() { return "file"; }

    @Override
    public boolean supports(DiffusionComponentArtifact artifact) {
        if (artifact == null) return false;
        String location = artifact.location();
        if (location.indexOf(':') < 0) return true;
        try { return "file".equalsIgnoreCase(URI.create(location).getScheme()); }
        catch (RuntimeException ignored) { return false; }
    }

    @Override
    public ResolvedDiffusionArtifact resolve(
            DiffusionComponentArtifact artifact, DiffusionArtifactResolutionContext context) {
        Objects.requireNonNull(artifact, "artifact");
        Objects.requireNonNull(context, "context");
        try {
            Path path = artifact.location().indexOf(':') < 0
                    ? Path.of(artifact.location()).toAbsolutePath().normalize()
                    : Path.of(URI.create(artifact.location())).toAbsolutePath().normalize();
            if (!Files.exists(path)) throw new IOException("Local artifact does not exist");
            if (!Files.isRegularFile(path) && !Files.isDirectory(path)) {
                throw new IOException("Local artifact must be a regular file or directory");
            }
            String digest = null;
            if (Files.isRegularFile(path)) {
                digest = sha256(path);
                if (artifact.sha256() != null && !artifact.sha256().equalsIgnoreCase(digest)) {
                    throw new IOException("Local artifact does not match its declared SHA-256");
                }
            } else if (artifact.sha256() != null) {
                throw new IOException("A directory artifact cannot be checked against a single-file SHA-256");
            }
            return new ResolvedDiffusionArtifact(artifact, path, null, digest, false);
        } catch (Exception e) {
            throw new DiffusionArtifactResolutionException(
                    artifact, "Unable to resolve local diffusion artifact", e);
        }
    }

    private static String sha256(Path path) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (var input = Files.newInputStream(path)) {
                byte[] buffer = new byte[1024 * 1024];
                int read;
                while ((read = input.read(buffer)) >= 0) if (read > 0) digest.update(buffer, 0, read);
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
    }
}
