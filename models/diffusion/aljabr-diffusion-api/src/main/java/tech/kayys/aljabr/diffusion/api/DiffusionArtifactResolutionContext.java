package tech.kayys.aljabr.diffusion.api;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Objects;

/** Immutable policy and environment supplied to artifact resolvers. */
public record DiffusionArtifactResolutionContext(
        Path cacheDirectory,
        Duration requestTimeout,
        boolean verifyChecksum) {

    public DiffusionArtifactResolutionContext {
        Objects.requireNonNull(cacheDirectory, "cacheDirectory");
        Objects.requireNonNull(requestTimeout, "requestTimeout");
        if (requestTimeout.isNegative() || requestTimeout.isZero()) {
            throw new IllegalArgumentException("requestTimeout must be positive");
        }
        cacheDirectory = cacheDirectory.toAbsolutePath().normalize();
    }
}
