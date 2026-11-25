package tech.kayys.aljabr.diffusion.safetensors;

import tech.kayys.aljabr.diffusion.api.DiffusionComponentArtifact;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentLoadException;
import tech.kayys.aljabr.diffusion.api.DiffusionComponentLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;

import java.net.URI;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Set;

/** SafeTensors adapter delegating parsing and shard lifetime to the existing loader. */
public final class SafetensorsDiffusionComponentLoader
        implements DiffusionComponentLoader<SafetensorsDiffusionComponent> {
    public static final String FORMAT = "safetensors";
    private final SafetensorShardLoader shardLoader;

    public SafetensorsDiffusionComponentLoader(SafetensorShardLoader shardLoader) {
        this.shardLoader = Objects.requireNonNull(shardLoader, "shardLoader");
    }

    @Override public String format() { return FORMAT; }

    @Override
    public SafetensorsDiffusionComponent load(DiffusionComponentArtifact artifact) {
        Objects.requireNonNull(artifact, "artifact");
        if (!supports(artifact)) {
            throw new DiffusionComponentLoadException(
                    artifact, "Unsupported component format: " + artifact.format());
        }
        try {
            return new SafetensorsDiffusionComponent(
                    artifact, shardLoader.open(resolveLocalPath(artifact.location())));
        } catch (RuntimeException failure) {
            throw new DiffusionComponentLoadException(
                    artifact, "Unable to open SafeTensors component at " + artifact.location(), failure);
        }
    }

    static Path resolveLocalPath(String location) {
        Objects.requireNonNull(location, "location");
        if (location.isBlank()) throw new IllegalArgumentException("Component location must not be blank");
        if (location.startsWith("file:")) {
            return Path.of(URI.create(location)).toAbsolutePath().normalize();
        }
        if (location.matches("^[a-zA-Z][a-zA-Z0-9+.-]*:.*")) {
            throw new IllegalArgumentException(
                    "Unsupported SafeTensors location scheme; expected a local path or file URI: " + location);
        }
        return Path.of(location).toAbsolutePath().normalize();
    }

    /** Owned handle; all tensor views become invalid when this handle is closed. */
    public static final class SafetensorsDiffusionComponent implements AutoCloseable {
        private final DiffusionComponentArtifact artifact;
        private final SafetensorShardSession session;

        private SafetensorsDiffusionComponent(
                DiffusionComponentArtifact artifact, SafetensorShardSession session) {
            this.artifact = artifact;
            this.session = session;
        }

        public DiffusionComponentArtifact artifact() { return artifact; }
        public SafetensorTensor tensor(String name) { return session.tensor(name); }
        public boolean hasTensor(String name) { return session.findTensor(name).isPresent(); }
        public Set<String> tensorNames() { return session.tensorNames(); }
        public int tensorCount() { return session.tensorCount(); }
        public boolean isSharded() { return session.isSharded(); }
        @Override public void close() { session.close(); }
    }
}
