package tech.kayys.aljabr.safetensor.chronos2;

import java.nio.file.Path;
import java.util.Objects;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.aljabr.models.chronos2.Chronos2Config;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;

/**
 * Opens single-file or sharded Chronos-2 checkpoints through Aljabr's shared
 * Safetensors loader. Model-family metadata validation and parameter binding
 * are delegated to {@link Chronos2SafetensorBindings}.
 */
@ApplicationScoped
public final class Chronos2SafetensorLoader {
    private final SafetensorShardLoader shardLoader;
    private final Chronos2SafetensorBindings bindings;

    @Inject
    public Chronos2SafetensorLoader(SafetensorShardLoader shardLoader) {
        this(shardLoader, new Chronos2SafetensorBindings());
    }

    Chronos2SafetensorLoader(SafetensorShardLoader shardLoader, Chronos2SafetensorBindings bindings) {
        this.shardLoader = Objects.requireNonNull(shardLoader, "shardLoader");
        this.bindings = Objects.requireNonNull(bindings, "bindings");
    }

    /**
     * Opens a checkpoint and retains its mapped shard session for the returned
     * checkpoint lifetime. The caller must close the returned checkpoint.
     */
    public Chronos2SafetensorCheckpoint open(Path checkpoint, Chronos2Config config) {
        Objects.requireNonNull(checkpoint, "checkpoint");
        Objects.requireNonNull(config, "config");
        SafetensorShardLoader.SafetensorShardSession session = shardLoader.open(checkpoint);
        return new Chronos2SafetensorCheckpoint(session, config, bindings);
    }
}
