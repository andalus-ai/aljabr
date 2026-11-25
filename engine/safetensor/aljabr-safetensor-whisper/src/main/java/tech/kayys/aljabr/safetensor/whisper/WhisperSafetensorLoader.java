package tech.kayys.aljabr.safetensor.whisper;

import java.nio.file.Path;
import java.util.Objects;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.aljabr.models.whisper.WhisperLargeV3Config;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;

/**
 * Opens single-file and sharded Whisper checkpoints through Aljabr's shared
 * zero-copy Safetensors loader after validating the Hugging Face architecture.
 */
@ApplicationScoped
public final class WhisperSafetensorLoader {
    private final SafetensorShardLoader shardLoader;

    @Inject
    public WhisperSafetensorLoader(SafetensorShardLoader shardLoader) {
        this.shardLoader = Objects.requireNonNull(shardLoader, "shardLoader");
    }

    /**
     * Opens a large-v3 checkpoint directory or a Safetensors file belonging to
     * a directory that contains its Hugging Face config.json.
     *
     * <p>For model directories, the architecture is validated before any
     * weight shard is mapped. The returned checkpoint owns the shard session.</p>
     */
    public WhisperSafetensorCheckpoint open(Path checkpointDirectory) {
        Objects.requireNonNull(checkpointDirectory, "checkpointDirectory");
        WhisperLargeV3Config config = WhisperLargeV3Config.fromCheckpoint(checkpointDirectory);
        config.requireLargeV3Architecture();
        return new WhisperSafetensorCheckpoint(shardLoader.open(checkpointDirectory));
    }
}
