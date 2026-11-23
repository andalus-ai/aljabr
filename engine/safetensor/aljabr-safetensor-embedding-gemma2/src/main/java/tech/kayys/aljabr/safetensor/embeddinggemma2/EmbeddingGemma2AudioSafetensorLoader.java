package tech.kayys.aljabr.safetensor.embeddinggemma2;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2CheckpointMapper;
import tech.kayys.aljabr.models.embeddinggemma2.audio.EmbeddingGemma2AudioCheckpointMapper;
import tech.kayys.aljabr.models.embeddinggemma2.audio.EmbeddingGemma2AudioEncoder;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;
import tech.kayys.aljabr.safetensor.loader.SafetensorDType;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;

/**
 * Safetensors storage adapter for the EmbeddingGemma 2 audio tower.
 *
 * <p>This adapter owns shard lifecycle and dtype conversion. Audio checkpoint
 * names, shapes, and model-specific weight assembly remain in the model
 * module's {@link EmbeddingGemma2AudioCheckpointMapper}. Only tensors in the
 * {@code audio_tower.} namespace are materialized; text and vision weights
 * remain untouched.</p>
 *
 * <p>The current Java reference encoder consumes detached float32 arrays.
 * Therefore the mapped Safetensors session is closed after construction and
 * the returned encoder remains independent of the file mapping.</p>
 */
public final class EmbeddingGemma2AudioSafetensorLoader {
    private static final String AUDIO_PREFIX = "audio_tower.";

    private final SafetensorShardLoader shardLoader;

    public EmbeddingGemma2AudioSafetensorLoader(SafetensorShardLoader shardLoader) {
        this.shardLoader = Objects.requireNonNull(shardLoader, "shardLoader must not be null");
    }

    /** Loads the audio tower from a single-file or sharded Safetensors checkpoint. */
    public EmbeddingGemma2AudioEncoder load(
            Path checkpoint, EmbeddingGemma2AudioConfig config) {
        Objects.requireNonNull(checkpoint, "checkpoint must not be null");
        Objects.requireNonNull(config, "config must not be null");
        try (SafetensorShardLoader.SafetensorShardSession session = shardLoader.open(checkpoint)) {
            return new EmbeddingGemma2AudioEncoder(config, mapWeights(session, config));
        }
    }

    /**
     * Materializes and maps the audio checkpoint bundle from an open session.
     * The caller retains responsibility for closing the session.
     */
    public EmbeddingGemma2AudioEncoder.Weights mapWeights(
            SafetensorShardLoader.SafetensorShardSession session,
            EmbeddingGemma2AudioConfig config) {
        Objects.requireNonNull(session, "session must not be null");
        Objects.requireNonNull(config, "config must not be null");

        SafetensorDType expectedDtype = "bfloat16".equalsIgnoreCase(config.dtype())
                ? SafetensorDType.BF16 : SafetensorDType.F32;
        Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> tensors = new LinkedHashMap<>();
        for (String name : session.tensorNames()) {
            if (!name.startsWith(AUDIO_PREFIX)) {
                continue;
            }
            SafetensorTensor tensor = session.tensor(name);
            if (tensor.dtype() != expectedDtype) {
                throw new IllegalArgumentException("EmbeddingGemma 2 audio config dtype " + config.dtype()
                        + " requires " + expectedDtype.jsonName() + " tensors, but '" + name
                        + "' uses " + tensor.dtype().jsonName());
            }
            tensors.put(name, toCheckpointTensor(tensor));
        }
        return EmbeddingGemma2AudioCheckpointMapper.map(config, tensors);
    }

    /** Converts a live mapped tensor to an owned, detached float32 checkpoint tensor. */
    public static EmbeddingGemma2CheckpointMapper.CheckpointTensor toCheckpointTensor(
            SafetensorTensor tensor) {
        Objects.requireNonNull(tensor, "tensor must not be null");
        SafetensorDType dtype = tensor.dtype();
        if (dtype != SafetensorDType.BF16 && dtype != SafetensorDType.F32) {
            throw new IllegalArgumentException("EmbeddingGemma 2 audio supports BF16/F32 checkpoint tensors, not "
                    + dtype.jsonName() + " for '" + tensor.name() + "'");
        }

        long[] sourceShape = tensor.shape();
        int[] shape = new int[sourceShape.length];
        for (int dimension = 0; dimension < sourceShape.length; dimension++) {
            try {
                shape[dimension] = Math.toIntExact(sourceShape[dimension]);
            } catch (ArithmeticException e) {
                throw new IllegalArgumentException("tensor dimension exceeds Java array limits for '"
                        + tensor.name() + "'", e);
            }
        }
        // SafetensorTensor performs little-endian BF16 expansion or F32 copy.
        return EmbeddingGemma2CheckpointMapper.CheckpointTensor.takeOwnership(shape, tensor.toFloatArray());
    }
}
