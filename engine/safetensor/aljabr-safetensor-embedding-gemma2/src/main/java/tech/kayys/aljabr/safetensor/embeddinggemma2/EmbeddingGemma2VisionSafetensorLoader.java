package tech.kayys.aljabr.safetensor.embeddinggemma2;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2CheckpointMapper;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2VisionConfig;
import tech.kayys.aljabr.models.embeddinggemma2.vision.EmbeddingGemma2VisionImagePipeline;
import tech.kayys.aljabr.safetensor.loader.SafetensorDType;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;

/**
 * Safetensors storage adapter for the EmbeddingGemma 2 image path.
 *
 * <p>The adapter owns file/shard access and dtype conversion. Model-family
 * tensor naming, shape validation, and executable pipeline composition remain
 * in the EmbeddingGemma 2 model module. Only the vision tower and image bridge
 * namespaces are materialized; language-model tensors are left untouched.</p>
 *
 * <p>Tensor bytes are read from the loader's zero-copy mapped views and
 * converted to detached float32 arrays because the current reference vision
 * operators consume heap float arrays. The resulting pipeline is independent
 * of the mapped session and remains usable after this loader closes it.</p>
 */
public final class EmbeddingGemma2VisionSafetensorLoader {
    private static final String VISION_PREFIX = "vision_tower.";
    private static final String BRIDGE_PREFIX = "embed_vision.";

    private final SafetensorShardLoader shardLoader;

    public EmbeddingGemma2VisionSafetensorLoader(SafetensorShardLoader shardLoader) {
        this.shardLoader = Objects.requireNonNull(shardLoader, "shardLoader must not be null");
    }

    /** Loads a single-file or sharded checkpoint and closes mapped shards after pipeline construction. */
    public EmbeddingGemma2VisionImagePipeline load(
            Path checkpoint,
            EmbeddingGemma2VisionConfig config,
            int textHiddenSize,
            float bridgeEpsilon,
            SafetensorDType checkpointDtype) {
        Objects.requireNonNull(checkpoint, "checkpoint must not be null");
        try (SafetensorShardLoader.SafetensorShardSession session = shardLoader.open(checkpoint)) {
            return EmbeddingGemma2VisionImagePipeline.fromCheckpoint(
                    config, mapTensors(session, checkpointDtype), textHiddenSize, bridgeEpsilon);
        }
    }

    /**
     * Loads a single-file or sharded checkpoint, inferring the vision checkpoint
     * dtype from its first selected tensor and rejecting mixed selected dtypes.
     */
    public EmbeddingGemma2VisionImagePipeline load(
            Path checkpoint,
            EmbeddingGemma2VisionConfig config,
            int textHiddenSize,
            float bridgeEpsilon) {
        Objects.requireNonNull(checkpoint, "checkpoint must not be null");
        try (SafetensorShardLoader.SafetensorShardSession session = shardLoader.open(checkpoint)) {
            return EmbeddingGemma2VisionImagePipeline.fromCheckpoint(
                    config, mapTensors(session), textHiddenSize, bridgeEpsilon);
        }
    }

    /**
     * Materializes vision and bridge tensors, inferring dtype from the first
     * selected tensor. The caller retains responsibility for closing session.
     */
    public Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> mapTensors(
            SafetensorShardLoader.SafetensorShardSession session) {
        Objects.requireNonNull(session, "session must not be null");
        SafetensorDType inferred = null;
        for (String name : session.tensorNames()) {
            if (name.startsWith(VISION_PREFIX) || name.startsWith(BRIDGE_PREFIX)) {
                inferred = session.tensor(name).dtype();
                break;
            }
        }
        if (inferred == null) {
            throw new IllegalArgumentException(
                    "checkpoint contains no EmbeddingGemma 2 vision or bridge tensors");
        }
        return mapTensors(session, inferred);
    }

    /**
     * Materializes the named vision and bridge tensors from an open session.
     * The caller retains responsibility for closing the session.
     */
    public Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> mapTensors(
            SafetensorShardLoader.SafetensorShardSession session,
            SafetensorDType checkpointDtype) {
        Objects.requireNonNull(session, "session must not be null");
        requireSupportedDtype(checkpointDtype);

        Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> tensors = new LinkedHashMap<>();
        for (String name : session.tensorNames()) {
            if (!name.startsWith(VISION_PREFIX) && !name.startsWith(BRIDGE_PREFIX)) {
                continue;
            }
            SafetensorTensor tensor = session.tensor(name);
            if (tensor.dtype() != checkpointDtype) {
                throw new IllegalArgumentException("EmbeddingGemma 2 vision checkpoint requires "
                        + checkpointDtype.jsonName() + " tensors, but '" + name + "' uses "
                        + tensor.dtype().jsonName());
            }
            tensors.put(name, toCheckpointTensor(tensor));
        }
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("checkpoint contains no EmbeddingGemma 2 vision or bridge tensors");
        }
        return Map.copyOf(tensors);
    }

    /** Converts one live mapped tensor to a detached, exclusively-owned float32 checkpoint tensor. */
    public static EmbeddingGemma2CheckpointMapper.CheckpointTensor toCheckpointTensor(
            SafetensorTensor tensor) {
        Objects.requireNonNull(tensor, "tensor must not be null");
        requireSupportedDtype(tensor.dtype());

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
        float[] values = tensor.toFloatArray();
        return EmbeddingGemma2CheckpointMapper.CheckpointTensor.takeOwnership(shape, values);
    }

    private static void requireSupportedDtype(SafetensorDType dtype) {
        Objects.requireNonNull(dtype, "checkpointDtype must not be null");
        if (dtype != SafetensorDType.BF16 && dtype != SafetensorDType.F16 && dtype != SafetensorDType.F32) {
            throw new IllegalArgumentException("EmbeddingGemma 2 vision supports BF16/F16/F32 checkpoint tensors, not "
                    + dtype.jsonName());
        }
    }
}
