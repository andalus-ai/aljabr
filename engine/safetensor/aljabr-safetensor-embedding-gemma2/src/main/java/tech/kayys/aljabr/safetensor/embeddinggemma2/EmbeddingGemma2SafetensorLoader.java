package tech.kayys.aljabr.safetensor.embeddinggemma2;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2CheckpointMapper;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2TextBackbone;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;
import tech.kayys.aljabr.safetensor.loader.SafetensorDType;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;

/**
 * Safetensors storage adapter for the EmbeddingGemma 2 text model.
 *
 * <p>All file and shard lifecycle management remains in Aljabr's shared
 * Safetensors loader. This adapter only selects text tensors, converts BF16, F16, or F32 payloads to detached float32 arrays, and delegates tensor-name and
 * shape validation to the model-family checkpoint mapper.</p>
 *
 * <p>The returned backbone owns heap arrays and is independent of the
 * memory-mapped session, which is closed before this method returns.</p>
 */
public final class EmbeddingGemma2SafetensorLoader {
    private static final String TEXT_PREFIX = "language_model.";

    private final SafetensorShardLoader shardLoader;

    public EmbeddingGemma2SafetensorLoader(SafetensorShardLoader shardLoader) {
        this.shardLoader = Objects.requireNonNull(shardLoader, "shardLoader must not be null");
    }

    /** Loads a single-file or sharded checkpoint and closes all mapped shards after materialization. */
    public EmbeddingGemma2TextBackbone load(Path checkpoint, EmbeddingGemma2Config config) {
        Objects.requireNonNull(checkpoint, "checkpoint must not be null");
        Objects.requireNonNull(config, "config must not be null");
        try (SafetensorShardLoader.SafetensorShardSession session = shardLoader.open(checkpoint)) {
            return new EmbeddingGemma2TextBackbone(config, mapWeights(session, config));
        }
    }

    /**
     * Materializes the text checkpoint bundle from an already-open session.
     * The caller retains responsibility for closing the session.
     */
    public EmbeddingGemma2TextBackbone.Weights mapWeights(
            SafetensorShardLoader.SafetensorShardSession session,
            EmbeddingGemma2Config config) {
        Objects.requireNonNull(session, "session must not be null");
        Objects.requireNonNull(config, "config must not be null");

        SafetensorDType expectedDtype = "bfloat16".equalsIgnoreCase(config.dtype())
                ? SafetensorDType.BF16 : SafetensorDType.F32;
        Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> tensors = new LinkedHashMap<>();
        for (String name : session.tensorNames()) {
            if (!name.startsWith(TEXT_PREFIX)) {
                continue;
            }
            SafetensorTensor tensor = session.tensor(name);
            if (tensor.dtype() != expectedDtype) {
                throw new IllegalArgumentException("EmbeddingGemma 2 config dtype " + config.dtype()
                        + " requires " + expectedDtype.jsonName() + " tensors, but '" + name
                        + "' uses " + tensor.dtype().jsonName());
            }
            tensors.put(name, toCheckpointTensor(tensor));
        }
        return EmbeddingGemma2CheckpointMapper.map(config, tensors);
    }

    /**
     * Converts one live SafeTensors tensor to an owned, detached float32 value.
     * The returned value remains valid after the parent mapped session closes.
     */
    public static EmbeddingGemma2CheckpointMapper.CheckpointTensor toCheckpointTensor(
            SafetensorTensor tensor) {
        Objects.requireNonNull(tensor, "tensor must not be null");
        SafetensorDType dtype = tensor.dtype();
        if (dtype != SafetensorDType.BF16 && dtype != SafetensorDType.F16
                && dtype != SafetensorDType.F32) {
            throw new IllegalArgumentException("EmbeddingGemma 2 supports BF16/F16/F32 checkpoint tensors, not "
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
        // toFloatArray performs the loader's little-endian BF16 expansion or F32 copy.
        float[] values = tensor.toFloatArray();
        return EmbeddingGemma2CheckpointMapper.CheckpointTensor.takeOwnership(shape, values);
    }
}
