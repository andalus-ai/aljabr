package tech.kayys.aljabr.safetensor.bert;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import tech.kayys.aljabr.models.bert.BertCheckpointMapper;
import tech.kayys.aljabr.models.bert.BertConfig;
import tech.kayys.aljabr.models.bert.BertEncoder;
import tech.kayys.aljabr.safetensor.loader.SafetensorDType;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;

/**
 * Safetensors storage adapter for reusable BERT-family encoders.
 *
 * <p>The shared shard loader owns file discovery, mmap, and shard lifetime.
 * This adapter validates and converts only the model's expected dense
 * parameters, then copies them into the encoder's backend-owned state.</p>
 */
@ApplicationScoped
public final class BertSafetensorLoader {
    private final SafetensorShardLoader shardLoader;

    @Inject
    public BertSafetensorLoader(SafetensorShardLoader shardLoader) {
        this.shardLoader = Objects.requireNonNull(shardLoader, "shardLoader");
    }

    /** Loads a single-file or sharded checkpoint into an already configured encoder. */
    public void load(Path checkpoint, BertEncoder encoder) {
        Objects.requireNonNull(checkpoint, "checkpoint");
        Objects.requireNonNull(encoder, "encoder");
        BertCheckpointMapper mapper = new BertCheckpointMapper();
        BertConfig config = encoder.config();
        Set<String> expectedNames = mapper.expectedTensorShapes(config).keySet();
        try (SafetensorShardLoader.SafetensorShardSession session = shardLoader.open(checkpoint)) {
            Map<String, BertCheckpointMapper.CheckpointTensor> tensors = new LinkedHashMap<>();
            for (String name : expectedNames) {
                SafetensorTensor tensor = session.findTensor(name).orElseThrow(
                        () -> new IllegalArgumentException("Missing required BERT Safetensors tensor: " + name));
                tensors.put(name, toCheckpointTensor(tensor));
            }
            // The mapper validates names and dimensions; BertEncoder performs a strict,
            // shape-checked copy into backend-owned parameters before the mmap is closed.
            encoder.loadCheckpoint(tensors);
        }
    }

    /** Converts one live F32/F16/BF16 tensor to an owned dense F32 checkpoint value. */
    public static BertCheckpointMapper.CheckpointTensor toCheckpointTensor(SafetensorTensor tensor) {
        Objects.requireNonNull(tensor, "tensor");
        SafetensorDType dtype = tensor.dtype();
        if (dtype != SafetensorDType.F32 && dtype != SafetensorDType.F16
                && dtype != SafetensorDType.BF16) {
            throw new IllegalArgumentException("BERT checkpoints require F32/F16/BF16 weights; tensor '"
                    + tensor.name() + "' uses " + dtype.jsonName());
        }
        long[] sourceShape = tensor.shape();
        int[] shape = new int[sourceShape.length];
        for (int i = 0; i < sourceShape.length; i++) {
            try {
                shape[i] = Math.toIntExact(sourceShape[i]);
            } catch (ArithmeticException failure) {
                throw new IllegalArgumentException("BERT tensor dimension exceeds Java array limits: "
                        + tensor.name(), failure);
            }
        }
        return new BertCheckpointMapper.CheckpointTensor(shape, tensor.toFloatArray());
    }
}
