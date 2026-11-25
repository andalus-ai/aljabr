package tech.kayys.aljabr.safetensor.chronos2;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.models.chronos2.Chronos2CheckpointMapper;
import tech.kayys.aljabr.models.chronos2.Chronos2Config;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;

/**
 * Validates Chronos-2 checkpoint metadata and binds the original zero-copy
 * Safetensors tensor views to logical model parameter names.
 *
 * <p>This storage adapter does not materialize float arrays. Every returned
 * tensor view remains backed by the loader-owned memory segment and is valid
 * only while the owning checkpoint session remains open.</p>
 */
public final class Chronos2SafetensorBindings {
    private final Chronos2CheckpointMapper mapper;

    public Chronos2SafetensorBindings() {
        this(new Chronos2CheckpointMapper());
    }

    Chronos2SafetensorBindings(Chronos2CheckpointMapper mapper) {
        this.mapper = Objects.requireNonNull(mapper, "mapper");
    }

    /**
     * Validate expected names, F32 dtypes and shapes, then retain the original
     * tensor views without copying payload bytes.
     *
     * @param config validated model configuration
     * @param tensors tensor views keyed by their checkpoint names
     * @param checkpointTensorCount total tensor count across all checkpoint shards
     */
    public Binding bind(Chronos2Config config, Map<String, SafetensorTensor> tensors,
            int checkpointTensorCount) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(tensors, "tensors");
        if (checkpointTensorCount < tensors.size()) {
            throw new IllegalArgumentException("checkpoint tensor count cannot be smaller than supplied tensor views");
        }

        Map<String, Chronos2CheckpointMapper.TensorMetadata> metadata = new LinkedHashMap<>();
        for (Map.Entry<String, SafetensorTensor> entry : tensors.entrySet()) {
            SafetensorTensor tensor = Objects.requireNonNull(entry.getValue(), "tensor view");
            if (!entry.getKey().equals(tensor.name())) {
                throw new IllegalArgumentException("tensor map key '" + entry.getKey()
                        + "' does not match tensor name '" + tensor.name() + "'");
            }
            metadata.put(entry.getKey(), new Chronos2CheckpointMapper.TensorMetadata(
                    tensor.dtype().jsonName(), tensor.shape()));
        }

        Chronos2CheckpointMapper.CheckpointMapping mapping = mapper.mapMetadata(config, metadata);
        if (checkpointTensorCount < mapping.expectedTensorCount()) {
            throw new IllegalArgumentException("checkpoint contains fewer tensors than Chronos-2 requires");
        }
        Map<String, SafetensorTensor> retained = new LinkedHashMap<>();
        for (String name : mapping.parameterBindings().keySet()) {
            retained.put(name, tensors.get(name));
        }
        return new Binding(mapping, retained);
    }

    /** Immutable name mapping plus references to loader-owned tensor views. */
    public record Binding(
            Chronos2CheckpointMapper.CheckpointMapping mapping,
            Map<String, SafetensorTensor> tensorsByCheckpointName) {
        public Binding {
            Objects.requireNonNull(mapping, "mapping");
            tensorsByCheckpointName = Map.copyOf(
                    Objects.requireNonNull(tensorsByCheckpointName, "tensorsByCheckpointName"));
            if (tensorsByCheckpointName.size() != mapping.expectedTensorCount()) {
                throw new IllegalArgumentException("every expected checkpoint tensor must be bound");
            }
        }

        public SafetensorTensor tensorForParameter(String parameterName) {
            Objects.requireNonNull(parameterName, "parameterName");
            String checkpointName = mapping.parameterBindings().entrySet().stream()
                    .filter(entry -> entry.getValue().equals(parameterName))
                    .map(Map.Entry::getKey)
                    .findFirst()
                    .orElseThrow(() -> new IllegalArgumentException(
                            "unknown Chronos-2 parameter: " + parameterName));
            return tensorsByCheckpointName.get(checkpointName);
        }
    }
}
