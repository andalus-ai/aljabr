package tech.kayys.aljabr.safetensor.chronos2;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.models.chronos2.Chronos2CheckpointMapper;
import tech.kayys.aljabr.models.chronos2.Chronos2Config;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;

/** Validates Chronos-2 checkpoint metadata and binds zero-copy Safetensors views. */
public final class Chronos2SafetensorBindings {
    private final Chronos2CheckpointMapper mapper;
    public Chronos2SafetensorBindings() { this(new Chronos2CheckpointMapper()); }
    Chronos2SafetensorBindings(Chronos2CheckpointMapper mapper) {
        this.mapper = Objects.requireNonNull(mapper, "mapper");
    }
    public Map<String, Chronos2CheckpointMapper.TensorSpec> expectedTensors(Chronos2Config config) {
        return mapper.expectedTensors(Objects.requireNonNull(config, "config"));
    }
    public Binding bind(Chronos2Config config, Map<String, SafetensorTensor> tensors, int checkpointTensorCount) {
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
        for (String name : mapping.parameterBindings().keySet()) retained.put(name, tensors.get(name));
        return new Binding(mapping, retained);
    }
    public record Binding(Chronos2CheckpointMapper.CheckpointMapping mapping,
            Map<String, SafetensorTensor> tensorsByCheckpointName) {
        public Binding {
            Objects.requireNonNull(mapping, "mapping");
            tensorsByCheckpointName = Map.copyOf(Objects.requireNonNull(tensorsByCheckpointName, "tensorsByCheckpointName"));
            if (tensorsByCheckpointName.size() != mapping.expectedTensorCount()) {
                throw new IllegalArgumentException("every expected checkpoint tensor must be bound");
            }
        }
        public String checkpointNameForParameter(String parameterName) {
            Objects.requireNonNull(parameterName, "parameterName");
            return mapping.parameterBindings().entrySet().stream()
                    .filter(entry -> entry.getValue().equals(parameterName))
                    .map(Map.Entry::getKey).findFirst()
                    .orElseThrow(() -> new IllegalArgumentException("unknown Chronos-2 parameter: " + parameterName));
        }
        public SafetensorTensor tensorForParameter(String parameterName) {
            return tensorsByCheckpointName.get(checkpointNameForParameter(parameterName));
        }
    }
}
