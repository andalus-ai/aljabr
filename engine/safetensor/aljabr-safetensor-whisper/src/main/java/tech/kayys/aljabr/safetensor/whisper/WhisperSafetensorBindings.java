package tech.kayys.aljabr.safetensor.whisper;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import tech.kayys.aljabr.models.whisper.WhisperEncoderCheckpointMapper;
import tech.kayys.aljabr.models.whisper.WhisperLargeV3Config;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;

/**
 * Binds validated Whisper encoder parameter names to zero-copy Safetensors views.
 *
 * <p>The model module owns the architecture contract; this optional storage
 * module owns the Safetensors integration. Returned views remain valid only
 * while the supplied {@link WhisperSafetensorCheckpoint} remains open.</p>
 */
public final class WhisperSafetensorBindings {
    private final WhisperEncoderCheckpointMapper mapper;

    public WhisperSafetensorBindings() {
        this(new WhisperEncoderCheckpointMapper());
    }

    WhisperSafetensorBindings(WhisperEncoderCheckpointMapper mapper) {
        this.mapper = Objects.requireNonNull(mapper, "mapper");
    }

    /** Returns the expected HF checkpoint names and their logical parameter shapes. */
    public Map<String, WhisperEncoderCheckpointMapper.TensorSpec> expectedTensors(
            WhisperLargeV3Config config) {
        return mapper.expectedTensors(Objects.requireNonNull(config, "config"));
    }

    /**
     * Resolves all encoder tensors, validates their names/dtypes/shapes, and
     * returns logical parameter bindings without copying tensor payloads.
     */
    public Binding bind(WhisperLargeV3Config config, WhisperSafetensorCheckpoint checkpoint) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(checkpoint, "checkpoint");
        Map<String, SafetensorTensor> views = new LinkedHashMap<>();
        Map<String, WhisperEncoderCheckpointMapper.TensorMetadata> metadata = new LinkedHashMap<>();
        for (String name : mapper.expectedTensors(config).keySet()) {
            SafetensorTensor tensor = checkpoint.tensor(name);
            views.put(name, tensor);
            metadata.put(name, new WhisperEncoderCheckpointMapper.TensorMetadata(
                    tensor.dtype().jsonName(), tensor.shape()));
        }
        WhisperEncoderCheckpointMapper.CheckpointMapping mapping =
                mapper.mapMetadata(config, metadata);
        return new Binding(mapping, views);
    }

    /** Immutable parameter map; underlying views are owned by the checkpoint. */
    public record Binding(
            WhisperEncoderCheckpointMapper.CheckpointMapping mapping,
            Map<String, SafetensorTensor> tensorsByCheckpointName) {
        public Binding {
            Objects.requireNonNull(mapping, "mapping");
            tensorsByCheckpointName = Map.copyOf(
                    Objects.requireNonNull(tensorsByCheckpointName, "tensorsByCheckpointName"));
            if (tensorsByCheckpointName.size() != mapping.parameterBindings().size()) {
                throw new IllegalArgumentException("every expected Whisper encoder tensor must be bound");
            }
        }

        /** Returns the zero-copy view for an Aljabr logical parameter name. */
        public SafetensorTensor tensorForParameter(String parameterName) {
            Objects.requireNonNull(parameterName, "parameterName");
            return mapping.parameterBindings().entrySet().stream()
                    .filter(entry -> entry.getValue().parameterName().equals(parameterName))
                    .map(entry -> tensorsByCheckpointName.get(entry.getKey()))
                    .findFirst()
                    .orElseThrow(() -> new IllegalArgumentException(
                            "unknown Whisper encoder parameter: " + parameterName));
        }
    }
}
