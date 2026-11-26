package tech.kayys.aljabr.models.whisper;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class WhisperEncoderCheckpointMapperTest {

    private final WhisperEncoderCheckpointMapper mapper = new WhisperEncoderCheckpointMapper();
    private final WhisperLargeV3Config config = WhisperLargeV3Config.canonical();

    @Test
    void describesCompleteLargeV3EncoderLayout() {
        Map<String, WhisperEncoderCheckpointMapper.TensorSpec> expected = mapper.expectedTensors(config);

        assertEquals(519, expected.size());
        assertTrue(java.util.Arrays.equals(new long[] {1280, 128, 3},
                expected.get("model.encoder.conv1.weight").shape()));
        assertTrue(java.util.Arrays.equals(new long[] {5120, 1280},
                expected.get("model.encoder.layers.31.fc1.weight").shape()));
        assertTrue(expected.containsKey("model.encoder.layer_norm.weight"));
    }

    @Test
    void validatesEveryExpectedTensorWithoutRequiringDecoderWeights() {
        Map<String, WhisperEncoderCheckpointMapper.TensorMetadata> metadata = new LinkedHashMap<>();
        mapper.expectedTensors(config).forEach((name, spec) ->
                metadata.put(name, new WhisperEncoderCheckpointMapper.TensorMetadata("F16", spec.shape())));

        var mapping = mapper.mapMetadata(config, metadata);

        assertEquals(519, mapping.parameterBindings().size());
        assertEquals("encoder.layers.0.self_attn.q_proj.weight",
                mapping.parameterBindings().get(
                        "model.encoder.layers.0.self_attn.q_proj.weight").parameterName());
    }

    @Test
    void rejectsMissingTensor() {
        assertThrows(IllegalArgumentException.class,
                () -> mapper.mapMetadata(config, Map.of()));
    }

    @Test
    void rejectsWrongShapeAndNonFloatingDtype() {
        Map<String, WhisperEncoderCheckpointMapper.TensorMetadata> metadata = new LinkedHashMap<>();
        mapper.expectedTensors(config).forEach((name, spec) ->
                metadata.put(name, new WhisperEncoderCheckpointMapper.TensorMetadata("F32", spec.shape())));

        metadata.put("model.encoder.conv1.weight",
                new WhisperEncoderCheckpointMapper.TensorMetadata("F32", new long[] {1280, 128, 1}));
        assertThrows(IllegalArgumentException.class, () -> mapper.mapMetadata(config, metadata));

        metadata.put("model.encoder.conv1.weight",
                new WhisperEncoderCheckpointMapper.TensorMetadata("I32", new long[] {1280, 128, 3}));
        assertThrows(IllegalArgumentException.class, () -> mapper.mapMetadata(config, metadata));
    }
}
