package tech.kayys.aljabr.models.chronos2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class Chronos2CheckpointMapperTest {
    private Chronos2Config config() {
        return new Chronos2Config(
                8, 4, 4, 4, 4, 8,
                8, 2, 16, 2, 2, 1e-6, 10000.0,
                true, true, List.of(0.1, 0.5, 0.9));
    }

    @Test
    void describesChronos2CheckpointTensorNamesAndShapes() {
        var specs = new Chronos2CheckpointMapper().expectedTensors(config());
        assertEquals(2L, specs.get("shared.weight").shape()[0]);
        assertEquals(12L, specs.get("input_patch_embedding.hidden_layer.weight").shape()[1]);
        assertEquals(12L, specs.get("output_patch_embedding.output_layer.weight").shape()[0]);
        assertTrue(specs.containsKey("encoder.block.1.layer.1.self_attention.q.weight"));
        assertTrue(specs.containsKey("encoder.final_layer_norm.weight"));
    }

    @Test
    void validatesCompleteCheckpointAndReturnsIdentityBindings() {
        var mapper = new Chronos2CheckpointMapper();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        mapper.expectedTensors(config()).forEach((name, spec) -> shapes.put(name, spec.shape()));

        var mapping = mapper.map(config(), shapes);
        assertEquals(shapes.size(), mapping.parameterBindings().size());
        assertEquals("encoder.block.0.layer.0.self_attention.q.weight",
                mapping.parameterBindings().get("encoder.block.0.layer.0.self_attention.q.weight"));
    }

    @Test
    void rejectsMissingOrMismatchedCheckpointTensors() {
        var mapper = new Chronos2CheckpointMapper();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        mapper.expectedTensors(config()).forEach((name, spec) -> shapes.put(name, spec.shape()));
        shapes.remove("encoder.final_layer_norm.weight");
        assertThrows(IllegalArgumentException.class, () -> mapper.map(config(), shapes));

        shapes.put("encoder.final_layer_norm.weight", new long[] {7});
        assertThrows(IllegalArgumentException.class, () -> mapper.map(config(), shapes));
    }

    @Test
    void validatesSafetensorsDtypeMetadata() {
        var mapper = new Chronos2CheckpointMapper();
        Map<String, Chronos2CheckpointMapper.TensorMetadata> metadata = new LinkedHashMap<>();
        mapper.expectedTensors(config()).forEach((name, spec) ->
                metadata.put(name, new Chronos2CheckpointMapper.TensorMetadata("F32", spec.shape())));
        assertEquals(metadata.size(), mapper.mapMetadata(config(), metadata).parameterBindings().size());

        metadata.put("shared.weight",
                new Chronos2CheckpointMapper.TensorMetadata("F16", new long[] {2, 8}));
        assertThrows(IllegalArgumentException.class, () -> mapper.mapMetadata(config(), metadata));
    }

    @Test
    void tensorShapesAreDefensivelyCopied() {
        var spec = new Chronos2CheckpointMapper().expectedTensors(config()).get("shared.weight");
        long[] shape = spec.shape();
        shape[0] = 99;
        assertEquals(2L, spec.shape()[0]);
    }
}
