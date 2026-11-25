package tech.kayys.aljabr.models.bert;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class BertCheckpointMapperTest {
    private static BertConfig config() {
        return new BertConfig(32, 8, 2, 2, 16, 16, 2, 0, 1.0e-12, "gelu");
    }

    private static Map<String, BertCheckpointMapper.CheckpointTensor> completeCheckpoint(
            BertCheckpointMapper mapper, BertConfig config) {
        Map<String, BertCheckpointMapper.CheckpointTensor> tensors = new LinkedHashMap<>();
        mapper.expectedTensorShapes(config).forEach((name, shape) -> {
            int count = 1;
            for (int dimension : shape) count = Math.multiplyExact(count, dimension);
            tensors.put(name, new BertCheckpointMapper.CheckpointTensor(shape, new float[count]));
        });
        return tensors;
    }

    @Test
    void mapsAllBertEncoderParametersToReusableModuleNames() {
        BertCheckpointMapper mapper = new BertCheckpointMapper();
        Map<String, BertCheckpointMapper.CheckpointTensor> mapped =
                mapper.map(config(), completeCheckpoint(mapper, config()));
        assertEquals(mapper.parameterBindings(config()).size(), mapped.size());
        assertEquals(true, mapped.containsKey("encoder_layer_1.self_attn.query.weight"));
        assertEquals(true, mapped.containsKey("embeddings.layer_norm.bias"));
    }

    @Test
    void ignoresUnusedPoolerAndPersistentIndexBuffers() {
        BertCheckpointMapper mapper = new BertCheckpointMapper();
        Map<String, BertCheckpointMapper.CheckpointTensor> tensors = completeCheckpoint(mapper, config());
        tensors.put("pooler.dense.weight", new BertCheckpointMapper.CheckpointTensor(
                new int[] {8, 8}, new float[64]));
        tensors.put("pooler.dense.bias", new BertCheckpointMapper.CheckpointTensor(
                new int[] {8}, new float[8]));
        tensors.put("embeddings.position_ids", new BertCheckpointMapper.CheckpointTensor(
                new int[] {1, 16}, new float[16]));
        assertEquals(mapper.parameterBindings(config()).size(), mapper.map(config(), tensors).size());
    }

    @Test
    void rejectsMissingUnexpectedAndMismatchedTensors() {
        BertCheckpointMapper mapper = new BertCheckpointMapper();
        Map<String, BertCheckpointMapper.CheckpointTensor> missing = completeCheckpoint(mapper, config());
        missing.remove("embeddings.word_embeddings.weight");
        assertThrows(IllegalArgumentException.class, () -> mapper.map(config(), missing));

        Map<String, BertCheckpointMapper.CheckpointTensor> unexpected = completeCheckpoint(mapper, config());
        unexpected.put("unexpected.weight", new BertCheckpointMapper.CheckpointTensor(
                new int[] {1}, new float[] {0}));
        assertThrows(IllegalArgumentException.class, () -> mapper.map(config(), unexpected));

        Map<String, BertCheckpointMapper.CheckpointTensor> wrongShape = completeCheckpoint(mapper, config());
        wrongShape.put("embeddings.word_embeddings.weight", new BertCheckpointMapper.CheckpointTensor(
                new int[] {31, 8}, new float[31 * 8]));
        assertThrows(IllegalArgumentException.class, () -> mapper.map(config(), wrongShape));
    }
}
