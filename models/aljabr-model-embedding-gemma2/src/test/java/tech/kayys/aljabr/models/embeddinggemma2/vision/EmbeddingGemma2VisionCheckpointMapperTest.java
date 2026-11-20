package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2CheckpointMapper.CheckpointTensor;

class EmbeddingGemma2VisionCheckpointMapperTest {
    private static final String PREFIX = "vision_tower.";

    @Test
    void loadsAndAssemblesExecutableVisionTowerFromCheckpointTensors() {
        var config = new tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2VisionConfig(
                tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2VisionConfig.MODEL_TYPE,
                4, 6, 1, 2, 2, 2, 1, 3, 1, 70, false,
                1.0e-6, 100.0,
                tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2VisionConfig.DEFAULT_HIDDEN_ACTIVATION,
                false);

        var model = EmbeddingGemma2VisionCheckpointMapper.load(config, tensors(false));

        assertNotNull(model.patchEmbedder());
        assertNotNull(model.encoder());
        assertNotNull(model.pooler());
        assertNull(model.standardizer());
        assertEquals(1, model.encoder().layerCount());

        float[][] encoded = model.encoder().encodePatches(
                new float[][]{{0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f},
                        {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}},
                new int[][]{{0, 0}, {1, 0}, {0, 1}, {1, 1}},
                new boolean[]{true, true, true, true});
        assertEquals(4, encoded.length);
        assertEquals(4, encoded[0].length);
    }

    @Test
    void assemblesOptionalStandardizerOnlyWhenConfigured() {
        var config = new tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2VisionConfig(
                tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2VisionConfig.MODEL_TYPE,
                4, 6, 1, 2, 2, 2, 1, 3, 1, 70, true,
                1.0e-6, 100.0,
                tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2VisionConfig.DEFAULT_HIDDEN_ACTIVATION,
                false);

        var model = EmbeddingGemma2VisionCheckpointMapper.load(config, tensors(true));

        assertNotNull(model.standardizer());
        assertEquals(4, model.standardizer().hiddenSize());
    }

    @Test
    void mapsHuggingFaceVisionTowerWeightsIntoTypedBundle() {
        var mapped = EmbeddingGemma2VisionCheckpointMapper.map(spec(true), tensors(true));

        assertEquals(4, mapped.patchEmbedder().inputProjection().length);
        assertEquals(3, mapped.patchEmbedder().inputProjection()[0].length);
        assertEquals(2, mapped.patchEmbedder().positionEmbeddingTable().length);
        assertEquals(1, mapped.layers().size());
        assertEquals(4, mapped.layers().get(0).attention().query().length);
        assertEquals(6, mapped.layers().get(0).mlp().gate().length);
        assertEquals(4, mapped.layers().get(0).norms().inputNormScale().length);
        assertEquals(4, mapped.legacyPoolerScale().length);
        assertEquals(4, mapped.standardizationBias().length);
        assertEquals(4, mapped.standardizationScale().length);
    }

    @Test
    void omitsStandardizationWeightsWhenDisabled() {
        var mapped = EmbeddingGemma2VisionCheckpointMapper.map(spec(false), tensors(false));
        assertNull(mapped.standardizationBias());
        assertNull(mapped.standardizationScale());
    }

    @Test
    void acceptsCheckpointsWithoutLegacyPoolerScale() {
        Map<String, CheckpointTensor> tensors = tensors(false);
        tensors.remove(PREFIX + "pooler.scale");
        var mapped = EmbeddingGemma2VisionCheckpointMapper.map(spec(false), tensors);
        assertNull(mapped.legacyPoolerScale());
    }

    @Test
    void rejectsMissingVisionTensor() {
        Map<String, CheckpointTensor> tensors = tensors(false);
        tensors.remove(PREFIX + "encoder.layers.0.self_attn.q_proj.linear.weight");
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2VisionCheckpointMapper.map(spec(false), tensors));
    }

    @Test
    void rejectsWrongVisionTensorShape() {
        Map<String, CheckpointTensor> tensors = tensors(false);
        tensors.put(PREFIX + "patch_embedder.input_proj.weight", tensor(new int[]{4, 4}, 16));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2VisionCheckpointMapper.map(spec(false), tensors));
    }

    @Test
    void rejectsUnknownVisionTensorInsteadOfSilentlyIgnoringIt() {
        Map<String, CheckpointTensor> tensors = tensors(false);
        tensors.put(PREFIX + "encoder.layers.0.unmapped.weight", tensor(new int[]{1}, 1));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2VisionCheckpointMapper.map(spec(false), tensors));
    }

    @Test
    void requiresBothStandardizationVectorsWhenEnabled() {
        Map<String, CheckpointTensor> tensors = tensors(true);
        tensors.remove(PREFIX + "std_scale");
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2VisionCheckpointMapper.map(spec(true), tensors));
    }

    @Test
    void validatesHeadGeometry() {
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingGemma2VisionCheckpointMapper.VisionSpec(
                        4, 6, 1, 2, 1, 2, 1, 3, false));
    }

    private static EmbeddingGemma2VisionCheckpointMapper.VisionSpec spec(boolean standardize) {
        return new EmbeddingGemma2VisionCheckpointMapper.VisionSpec(
                4, 6, 1, 2, 2, 2, 1, 3, standardize);
    }

    private static Map<String, CheckpointTensor> tensors(boolean standardize) {
        Map<String, CheckpointTensor> values = new HashMap<>();
        put(values, "patch_embedder.input_proj.weight", 4, 3);
        put(values, "patch_embedder.position_embedding_table", 2, 3, 4);
        String layer = "encoder.layers.0.";
        put(values, layer + "self_attn.q_proj.linear.weight", 4, 4);
        put(values, layer + "self_attn.k_proj.linear.weight", 4, 4);
        put(values, layer + "self_attn.v_proj.linear.weight", 4, 4);
        put(values, layer + "self_attn.o_proj.linear.weight", 4, 4);
        put(values, layer + "self_attn.q_norm.weight", 2);
        put(values, layer + "self_attn.k_norm.weight", 2);
        put(values, layer + "mlp.gate_proj.linear.weight", 6, 4);
        put(values, layer + "mlp.up_proj.linear.weight", 6, 4);
        put(values, layer + "mlp.down_proj.linear.weight", 4, 6);
        put(values, layer + "input_layernorm.weight", 4);
        put(values, layer + "post_attention_layernorm.weight", 4);
        put(values, layer + "pre_feedforward_layernorm.weight", 4);
        put(values, layer + "post_feedforward_layernorm.weight", 4);
        put(values, "pooler.scale", 1, 1, 4);
        if (standardize) {
            put(values, "std_bias", 4);
            put(values, "std_scale", 4);
        }
        return values;
    }

    private static void put(Map<String, CheckpointTensor> target, String name, int... shape) {
        target.put(PREFIX + name, tensor(shape, elementCount(shape)));
    }

    private static CheckpointTensor tensor(int[] shape, int count) {
        float[] values = new float[count];
        for (int i = 0; i < values.length; i++) values[i] = i + 1;
        return new CheckpointTensor(shape, values);
    }

    private static int elementCount(int[] shape) {
        int count = 1;
        for (int dimension : shape) count = Math.multiplyExact(count, dimension);
        return count;
    }
}
