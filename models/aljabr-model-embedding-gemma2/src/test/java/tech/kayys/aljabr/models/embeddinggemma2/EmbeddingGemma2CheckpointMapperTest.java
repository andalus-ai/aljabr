package tech.kayys.aljabr.models.embeddinggemma2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;

class EmbeddingGemma2CheckpointMapperTest {
    private static final ObjectMapper JSON = new ObjectMapper();

    @Test
    void mapsHuggingFaceTextWeightsIntoExecutableBackboneBundle() throws Exception {
        EmbeddingGemma2Config config = config();
        var weights = EmbeddingGemma2CheckpointMapper.map(config, tensors());
        var backbone = new EmbeddingGemma2TextBackbone(config, weights);

        assertEquals(16, weights.tokenEmbeddings().length);
        assertEquals(1, weights.layers().size());
        assertEquals(2, weights.layers().get(0).attention().query().length);
        assertEquals(2, weights.layers().get(0).encoder().pleGate().length);
        assertEquals(2, backbone.encode(new int[]{1, 2}).length);
        assertEquals(3, backbone.encodePooled(new int[]{1, 2}).length);
    }

    @Test
    void acceptsPreassembledInputEmbeddingsForMultimodalExecution() throws Exception {
        EmbeddingGemma2Config config = config();
        var backbone = new EmbeddingGemma2TextBackbone(
                config, EmbeddingGemma2CheckpointMapper.map(config, tensors()));

        float[][] embedded = backbone.embedInputTokens(new int[]{1, 2});
        float[][] output = backbone.encodeEmbeddings(
                embedded, new boolean[]{true, true}, new int[]{0, 1});

        assertEquals(2, embedded.length);
        assertEquals(config.hiddenSize(), embedded[0].length);
        assertEquals(2, output.length);
        assertEquals(config.embeddingDim(), output[0].length);
    }

    @Test
    void replacesOutOfVocabularyImagePlaceholderBeforeTextEmbeddingLookup() throws Exception {
        EmbeddingGemma2Config config = config();
        var backbone = new EmbeddingGemma2TextBackbone(
                config, EmbeddingGemma2CheckpointMapper.map(config, tensors()));

        float[][] merged = EmbeddingGemma2MultimodalInputAssembler.assemble(
                backbone,
                new int[]{1, 99, 2},
                99,
                0,
                java.util.Collections.singletonList(new float[][]{{7f, 8f}}));

        assertEquals(3, merged.length);
        org.junit.jupiter.api.Assertions.assertArrayEquals(new float[]{7f, 8f}, merged[1], 0f);
    }

    @Test
    void rejectsMissingRequiredTensor() throws Exception {
        Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> tensors = tensors();
        tensors.remove("language_model.norm.weight");
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2CheckpointMapper.map(config(), tensors));
    }

    @Test
    void rejectsWrongProjectionShape() throws Exception {
        Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> tensors = tensors();
        tensors.put("language_model.embedding_projection.weight", tensor(new int[]{2, 2}, 4));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2CheckpointMapper.map(config(), tensors));
    }

    @Test
    void rejectsUnknownTextTensorButIgnoresOtherModalities() throws Exception {
        Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> tensors = tensors();
        tensors.put("vision_tower.patch_embed.weight", tensor(new int[]{1}, 1));
        tensors.put("language_model.unmapped.weight", tensor(new int[]{1}, 1));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2CheckpointMapper.map(config(), tensors));
    }

    private static EmbeddingGemma2Config config() throws Exception {
        return new EmbeddingGemma2Config(
                EmbeddingGemma2Config.MODEL_TYPE,
                EmbeddingGemma2Config.TEXT_MODEL_TYPE,
                2, 2, 1, 1, 1, 2, 2,
                EmbeddingGemma2Config.DEFAULT_HIDDEN_ACTIVATION,
                3, 16, 16, 8192, "float32", 1.0e-6,
                null, List.of("full_attention"), Map.of(),
                JSON.readTree("""
                    {
                      "full_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                      "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
                    }
                    """));
    }

    private static Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> tensors() {
        Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> result = new HashMap<>();
        putMatrix(result, "language_model.embed_tokens.weight", 16, 2);
        putMatrix(result, "language_model.ple.per_layer_model_projection.weight", 2, 2);
        putVector(result, "language_model.ple.per_layer_projection_norm.weight", 2);
        String layer = "language_model.layers.0.";
        putMatrix(result, layer + "self_attn.q_proj.weight", 2, 2);
        putMatrix(result, layer + "self_attn.k_proj.weight", 2, 2);
        putMatrix(result, layer + "self_attn.v_proj.weight", 2, 2);
        putMatrix(result, layer + "self_attn.o_proj.weight", 2, 2);
        putVector(result, layer + "self_attn.q_norm.weight", 2);
        putVector(result, layer + "self_attn.k_norm.weight", 2);
        putMatrix(result, layer + "mlp.gate_proj.weight", 2, 2);
        putMatrix(result, layer + "mlp.up_proj.weight", 2, 2);
        putMatrix(result, layer + "mlp.down_proj.weight", 2, 2);
        putVector(result, layer + "input_layernorm.weight", 2);
        putVector(result, layer + "post_attention_layernorm.weight", 2);
        putVector(result, layer + "pre_feedforward_layernorm.weight", 2);
        putVector(result, layer + "post_feedforward_layernorm.weight", 2);
        putMatrix(result, layer + "ple_block.per_layer_input_gate.weight", 2, 2);
        putMatrix(result, layer + "ple_block.per_layer_projection.weight", 2, 2);
        putVector(result, layer + "ple_block.post_per_layer_input_norm.weight", 2);
        result.put(layer + "layer_scalar", tensor(new int[]{1}, 1));
        putVector(result, "language_model.norm.weight", 2);
        putMatrix(result, "language_model.embedding_projection.weight", 3, 2);
        return result;
    }

    private static void putMatrix(Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> tensors,
            String name, int rows, int columns) {
        tensors.put(name, tensor(new int[]{rows, columns}, rows * columns));
    }

    private static void putVector(Map<String, EmbeddingGemma2CheckpointMapper.CheckpointTensor> tensors,
            String name, int length) {
        tensors.put(name, tensor(new int[]{length}, length));
    }

    private static EmbeddingGemma2CheckpointMapper.CheckpointTensor tensor(int[] shape, int count) {
        float[] values = new float[count];
        java.util.Arrays.fill(values, 1f);
        return new EmbeddingGemma2CheckpointMapper.CheckpointTensor(shape, values);
    }
}
