package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2CheckpointMapper.CheckpointTensor;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2VisionConfig;

class EmbeddingGemma2VisionImagePipelineTest {
    @Test
    void composesNamedCheckpointWeightsProcessorVisionTowerAndTextBridge() {
        var spec = new EmbeddingGemma2VisionCheckpointMapper.VisionSpec(
                4, 4, 1, 1, 1, 4, 1, 32, false);
        var pipeline = EmbeddingGemma2VisionImagePipeline.fromCheckpoint(
                spec, tinyCheckpoint(), 3, 1e-6f, 1e-6f, 100d, 70, 2);

        var image = EmbeddingGemma2VisionImageProcessor.RgbImage.fromRgb8(
                4, 4, solidRgb(4, 4, (byte) 128));
        float[][] features = pipeline.encode(image);

        // 4x4 input is resized to a 16x16 patch grid with a 2x2 pooler.
        assertEquals(64, features.length);
        assertEquals(3, features[0].length);
        for (float[] token : features) {
            assertEquals(3, token.length);
            for (float value : token) assertEquals(0f, value, 0f);
        }
    }

    @Test
    void propagatesNonZeroPatchFeaturesThroughSpatialPoolerAndTextBridge() {
        var spec = new EmbeddingGemma2VisionCheckpointMapper.VisionSpec(
                4, 4, 1, 1, 1, 4, 1, 32, false);
        var pipeline = EmbeddingGemma2VisionImagePipeline.fromCheckpoint(
                spec, nonZeroCheckpoint(), 3, 1e-6f, 1e-6f, 100d, 70, 2);

        var whiteImage = EmbeddingGemma2VisionImageProcessor.RgbImage.fromRgb8(
                4, 4, solidRgb(4, 4, (byte) 255));
        float[][] features = pipeline.encode(whiteImage);

        assertEquals(64, features.length);
        // Pooling scales by sqrt(4); bridge RMSNorm spans all four vision channels,
        // of which three are non-zero: 2 / sqrt((3 * 2^2) / 4) = 2 / sqrt(3).
        float expected = (float) (2.0 / Math.sqrt(3.0));
        for (float[] token : features) {
            assertEquals(3, token.length);
            assertEquals(expected, token[0], 1e-5f);
            assertEquals(expected, token[1], 1e-5f);
            assertEquals(expected, token[2], 1e-5f);
        }
    }

    @Test
    void parsesHuggingFaceVisionAndProcessorConfigAndBuildsPipeline() throws Exception {
        var mapper = new ObjectMapper();
        var modelConfig = mapper.readTree("""
                {
                  "vision_config": {
                    "model_type": "gemma4_vision",
                    "hidden_size": 4,
                    "intermediate_size": 4,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "patch_size": 1,
                    "position_embedding_size": 32,
                    "pooling_kernel_size": 2,
                    "standardize": false,
                    "rms_norm_eps": 0.000001,
                    "rope_parameters": {"rope_type": "axial", "rope_theta": 100.0},
                    "hidden_activation": "gelu_pytorch_tanh",
                    "use_clipped_linears": false
                  }
                }
                """);
        var processorConfig = mapper.readTree("""
                {"patch_size": 1, "pooling_kernel_size": 2, "max_soft_tokens": 70}
                """);

        var config = EmbeddingGemma2VisionConfig.fromJson(modelConfig, processorConfig);
        assertEquals(1, config.patchSize());
        assertEquals(2, config.poolingKernelSize());
        assertEquals(70, config.maxSoftTokens());
        assertEquals(32, config.positionEmbeddingSize());

        var pipeline = EmbeddingGemma2VisionImagePipeline.fromCheckpoint(
                config, tinyCheckpoint(), 3, 1e-6f);
        var image = EmbeddingGemma2VisionImageProcessor.RgbImage.fromRgb8(
                4, 4, solidRgb(4, 4, (byte) 128));
        float[][] features = pipeline.encode(image);

        assertEquals(64, features.length);
        assertEquals(3, features[0].length);
    }

    @Test
    void retainsPerImageTokenBoundariesForDifferentAspectRatios() {
        var spec = new EmbeddingGemma2VisionCheckpointMapper.VisionSpec(
                4, 4, 1, 1, 1, 4, 1, 32, false);
        var pipeline = EmbeddingGemma2VisionImagePipeline.fromCheckpoint(
                spec, tinyCheckpoint(), 3, 1e-6f, 1e-6f, 100d, 70, 2);

        var square = EmbeddingGemma2VisionImageProcessor.RgbImage.fromRgb8(
                4, 4, solidRgb(4, 4, (byte) 128));
        var landscape = EmbeddingGemma2VisionImageProcessor.RgbImage.fromRgb8(
                8, 4, solidRgb(8, 4, (byte) 64));

        List<float[][]> result = pipeline.encodeBatch(List.of(square, landscape));

        assertEquals(2, result.size());
        assertEquals(64, result.get(0).length);
        assertEquals(55, result.get(1).length);
        assertEquals(3, result.get(0)[0].length);
        assertEquals(3, result.get(1)[0].length);
        assertTrue(result.get(0) != result.get(1));
    }

    private static Map<String, CheckpointTensor> tinyCheckpoint() {
        Map<String, CheckpointTensor> tensors = new HashMap<>();
        put(tensors, "vision_tower.patch_embedder.input_proj.weight", 4, 3);
        put(tensors, "vision_tower.patch_embedder.position_embedding_table", 2, 32, 4);

        String layer = "vision_tower.encoder.layers.0.";
        put(tensors, layer + "self_attn.q_proj.linear.weight", 4, 4);
        put(tensors, layer + "self_attn.k_proj.linear.weight", 4, 4);
        put(tensors, layer + "self_attn.v_proj.linear.weight", 4, 4);
        put(tensors, layer + "self_attn.o_proj.linear.weight", 4, 4);
        put(tensors, layer + "self_attn.q_norm.weight", 4);
        put(tensors, layer + "self_attn.k_norm.weight", 4);
        put(tensors, layer + "mlp.gate_proj.linear.weight", 4, 4);
        put(tensors, layer + "mlp.up_proj.linear.weight", 4, 4);
        put(tensors, layer + "mlp.down_proj.linear.weight", 4, 4);
        put(tensors, layer + "input_layernorm.weight", 4);
        put(tensors, layer + "post_attention_layernorm.weight", 4);
        put(tensors, layer + "pre_feedforward_layernorm.weight", 4);
        put(tensors, layer + "post_feedforward_layernorm.weight", 4);

        put(tensors, "embed_vision.embedding_projection.weight", 3, 4);
        return tensors;
    }

    private static Map<String, CheckpointTensor> nonZeroCheckpoint() {
        Map<String, CheckpointTensor> tensors = tinyCheckpoint();
        putValues(tensors, "vision_tower.patch_embedder.input_proj.weight",
                new int[]{4, 3},
                new float[]{1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f});
        putValues(tensors, "embed_vision.embedding_projection.weight",
                new int[]{3, 4},
                new float[]{1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f});
        return tensors;
    }

    private static void putValues(
            Map<String, CheckpointTensor> tensors, String name, int[] shape, float[] values) {
        tensors.put(name, new CheckpointTensor(shape, values));
    }

    private static void put(Map<String, CheckpointTensor> tensors, String name, int... shape) {
        int elements = 1;
        for (int dimension : shape) elements = Math.multiplyExact(elements, dimension);
        tensors.put(name, new CheckpointTensor(shape, new float[elements]));
    }

    private static byte[] solidRgb(int width, int height, byte value) {
        byte[] pixels = new byte[width * height * 3];
        java.util.Arrays.fill(pixels, value);
        return pixels;
    }
}
