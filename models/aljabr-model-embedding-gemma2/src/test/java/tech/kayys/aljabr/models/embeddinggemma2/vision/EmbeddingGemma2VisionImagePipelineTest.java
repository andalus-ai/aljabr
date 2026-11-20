package tech.kayys.aljabr.models.embeddinggemma2.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2CheckpointMapper.CheckpointTensor;

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
