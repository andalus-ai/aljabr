package tech.kayys.aljabr.safetensor.loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2VisionConfig;
import tech.kayys.aljabr.models.embeddinggemma2.vision.EmbeddingGemma2VisionImagePipeline;
import tech.kayys.aljabr.models.embeddinggemma2.vision.EmbeddingGemma2VisionImageProcessor;
import tech.kayys.aljabr.safetensor.embeddinggemma2.EmbeddingGemma2VisionSafetensorLoader;

/** Exercises the vision adapter against a real Safetensors session and detached checkpoint tensors. */
class EmbeddingGemma2VisionSafetensorPipelineTest {

    @Test
    void mapsOnlyVisionAndBridgeTensorsAndBuildsPipelineAfterSessionCloses() {
        Map<String, int[]> shapes = tinyCheckpointShapes();
        shapes.put("language_model.embed_tokens.weight", new int[] {2, 4});

        Map<String, SafetensorTensorInfo> infos = new LinkedHashMap<>();
        long offset = 0;
        for (Map.Entry<String, int[]> entry : shapes.entrySet()) {
            long elements = 1;
            for (int dimension : entry.getValue()) {
                elements = Math.multiplyExact(elements, dimension);
            }
            long byteLength = Math.multiplyExact(elements, (long) Float.BYTES);
            infos.put(entry.getKey(), SafetensorTensorInfo.of(
                    entry.getKey(), SafetensorDType.F32, toLongs(entry.getValue()), offset, byteLength));
            offset = Math.addExact(offset, byteLength);
        }

        Arena arena = Arena.ofShared();
        MemorySegment payload = arena.allocate(offset);
        var header = SafetensorHeader.of(0, infos, Map.of());
        var result = new SafetensorLoadResult(
                Path.of("embedding-gemma2-vision-fixture.safetensors"),
                header, payload, arena, SafetensorLoadResult.LoadMode.COPY);
        var session = SafetensorShardLoader.SafetensorShardSession.singleFile(result);

        Map<String, tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2CheckpointMapper.CheckpointTensor>
                checkpoint;
        try (session) {
            checkpoint = new EmbeddingGemma2VisionSafetensorLoader(new SafetensorShardLoader())
                    .mapTensors(session, SafetensorDType.F32);
            assertEquals(shapes.size() - 1, checkpoint.size());
            assertFalse(checkpoint.containsKey("language_model.embed_tokens.weight"));
        }

        var config = new EmbeddingGemma2VisionConfig(
                EmbeddingGemma2VisionConfig.MODEL_TYPE,
                4, 4, 1, 1, 1, 4, 1, 32, 2, 70,
                false, 1e-6, 100d, "gelu_pytorch_tanh", false);
        EmbeddingGemma2VisionImagePipeline pipeline =
                EmbeddingGemma2VisionImagePipeline.fromCheckpoint(config, checkpoint, 3, 1e-6f);

        byte[] pixels = new byte[4 * 4 * 3];
        java.util.Arrays.fill(pixels, (byte) 128);
        var image = EmbeddingGemma2VisionImageProcessor.RgbImage.fromRgb8(4, 4, pixels);
        float[][] features = pipeline.encode(image);

        assertEquals(64, features.length);
        assertEquals(3, features[0].length);
    }

    private static Map<String, int[]> tinyCheckpointShapes() {
        Map<String, int[]> shapes = new LinkedHashMap<>();
        shapes.put("vision_tower.patch_embedder.input_proj.weight", new int[] {4, 3});
        shapes.put("vision_tower.patch_embedder.position_embedding_table", new int[] {2, 32, 4});

        String layer = "vision_tower.encoder.layers.0.";
        shapes.put(layer + "self_attn.q_proj.linear.weight", new int[] {4, 4});
        shapes.put(layer + "self_attn.k_proj.linear.weight", new int[] {4, 4});
        shapes.put(layer + "self_attn.v_proj.linear.weight", new int[] {4, 4});
        shapes.put(layer + "self_attn.o_proj.linear.weight", new int[] {4, 4});
        shapes.put(layer + "self_attn.q_norm.weight", new int[] {4});
        shapes.put(layer + "self_attn.k_norm.weight", new int[] {4});
        shapes.put(layer + "mlp.gate_proj.linear.weight", new int[] {4, 4});
        shapes.put(layer + "mlp.up_proj.linear.weight", new int[] {4, 4});
        shapes.put(layer + "mlp.down_proj.linear.weight", new int[] {4, 4});
        shapes.put(layer + "input_layernorm.weight", new int[] {4});
        shapes.put(layer + "post_attention_layernorm.weight", new int[] {4});
        shapes.put(layer + "pre_feedforward_layernorm.weight", new int[] {4});
        shapes.put(layer + "post_feedforward_layernorm.weight", new int[] {4});
        shapes.put("embed_vision.embedding_projection.weight", new int[] {3, 4});
        return shapes;
    }

    private static long[] toLongs(int[] shape) {
        long[] result = new long[shape.length];
        for (int i = 0; i < shape.length; i++) result[i] = shape[i];
        return result;
    }
}
