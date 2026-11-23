package tech.kayys.aljabr.safetensor.loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;
import tech.kayys.aljabr.safetensor.embeddinggemma2.EmbeddingGemma2AudioSafetensorLoader;

/** Exercises the audio Safetensors adapter with a synthetic mapped checkpoint session. */
class EmbeddingGemma2AudioSafetensorLoaderTest {

    @Test
    void materializesOnlyAudioNamespaceAndMapsDetachedWeights() {
        Map<String, int[]> shapes = tinyAudioCheckpointShapes();
        shapes.put("language_model.embed_tokens.weight", new int[] {2, 2});
        var session = session(shapes, SafetensorDType.F32);

        tech.kayys.aljabr.models.embeddinggemma2.audio.EmbeddingGemma2AudioEncoder.Weights weights;
        try (session) {
            weights = new EmbeddingGemma2AudioSafetensorLoader(new SafetensorShardLoader())
                    .mapWeights(session, config("float32"));
        }

        assertEquals(1, weights.layers().size());
        assertEquals(3, weights.outputProjection().length);
        assertEquals(2, weights.layers().get(0).selfAttention().query().kernel().length);
    }

    @Test
    void rejectsCheckpointDtypeThatConflictsWithAudioConfig() {
        var session = session(tinyAudioCheckpointShapes(), SafetensorDType.F32);
        try (session) {
            assertThrows(IllegalArgumentException.class,
                    () -> new EmbeddingGemma2AudioSafetensorLoader(new SafetensorShardLoader())
                            .mapWeights(session, config("bfloat16")));
        }
    }

    private static EmbeddingGemma2AudioConfig config(String dtype) {
        return new EmbeddingGemma2AudioConfig(
                "gemma4_audio", 2, 1, 1, 3, 1, List.of(2, 2),
                2, 3, 0, 1e-6, 0.5, "silu", dtype, true,
                16_000, 4, 8, 4, 2, 0.0, 8_000.0, 0.001,
                0.0, 1.0, 0.0, 280, 40, 1, 2, 3, 1.0e10, 50.0, -1.0e9);
    }

    private static Map<String, int[]> tinyAudioCheckpointShapes() {
        Map<String, int[]> shapes = new LinkedHashMap<>();
        shapes.put("audio_tower.subsample_conv_projection.layer0.conv.weight", new int[] {2, 1, 3, 3});
        shapes.put("audio_tower.subsample_conv_projection.layer0.norm.weight", new int[] {2});
        shapes.put("audio_tower.subsample_conv_projection.layer1.conv.weight", new int[] {2, 2, 3, 3});
        shapes.put("audio_tower.subsample_conv_projection.layer1.norm.weight", new int[] {2});
        shapes.put("audio_tower.subsample_conv_projection.input_proj_linear.weight", new int[] {2, 2});

        String layer = "audio_tower.layers.0.";
        feedForward(shapes, layer + "feed_forward1.");
        feedForward(shapes, layer + "feed_forward2.");
        for (String projection : List.of("q_proj", "k_proj", "v_proj", "post")) {
            String prefix = layer + "self_attn." + projection + ".";
            shapes.put(prefix + "linear.weight", new int[] {2, 2});
            clipping(shapes, prefix);
        }
        shapes.put(layer + "self_attn.relative_k_proj.weight", new int[] {2, 2});
        shapes.put(layer + "self_attn.per_dim_scale", new int[] {2});

        String start = layer + "lconv1d.linear_start.";
        shapes.put(start + "linear.weight", new int[] {4, 2});
        clipping(shapes, start);
        String end = layer + "lconv1d.linear_end.";
        shapes.put(end + "linear.weight", new int[] {2, 2});
        clipping(shapes, end);
        shapes.put(layer + "lconv1d.depthwise_conv1d.weight", new int[] {2, 1, 1});
        shapes.put(layer + "lconv1d.pre_layer_norm.weight", new int[] {2});
        shapes.put(layer + "lconv1d.conv_norm.weight", new int[] {2});
        shapes.put(layer + "norm_pre_attn.weight", new int[] {2});
        shapes.put(layer + "norm_post_attn.weight", new int[] {2});
        shapes.put(layer + "norm_out.weight", new int[] {2});
        shapes.put("audio_tower.output_proj.weight", new int[] {3, 2});
        shapes.put("audio_tower.output_proj.bias", new int[] {3});
        return shapes;
    }

    private static void feedForward(Map<String, int[]> shapes, String prefix) {
        shapes.put(prefix + "ffw_layer_1.linear.weight", new int[] {8, 2});
        clipping(shapes, prefix + "ffw_layer_1.");
        shapes.put(prefix + "ffw_layer_2.linear.weight", new int[] {2, 8});
        clipping(shapes, prefix + "ffw_layer_2.");
        shapes.put(prefix + "pre_layer_norm.weight", new int[] {2});
        shapes.put(prefix + "post_layer_norm.weight", new int[] {2});
    }

    private static void clipping(Map<String, int[]> shapes, String prefix) {
        shapes.put(prefix + "input_min", new int[] {1});
        shapes.put(prefix + "input_max", new int[] {1});
        shapes.put(prefix + "output_min", new int[] {1});
        shapes.put(prefix + "output_max", new int[] {1});
    }

    private static SafetensorShardLoader.SafetensorShardSession session(
            Map<String, int[]> shapes, SafetensorDType dtype) {
        Map<String, SafetensorTensorInfo> infos = new LinkedHashMap<>();
        long offset = 0;
        for (Map.Entry<String, int[]> entry : shapes.entrySet()) {
            long elements = 1;
            for (int dimension : entry.getValue()) elements = Math.multiplyExact(elements, dimension);
            long bytes = Math.multiplyExact(elements, (long) dtype.byteSize());
            infos.put(entry.getKey(), SafetensorTensorInfo.of(
                    entry.getKey(), dtype, toLongs(entry.getValue()), offset, offset + bytes));
            offset += bytes;
        }
        Arena arena = Arena.ofShared();
        MemorySegment payload = arena.allocate(offset);
        var header = SafetensorHeader.of(0, infos, Map.of());
        var result = new SafetensorLoadResult(
                Path.of("embedding-gemma2-audio-fixture.safetensors"),
                header, payload, arena, SafetensorLoadResult.LoadMode.COPY);
        return SafetensorShardLoader.SafetensorShardSession.singleFile(result);
    }

    private static long[] toLongs(int[] shape) {
        long[] result = new long[shape.length];
        for (int i = 0; i < shape.length; i++) result[i] = shape[i];
        return result;
    }
}
