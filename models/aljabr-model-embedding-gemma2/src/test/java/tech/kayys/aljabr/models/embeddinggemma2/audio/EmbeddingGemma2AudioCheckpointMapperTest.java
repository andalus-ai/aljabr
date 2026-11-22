package tech.kayys.aljabr.models.embeddinggemma2.audio;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2CheckpointMapper.CheckpointTensor;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2AudioConfig;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingGemma2AudioCheckpointMapperTest {
    private static EmbeddingGemma2AudioConfig config() {
        return new EmbeddingGemma2AudioConfig(
                "gemma4_audio", 2, 1, 1, 3, 1, List.of(2, 2),
                2, 3, 0, 1e-6, 0.5, "silu", "float32", true,
                16_000, 4, 8, 4, 2, 0.0, 8_000.0, 0.001,
                0.0, 1.0, 0.0, 280, 40, 1, 2, 3, 1.0e10, 50.0, -1.0e9);
    }

    private static Map<String, CheckpointTensor> checkpoint() {
        Map<String, CheckpointTensor> tensors = new HashMap<>();
        put(tensors, "audio_tower.subsample_conv_projection.layer0.conv.weight",
                new int[]{2, 1, 3, 3}, 18);
        put(tensors, "audio_tower.subsample_conv_projection.layer0.norm.weight", new int[]{2}, 2);
        put(tensors, "audio_tower.subsample_conv_projection.layer1.conv.weight",
                new int[]{2, 2, 3, 3}, 36);
        put(tensors, "audio_tower.subsample_conv_projection.layer1.norm.weight", new int[]{2}, 2);
        put(tensors, "audio_tower.subsample_conv_projection.input_proj_linear.weight",
                new int[]{2, 2}, 4);

        String layer = "audio_tower.layers.0.";
        feedForward(tensors, layer + "feed_forward1.");
        feedForward(tensors, layer + "feed_forward2.");
        for (String projection : List.of("q_proj", "k_proj", "v_proj", "post")) {
            String prefix = layer + "self_attn." + projection + ".";
            put(tensors, prefix + "linear.weight", new int[]{2, 2}, 4);
            clipBounds(tensors, prefix);
        }
        put(tensors, layer + "self_attn.relative_k_proj.weight", new int[]{2, 2}, 4);
        put(tensors, layer + "self_attn.per_dim_scale", new int[]{2}, 2);

        String start = layer + "lconv1d.linear_start.";
        put(tensors, start + "linear.weight", new int[]{4, 2}, 8);
        clipBounds(tensors, start);
        String end = layer + "lconv1d.linear_end.";
        put(tensors, end + "linear.weight", new int[]{2, 2}, 4);
        clipBounds(tensors, end);
        put(tensors, layer + "lconv1d.depthwise_conv1d.weight", new int[]{2, 1, 1}, 2);
        put(tensors, layer + "lconv1d.pre_layer_norm.weight", new int[]{2}, 2);
        put(tensors, layer + "lconv1d.conv_norm.weight", new int[]{2}, 2);
        put(tensors, layer + "norm_pre_attn.weight", new int[]{2}, 2);
        put(tensors, layer + "norm_post_attn.weight", new int[]{2}, 2);
        put(tensors, layer + "norm_out.weight", new int[]{2}, 2);

        put(tensors, "audio_tower.output_proj.weight", new int[]{3, 2}, 6);
        put(tensors, "audio_tower.output_proj.bias", new int[]{3}, 3);
        return tensors;
    }

    private static void feedForward(Map<String, CheckpointTensor> tensors, String prefix) {
        put(tensors, prefix + "ffw_layer_1.linear.weight", new int[]{8, 2}, 16);
        clipBounds(tensors, prefix + "ffw_layer_1.");
        put(tensors, prefix + "ffw_layer_2.linear.weight", new int[]{2, 8}, 16);
        clipBounds(tensors, prefix + "ffw_layer_2.");
        put(tensors, prefix + "pre_layer_norm.weight", new int[]{2}, 2);
        put(tensors, prefix + "post_layer_norm.weight", new int[]{2}, 2);
    }

    private static void clipBounds(Map<String, CheckpointTensor> tensors, String prefix) {
        put(tensors, prefix + "input_min", new int[]{1}, 1);
        put(tensors, prefix + "input_max", new int[]{1}, 1);
        put(tensors, prefix + "output_min", new int[]{1}, 1);
        put(tensors, prefix + "output_max", new int[]{1}, 1);
    }

    private static void put(
            Map<String, CheckpointTensor> tensors, String name, int[] shape, int count) {
        tensors.put(name, new CheckpointTensor(shape, new float[count]));
    }

    @Test
    void mapsCompleteAudioTowerTensorBundle() {
        var weights = EmbeddingGemma2AudioCheckpointMapper.map(config(), checkpoint());

        assertNotNull(weights.subsampling());
        assertEquals(1, weights.layers().size());
        assertEquals(3, weights.outputProjection().length);
        assertEquals(3, weights.outputProjectionBias().length);
        assertEquals(2, weights.layers().get(0).selfAttention().query().kernel().length);
        assertEquals(2, weights.layers().get(0).lightConvolution().depthwiseKernel().length);
    }

    @Test
    void rejectsMissingRequiredTensor() {
        Map<String, CheckpointTensor> tensors = checkpoint();
        tensors.remove("audio_tower.output_proj.bias");

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2AudioCheckpointMapper.map(config(), tensors));
        assertTrue(error.getMessage().contains("output_proj.bias"));
    }

    @Test
    void rejectsIncorrectTensorShape() {
        Map<String, CheckpointTensor> tensors = checkpoint();
        tensors.put("audio_tower.output_proj.weight",
                new CheckpointTensor(new int[]{2, 3}, new float[6]));

        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2AudioCheckpointMapper.map(config(), tensors));
    }

    @Test
    void rejectsUnmappedAudioNamespaceTensors() {
        Map<String, CheckpointTensor> tensors = checkpoint();
        tensors.put("audio_tower.unexpected.weight",
                new CheckpointTensor(new int[]{1}, new float[]{1f}));

        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2AudioCheckpointMapper.map(config(), tensors));
    }
}
