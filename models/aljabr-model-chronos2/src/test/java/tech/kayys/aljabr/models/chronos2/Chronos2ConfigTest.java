package tech.kayys.aljabr.models.chronos2;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class Chronos2ConfigTest {

    @Test
    void parsesUpstreamAutogluonCheckpointConfiguration() {
        Chronos2Config config = Chronos2Config.fromHuggingFaceConfig(Map.of(
                "architectures", List.of("Chronos2Model"),
                "model_type", "t5",
                "d_model", 768,
                "d_kv", 64,
                "d_ff", 3072,
                "num_layers", 12,
                "num_heads", 12,
                "layer_norm_epsilon", 1e-6,
                "rope_theta", 10000.0,
                "chronos_config", Map.of(
                        "context_length", 8192,
                        "input_patch_size", 16,
                        "input_patch_stride", 16,
                        "output_patch_size", 16,
                        "max_output_patches", 64,
                        "time_encoding_scale", 8192,
                        "use_reg_token", true,
                        "use_arcsinh", true,
                        "quantiles", List.of(0.01, 0.1, 0.5, 0.9, 0.99))));

        assertEquals(768, config.modelDimension());
        assertEquals(12, config.layerCount());
        assertEquals(8192, config.contextLength());
        assertEquals(1024, config.maximumPredictionLength());
        assertEquals(List.of(0.01, 0.1, 0.5, 0.9, 0.99), config.quantiles());
    }

    @Test
    void rejectsNonChronosArchitecture() {
        assertThrows(IllegalArgumentException.class, () -> Chronos2Config.fromHuggingFaceConfig(Map.of(
                "architectures", List.of("T5EncoderModel"),
                "model_type", "t5",
                "chronos_config", Map.of())));
    }

    @Test
    void rejectsUnorderedQuantiles() {
        assertThrows(IllegalArgumentException.class, () -> new Chronos2Config(
                8192, 16, 16, 16, 64, 8192, 768, 64, 3072, 12, 12,
                1e-6, 10000.0, true, true, List.of(0.9, 0.5)));
    }
}
