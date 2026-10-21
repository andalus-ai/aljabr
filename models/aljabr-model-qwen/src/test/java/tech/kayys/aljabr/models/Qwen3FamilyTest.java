package tech.kayys.aljabr.models;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ModelArchitecture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Qwen3FamilyTest {

    @Test
    void exposesQkNormWeightsThroughArchitectureSpi() {
        ModelArchitecture architecture = new Qwen3Family();

        assertEquals("model.layers.7.self_attn.q_norm.weight", architecture.layerQueryNormWeight(7));
        assertEquals("model.layers.7.self_attn.k_norm.weight", architecture.layerKeyNormWeight(7));
    }

    @Test
    void keepsLegacyQkNormHelpersDelegatingToSpiNames() {
        Qwen3Family architecture = new Qwen3Family();

        assertEquals(architecture.layerQueryNormWeight(3), architecture.layerQNorm(3));
        assertEquals(architecture.layerKeyNormWeight(3), architecture.layerKNorm(3));
    }

    @Test
    void qwenPluginPublishesQwen3AdapterWithQkNormSupport() {
        QwenModelFamilyPlugin plugin = new QwenModelFamilyPlugin();

        ModelArchitecture qwen3 = plugin.architectureAdapters().stream()
                .filter(adapter -> adapter.id().equals("qwen3"))
                .findFirst()
                .orElseThrow();

        assertTrue(qwen3.supportedArchClassNames().contains("Qwen3ForCausalLM"));
        assertEquals("model.layers.0.self_attn.q_norm.weight", qwen3.layerQueryNormWeight(0));
        assertEquals("model.layers.0.self_attn.k_norm.weight", qwen3.layerKeyNormWeight(0));
    }
}
