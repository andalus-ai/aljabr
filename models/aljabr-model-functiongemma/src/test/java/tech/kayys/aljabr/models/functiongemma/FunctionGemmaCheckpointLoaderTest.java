package tech.kayys.aljabr.models.functiongemma;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelFamilyPluginRegistry;

import java.nio.file.Path;
import java.util.Objects;
import java.util.ServiceLoader;

import static org.junit.jupiter.api.Assertions.*;

class FunctionGemmaCheckpointLoaderTest {

    @Test
    void buildsRequiredTensorManifestFromCheckpointDecoderDepth() throws Exception {
        Path fixture = Path.of(Objects.requireNonNull(
                getClass().getResource("/model-family-fixtures/functiongemma/config.json")).toURI());
        ModelConfig config = ModelConfig.load(fixture, new ObjectMapper());

        var required = FunctionGemmaCheckpointLoader.requiredTensorNames(
                config, new FunctionGemmaArchitecture());

        assertEquals(18, config.numHiddenLayers());
        assertTrue(required.contains("model.embed_tokens.weight"));
        assertTrue(required.contains("model.norm.weight"));
        assertTrue(required.contains("model.layers.0.self_attn.q_proj.weight"));
        assertTrue(required.contains("model.layers.17.self_attn.q_proj.weight"));
        assertTrue(required.contains("model.layers.17.mlp.gate_proj.weight"));
        assertTrue(required.contains("model.layers.17.mlp.up_proj.weight"));
        assertTrue(required.contains("model.layers.17.mlp.down_proj.weight"));
        assertFalse(required.contains("model.layers.18.self_attn.q_proj.weight"));
    }

    @Test
    void resolvesTheCheckpointThroughTheServiceLoadedModelFamilyRegistry() throws Exception {
        Path fixture = Path.of(Objects.requireNonNull(
                getClass().getResource("/model-family-fixtures/functiongemma")).toURI());
        ModelFamilyPluginRegistry registry = ModelFamilyPluginRegistry.create();
        ServiceLoader.load(ModelFamilyPlugin.class).forEach(registry::register);

        assertEquals("functiongemma",
                registry.pluginForCheckpoint(fixture).orElseThrow().descriptor().id());
    }

    @Test
    void checkpointSessionOwnsTheSafetensorsResourceLifetime() {
        assertTrue(AutoCloseable.class.isAssignableFrom(
                FunctionGemmaCheckpointLoader.FunctionGemmaCheckpointSession.class));
    }
}
