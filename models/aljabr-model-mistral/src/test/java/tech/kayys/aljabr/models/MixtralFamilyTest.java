package tech.kayys.tafkir.models;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ModelArchitecture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MixtralFamilyTest {

    @Test
    void exposesMoeWeightsThroughArchitectureSpi() {
        ModelArchitecture arch = new MixtralFamily();

        assertNull(arch.layerFfnGateWeight(2));
        assertNull(arch.layerFfnUpWeight(2));
        assertNull(arch.layerFfnDownWeight(2));
        assertEquals("model.layers.2.block_sparse_moe.gate.weight", arch.layerMoeGateWeight(2));
        assertEquals("model.layers.2.block_sparse_moe.experts.5.w1.weight", arch.expertGateWeight(2, 5));
        assertEquals("model.layers.2.block_sparse_moe.experts.5.w3.weight", arch.expertUpWeight(2, 5));
        assertEquals("model.layers.2.block_sparse_moe.experts.5.w2.weight", arch.expertDownWeight(2, 5));
    }

    @Test
    void mistralPluginKeepsMixtralDetachedFromCoreBundle() {
        MistralModelFamilyPlugin plugin = new MistralModelFamilyPlugin();

        assertTrue(plugin.architectureAdapters().stream()
                .noneMatch(adapter -> adapter instanceof MixtralFamily));
        assertTrue(plugin.descriptor().modelTypes().contains("mistral"));
        assertTrue(plugin.descriptor().modelTypes().stream()
                .noneMatch(type -> type.equals("mixtral")));
    }
}
