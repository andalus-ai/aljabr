package tech.kayys.aljabr.spi.model.ir;

import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class ArchitectureDescriptorTest {

    @Test
    void architectureFamilySupportsKnownAndVendorDefinedIdentifiers() {
        var known = ArchitectureFamilies.DECODER_ONLY_TRANSFORMER;
        var custom = ArchitectureFamily.custom("vendor.experimental.sequence-model");

        assertEquals(ArchitectureKind.DECODER_ONLY_TRANSFORMER, known.kind());
        assertEquals("vendor.experimental.sequence-model", custom.id());
        assertEquals(ArchitectureKind.CUSTOM, custom.kind());
    }

    @Test
    void configurationPreservesTypedValuesAndOpaqueExtensions() {
        var properties = new LinkedHashMap<String, ConfigurationValue>();
        properties.put("hidden_size", ConfigurationValue.of(4096L));
        properties.put("rms_norm_eps", ConfigurationValue.of(new BigDecimal("0.000001")));
        properties.put("tie_word_embeddings", ConfigurationValue.of(false));
        properties.put("model_type", ConfigurationValue.of("gemma"));

        var configuration = new ArchitectureConfiguration(
                properties, Map.of("vendor.rotary_mode", "interleaved"));
        properties.clear();

        assertEquals(4096L, configuration.integerValue("hidden_size").orElseThrow());
        assertEquals(false, configuration.booleanValue("tie_word_embeddings").orElseThrow());
        assertEquals("gemma", configuration.stringValue("model_type").orElseThrow());
        assertInstanceOf(ConfigurationValue.DecimalValue.class,
                configuration.find("rms_norm_eps").orElseThrow());
        assertEquals("interleaved", configuration.extensions().get("vendor.rotary_mode"));
        assertThrows(UnsupportedOperationException.class,
                () -> configuration.properties().clear());
        assertThrows(UnsupportedOperationException.class,
                () -> configuration.extensions().put("x", "y"));
    }

    @Test
    void configurationSupportsImmutableTypedLists() {
        var input = new java.util.ArrayList<>(List.of("text", "image"));
        var value = new ConfigurationValue.StringListValue(input);
        input.add("audio");

        assertEquals(List.of("text", "image"), value.value());
        assertThrows(UnsupportedOperationException.class,
                () -> value.value().add("audio"));
    }

    @Test
    void architectureDescriptorIsImmutableAndKeepsUnknownCapabilities() {
        var capabilities = new java.util.HashSet<>(Set.of("causal-language-model", "vendor.speculative"));
        var extensions = new LinkedHashMap<>(Map.of("vendor.config_version", "3"));

        var descriptor = new ArchitectureDescriptor(
                "vendor.gemma-compatible",
                ArchitectureFamilies.DECODER_ONLY_TRANSFORMER,
                ArchitectureConfiguration.empty(),
                capabilities,
                extensions);

        capabilities.add("mutated");
        extensions.put("mutated", "true");

        assertEquals(Set.of("causal-language-model", "vendor.speculative"), descriptor.capabilities());
        assertFalse(descriptor.extensions().containsKey("mutated"));
        assertThrows(UnsupportedOperationException.class,
                () -> descriptor.capabilities().add("another"));
    }

    @Test
    void modelDescriptorCanBeEnrichedWithoutBreakingLegacyConstructor() {
        var base = new ModelDescriptor(
                ModelIdentity.of("model-1"), "gemma", "safetensors",
                Map.of(), Map.of(), List.of());

        assertTrue(base.architectureDescriptor().isEmpty());

        var architecture = new ArchitectureDescriptor(
                "gemma",
                ArchitectureFamilies.DECODER_ONLY_TRANSFORMER,
                new ArchitectureConfiguration(Map.of(
                        "hidden_size", ConfigurationValue.of(3072L)), Map.of()),
                Set.of("causal-language-model"),
                Map.of());

        var enriched = base.withArchitectureDescriptor(architecture);

        assertTrue(enriched.architectureDescriptor().isPresent());
        assertEquals(3072L, enriched.architectureDescriptor().orElseThrow()
                .configuration().integerValue("hidden_size").orElseThrow());
        assertTrue(base.architectureDescriptor().isEmpty());
        assertEquals("gemma", enriched.architecture());
    }

    @Test
    void rejectsArchitectureDescriptorWithMismatchedIdentifier() {
        var architecture = ArchitectureDescriptor.of(
                "gemma", ArchitectureFamilies.DECODER_ONLY_TRANSFORMER,
                ArchitectureConfiguration.empty());

        assertThrows(IllegalArgumentException.class, () -> new ModelDescriptor(
                ModelIdentity.of("model-1"), "llama", "safetensors",
                Map.of(), Map.of(), List.of(), java.util.Optional.of(architecture)));
    }

    @Test
    void rejectsInvalidArchitectureAndConfigurationEntries() {
        assertThrows(IllegalArgumentException.class,
                () -> ArchitectureFamily.custom(" "));
        assertThrows(IllegalArgumentException.class,
                () -> new ArchitectureDescriptor(
                        " ", ArchitectureFamilies.DECODER_ONLY_TRANSFORMER,
                        ArchitectureConfiguration.empty(), Set.of(), Map.of()));
        assertThrows(NullPointerException.class,
                () -> new ArchitectureConfiguration(null, Map.of()));
    }
}
