package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.ArchitectureConfiguration;
import tech.kayys.aljabr.spi.model.ir.ArchitectureDescriptor;
import tech.kayys.aljabr.spi.model.ir.ArchitectureFamily;
import tech.kayys.aljabr.spi.model.ir.ArchitectureKind;
import tech.kayys.aljabr.spi.model.ir.ConfigurationValue;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class AmirArchitectureDescriptorCodecTest {

    @Test
    void roundTripsTypedArchitectureConfigurationAndExtensions() {
        ArchitectureDescriptor original = new ArchitectureDescriptor(
                "gemma",
                new ArchitectureFamily("transformer.decoder", ArchitectureKind.DECODER_ONLY_TRANSFORMER),
                new ArchitectureConfiguration(
                        Map.of(
                                "hidden-size", new ConfigurationValue.IntegerValue(3072L),
                                "rms-epsilon", new ConfigurationValue.DecimalValue(new BigDecimal("0.00000100")),
                                "use-cache", new ConfigurationValue.BooleanValue(true),
                                "activation", new ConfigurationValue.StringValue("gelu"),
                                "layers", new ConfigurationValue.IntegerListValue(List.of(1L, 2L, 3L)),
                                "modalities", new ConfigurationValue.StringListValue(List.of("text", "image"))),
                        Map.of("vendor.config", "opaque")),
                Set.of("causal-attention", "grouped-query-attention"),
                Map.of("vendor.architecture", "v2"));

        ArchitectureDescriptor restored = AmirArchitectureDescriptorCodec.decode(
                AmirArchitectureDescriptorCodec.encode(original));

        assertThat(restored).isEqualTo(original);
        assertThat(restored.configuration().properties().get("rms-epsilon"))
                .isEqualTo(new ConfigurationValue.DecimalValue(new BigDecimal("0.00000100")));
    }

    @Test
    void rejectsUnknownConfigurationValueTypes() {
        String malformed = """
                {"id":"gemma","familyId":"transformer.decoder","familyKind":"DECODER_ONLY_TRANSFORMER",
                 "properties":{"hidden":{"type":"future-type","value":"x"}},
                 "configurationExtensions":{},"capabilities":[],"extensions":{}}
                """;

        assertThatThrownBy(() -> AmirArchitectureDescriptorCodec.decode(malformed))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Invalid AMIR architecture descriptor envelope");
    }
}
