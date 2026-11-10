package tech.kayys.aljabr.safetensor.quantization;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class QuantizationFormatRegistryTest {
    private QuantizedTensorDescriptor descriptor(String id, int version) {
        return QuantizedTensorDescriptor.builder()
                .formatId(id)
                .formatVersion(version)
                .logicalShape(4, 4)
                .logicalDtype("F16")
                .storageDtype("U4")
                .bitWidth(4)
                .payloadLengthBytes(8)
                .build();
    }

    @Test
    void registryStartsEmptyAndRequiresExactVersion() {
        var registry = new QuantizationFormatRegistry();
        assertTrue(registry.getDefinitions().isEmpty());
        assertFalse(registry.supports("example.int4", 1));
        assertThrows(QuantizationFormatRegistry.UnsupportedQuantizationFormatException.class,
                () -> registry.validate(descriptor("example.int4", 1)));

        registry.register(new QuantizationFormatRegistry.FormatDefinition(
                "example.int4", 1, "Example four-bit layout", ignored -> { }));

        assertTrue(registry.supports("example.int4", 1));
        assertFalse(registry.supports("example.int4", 2));
        assertDoesNotThrow(() -> registry.validate(descriptor("example.int4", 1)));
        assertThrows(QuantizationFormatRegistry.UnsupportedQuantizationFormatException.class,
                () -> registry.validate(descriptor("example.int4", 2)));
    }

    @Test
    void invokesFormatSpecificValidator() {
        var registry = new QuantizationFormatRegistry();
        registry.register(new QuantizationFormatRegistry.FormatDefinition(
                "example.grouped", 1, "Grouped example", descriptor -> {
                    if (descriptor.getGroupSize() == null || descriptor.getGroupSize() != 4) {
                        throw new IllegalArgumentException("groupSize must be 4");
                    }
                }));

        assertThrows(IllegalArgumentException.class,
                () -> registry.validate(descriptor("example.grouped", 1)));

        var valid = QuantizedTensorDescriptor.builder()
                .formatId("example.grouped").formatVersion(1)
                .logicalShape(4, 4).logicalDtype("F16").storageDtype("U4")
                .bitWidth(4).payloadLengthBytes(8).groupSize(4).build();
        assertDoesNotThrow(() -> registry.validate(valid));
    }

    @Test
    void rejectsDuplicateRegistrationAndInvalidDefinition() {
        var registry = new QuantizationFormatRegistry();
        var definition = new QuantizationFormatRegistry.FormatDefinition(
                "example.int8", 1, "Example eight-bit layout", ignored -> { });
        registry.register(definition);

        assertThrows(IllegalStateException.class, () -> registry.register(definition));
        assertThrows(IllegalArgumentException.class, () ->
                new QuantizationFormatRegistry.FormatDefinition("example.int8", 0, "bad", ignored -> { }));
        assertThrows(IllegalArgumentException.class, () ->
                new QuantizationFormatRegistry.FormatDefinition("example.int8", 1, " ", ignored -> { }));
        assertThrows(NullPointerException.class, () ->
                new QuantizationFormatRegistry.FormatDefinition("example.int8", 1, "bad", null));
    }

    @Test
    void definitionsAreReturnedAsImmutableSnapshot() {
        var registry = new QuantizationFormatRegistry();
        registry.register(new QuantizationFormatRegistry.FormatDefinition(
                "example.b", 1, "B", ignored -> { }));
        registry.register(new QuantizationFormatRegistry.FormatDefinition(
                "example.a", 1, "A", ignored -> { }));

        var definitions = registry.getDefinitions();
        assertEquals("example.a", definitions.keySet().iterator().next().formatId());
        assertThrows(UnsupportedOperationException.class, definitions::clear);
    }
}
