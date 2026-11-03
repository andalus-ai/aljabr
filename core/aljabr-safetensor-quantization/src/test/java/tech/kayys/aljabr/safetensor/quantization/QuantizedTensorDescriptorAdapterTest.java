package tech.kayys.aljabr.safetensor.quantization;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class QuantizedTensorDescriptorAdapterTest {

    private QuantizedTensorDescriptorAdapter.Representation representation() {
        return new QuantizedTensorDescriptorAdapter.Representation(
                "example.int4.packed",
                1,
                "U4",
                4,
                QuantizedTensorDescriptor.PackingOrder.LOW_BITS_FIRST,
                4,
                1,
                32);
    }

    @Test
    void mapsOnlyLegacyFactsAndExplicitRepresentationDetails() {
        QuantizedTensorInfo info = QuantizedTensorInfo.builder()
                .name("weights")
                .shape(new int[] {4, 8})
                .originalDtype("F16")
                .quantizedDtype("INT4")
                .scales(new float[] {0.5f, 0.25f})
                .zeros(new float[] {0.0f, 1.0f})
                .metadata(new byte[] {7, 8})
                .build();

        QuantizedTensorDescriptor descriptor =
                QuantizedTensorDescriptorAdapter.toDescriptor(info, representation());

        assertEquals("example.int4.packed", descriptor.getFormatId());
        assertEquals(1, descriptor.getFormatVersion());
        assertArrayEquals(new long[] {4, 8}, descriptor.getLogicalShape());
        assertEquals("F16", descriptor.getLogicalDtype());
        assertEquals("U4", descriptor.getStorageDtype());
        assertEquals(4, descriptor.getBitWidth());
        assertEquals(QuantizedTensorDescriptor.PackingOrder.LOW_BITS_FIRST,
                descriptor.getPackingOrder());
        assertEquals(4, descriptor.getPayloadLengthBytes());
        assertEquals(1, descriptor.getQuantizationAxis());
        assertEquals(32, descriptor.getGroupSize());
        assertEquals(2, descriptor.getScaleCount());
        assertEquals(2, descriptor.getZeroPointCount());
        assertArrayEquals(new byte[] {7, 8}, descriptor.getAuxiliaryMetadata());
    }

    @Test
    void rejectsMissingLegacyShapeInsteadOfGuessing() {
        QuantizedTensorInfo info = QuantizedTensorInfo.builder().build();

        assertThrows(NullPointerException.class,
                () -> QuantizedTensorDescriptorAdapter.toDescriptor(info, representation()));
    }

    @Test
    void rejectsPayloadThatCannotContainTheDeclaredTensor() {
        QuantizedTensorInfo info = QuantizedTensorInfo.builder()
                .shape(new int[] {4, 8})
                .originalDtype("F16")
                .build();
        var invalid = new QuantizedTensorDescriptorAdapter.Representation(
                "example.int4.packed", 1, "U4", 4,
                QuantizedTensorDescriptor.PackingOrder.LOW_BITS_FIRST,
                1, 1, 32);

        assertThrows(IllegalArgumentException.class,
                () -> QuantizedTensorDescriptorAdapter.toDescriptor(info, invalid));
    }

    @Test
    void doesNotInferFormatFromLegacyDtypeOrStrategy() {
        QuantizedTensorInfo info = QuantizedTensorInfo.builder()
                .shape(new int[] {4, 8})
                .originalDtype("F16")
                .quantizedDtype("INT4")
                .build();

        assertThrows(NullPointerException.class,
                () -> QuantizedTensorDescriptorAdapter.toDescriptor(info, null));
    }
}
