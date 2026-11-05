package tech.kayys.aljabr.safetensor.quantization;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class QuantizationParameterDescriptorTest {

    private QuantizationParameterDescriptor parameters(long[] shape, long payloadBytes) {
        return new QuantizationParameterDescriptor(
                "F32", "F32", shape, 32,
                QuantizedTensorDescriptor.PackingOrder.NOT_APPLICABLE,
                payloadBytes);
    }

    @Test
    void describesScaleAndZeroPointLayoutsIndependently() {
        var scales = parameters(new long[] {3}, 12);
        var zeros = new QuantizationParameterDescriptor(
                "I8", "I8", new long[] {3}, 8,
                QuantizedTensorDescriptor.PackingOrder.NOT_APPLICABLE, 3);

        var descriptor = QuantizedTensorDescriptor.builder()
                .formatId("vendor.format")
                .logicalShape(3, 5)
                .logicalDtype("F16")
                .storageDtype("U4")
                .bitWidth(4)
                .payloadLengthBytes(8)
                .scaleCount(3)
                .zeroPointCount(3)
                .scaleParameters(scales)
                .zeroPointParameters(zeros)
                .build();

        assertEquals("F32", descriptor.getScaleParameters().getLogicalDtype());
        assertEquals(32, descriptor.getScaleParameters().getBitWidth());
        assertArrayEquals(new long[] {3}, descriptor.getScaleParameters().getShape());
        assertEquals("I8", descriptor.getZeroPointParameters().getStorageDtype());
        assertEquals(3, descriptor.getZeroPointParameters().getPayloadLengthBytes());
    }

    @Test
    void rejectsParameterShapeCountMismatch() {
        assertThrows(IllegalArgumentException.class, () ->
                QuantizedTensorDescriptor.builder()
                        .formatId("vendor.format")
                        .logicalShape(3, 5)
                        .logicalDtype("F16")
                        .storageDtype("U4")
                        .bitWidth(4)
                        .payloadLengthBytes(8)
                        .scaleCount(3)
                        .scaleParameters(parameters(new long[] {2}, 8))
                        .build());
    }

    @Test
    void rejectsParameterPayloadTooShort() {
        assertThrows(IllegalArgumentException.class, () ->
                new QuantizationParameterDescriptor(
                        "F32", "F32", new long[] {3}, 32,
                        QuantizedTensorDescriptor.PackingOrder.NOT_APPLICABLE, 8));
    }

    @Test
    void defensivelyCopiesParameterShape() {
        long[] shape = {3};
        var descriptor = parameters(shape, 12);
        shape[0] = 99;
        long[] returned = descriptor.getShape();
        returned[0] = 88;
        assertArrayEquals(new long[] {3}, descriptor.getShape());
    }
}
