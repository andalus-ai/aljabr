package tech.kayys.aljabr.safetensor.quantization;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class QuantizedTensorDescriptorTest {
    private QuantizedTensorDescriptor valid() {
        return QuantizedTensorDescriptor.builder()
                .formatId("vendor.format")
                .formatVersion(2)
                .logicalShape(3, 5)
                .logicalDtype("F16")
                .storageDtype("U4")
                .bitWidth(4)
                .packingOrder(QuantizedTensorDescriptor.PackingOrder.LOW_BITS_FIRST)
                .payloadLengthBytes(8)
                .quantizationAxis(0)
                .groupSize(5)
                .scaleCount(3)
                .zeroPointCount(3)
                .codebookId("vendor.codebook.v1")
                .auxiliaryMetadata(new byte[] {1, 2})
                .build();
    }

    @Test
    void preservesRepresentationIdentityAndMetadata() {
        var descriptor = valid();
        assertEquals("vendor.format", descriptor.getFormatId());
        assertEquals(2, descriptor.getFormatVersion());
        assertArrayEquals(new long[] {3, 5}, descriptor.getLogicalShape());
        assertEquals(4, descriptor.getBitWidth());
        assertEquals(8, descriptor.getPayloadLengthBytes());
        assertEquals(3, descriptor.getScaleCount());
        assertEquals("vendor.codebook.v1", descriptor.getCodebookId());
    }

    @Test
    void defensivelyCopiesArrayInputsAndOutputs() {
        long[] shape = {3, 5};
        byte[] metadata = {1, 2};
        var descriptor = QuantizedTensorDescriptor.builder()
                .formatId("vendor.format").logicalShape(shape)
                .logicalDtype("F16").storageDtype("U4").bitWidth(4)
                .payloadLengthBytes(8).auxiliaryMetadata(metadata).build();
        shape[0] = 99;
        metadata[0] = 99;
        long[] returnedShape = descriptor.getLogicalShape();
        byte[] returnedMetadata = descriptor.getAuxiliaryMetadata();
        returnedShape[0] = 88;
        returnedMetadata[0] = 88;
        assertArrayEquals(new long[] {3, 5}, descriptor.getLogicalShape());
        assertArrayEquals(new byte[] {1, 2}, descriptor.getAuxiliaryMetadata());
    }

    @Test
    void rejectsMissingIdentityAndInvalidVersion() {
        assertThrows(IllegalArgumentException.class, () -> QuantizedTensorDescriptor.builder().build());
        assertThrows(IllegalArgumentException.class, () -> QuantizedTensorDescriptor.builder()
                .formatId("x").formatVersion(0).logicalShape(1).logicalDtype("F16")
                .storageDtype("U8").bitWidth(8).payloadLengthBytes(1).build());
    }

    @Test
    void rejectsPayloadShorterThanPackedLogicalTensor() {
        assertThrows(IllegalArgumentException.class, () -> QuantizedTensorDescriptor.builder()
                .formatId("x").logicalShape(3, 5).logicalDtype("F16")
                .storageDtype("U4").bitWidth(4).payloadLengthBytes(7).build());
    }

    @Test
    void rejectsInvalidAxisAndShape() {
        assertThrows(IllegalArgumentException.class, () -> QuantizedTensorDescriptor.builder()
                .formatId("x").logicalShape(2, 3).logicalDtype("F16")
                .storageDtype("U4").bitWidth(4).payloadLengthBytes(3)
                .quantizationAxis(2).build());
        assertThrows(IllegalArgumentException.class, () -> QuantizedTensorDescriptor.builder()
                .formatId("x").logicalShape(2, -1).logicalDtype("F16")
                .storageDtype("U4").bitWidth(4).payloadLengthBytes(1).build());
    }
}
