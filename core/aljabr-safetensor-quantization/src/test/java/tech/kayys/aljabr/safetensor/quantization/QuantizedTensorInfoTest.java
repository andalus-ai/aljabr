package tech.kayys.aljabr.safetensor.quantization;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class QuantizedTensorInfoTest {

    @Test
    void defensivelyCopiesArrayInputsAndOutputs() {
        int[] shape = {2, 3};
        float[] scales = {0.5f, 0.25f};
        float[] zeros = {0.0f, 1.0f};
        byte[] metadata = {1, 2, 3};

        QuantizedTensorInfo info = QuantizedTensorInfo.builder()
                .shape(shape)
                .scales(scales)
                .zeros(zeros)
                .metadata(metadata)
                .build();

        shape[0] = 99;
        scales[0] = 99.0f;
        zeros[0] = 99.0f;
        metadata[0] = 99;

        assertArrayEquals(new int[] {2, 3}, info.getShape());
        assertArrayEquals(new float[] {0.5f, 0.25f}, info.getScales());
        assertArrayEquals(new float[] {0.0f, 1.0f}, info.getZeros());
        assertArrayEquals(new byte[] {1, 2, 3}, info.getMetadata());

        info.getShape()[0] = 88;
        info.getScales()[0] = 88.0f;
        info.getZeros()[0] = 88.0f;
        info.getMetadata()[0] = 88;

        assertArrayEquals(new int[] {2, 3}, info.getShape());
        assertArrayEquals(new float[] {0.5f, 0.25f}, info.getScales());
        assertArrayEquals(new float[] {0.0f, 1.0f}, info.getZeros());
        assertArrayEquals(new byte[] {1, 2, 3}, info.getMetadata());
    }
}
