package tech.kayys.aljabr.safetensor.loader;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.safetensor.embeddinggemma2.EmbeddingGemma2SafetensorLoader;

class EmbeddingGemma2SafetensorConversionTest {

    @Test
    void expandsLittleEndianBfloat16IntoDetachedFloat32Values() {
        byte[] bytes = {(byte) 0x80, 0x3f, 0x00, 0x40}; // 1.0, 2.0 in BF16
        SafetensorTensor tensor = tensor("weight", SafetensorDType.BF16, new long[]{2}, bytes);

        var converted = EmbeddingGemma2SafetensorLoader.toCheckpointTensor(tensor);
        tensor.close();

        assertArrayEquals(new int[]{2}, converted.shape());
        assertArrayEquals(new float[]{1.0f, 2.0f}, converted.values());
    }

    @Test
    void copiesFloat32PayloadWithoutChangingValues() {
        byte[] bytes = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(-0.5f).putFloat(3.25f).array();
        SafetensorTensor tensor = tensor("weight", SafetensorDType.F32, new long[]{1, 2}, bytes);

        var converted = EmbeddingGemma2SafetensorLoader.toCheckpointTensor(tensor);

        assertArrayEquals(new int[]{1, 2}, converted.shape());
        assertArrayEquals(new float[]{-0.5f, 3.25f}, converted.values());
    }

    @Test
    void expandsLittleEndianFloat16IntoDetachedFloat32Values() {
        // IEEE-754 binary16 encodings for 1.0 and -2.0.
        byte[] bytes = {0x00, 0x3c, 0x00, (byte) 0xc0};
        SafetensorTensor tensor = tensor("weight", SafetensorDType.F16, new long[]{2}, bytes);

        var converted = EmbeddingGemma2SafetensorLoader.toCheckpointTensor(tensor);
        tensor.close();

        assertArrayEquals(new int[]{2}, converted.shape());
        assertArrayEquals(new float[]{1.0f, -2.0f}, converted.values());
    }

    private static SafetensorTensor tensor(
            String name, SafetensorDType dtype, long[] shape, byte[] bytes) {
        SafetensorTensorInfo info = SafetensorTensorInfo.of(name, dtype, shape, 0, bytes.length);
        return new SafetensorTensor(info, MemorySegment.ofArray(bytes));
    }
}
