package tech.kayys.aljabr.converter.gguf;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.core.GgmlType;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class GgmlNativeQuantizerTest {
    @Test
    void shouldUseGgmlReferenceLayoutForQ4KWhenAvailable() {
        GgmlNativeQuantizer quantizer = GgmlNativeQuantizer.load().orElse(null);
        assumeTrue(quantizer != null, "local ggml library is not installed");

        byte[] f32 = new byte[256 * Float.BYTES];
        ByteBuffer buf = ByteBuffer.wrap(f32).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < 256; i++) {
            buf.putFloat((float) Math.sin(i / 7.0) * 0.25f);
        }

        byte[] q4k = quantizer.quantize(f32, 256, 256, GgmlType.Q4_K);

        assertThat(q4k).hasSize(144);
        assertThat(readU16(q4k, 0)).as("d").isNotZero();
        assertThat(readU16(q4k, 2)).as("dmin").isNotZero();
        assertThat(q4k).containsAnyOf((byte) 0x01, (byte) 0x02, (byte) 0x03, (byte) 0x04);
    }

    private static int readU16(byte[] data, int offset) {
        return (data[offset] & 0xff) | ((data[offset + 1] & 0xff) << 8);
    }
}
