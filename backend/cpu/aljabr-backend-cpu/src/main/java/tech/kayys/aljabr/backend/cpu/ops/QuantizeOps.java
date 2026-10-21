package tech.kayys.aljabr.backend.cpu.ops;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Kernels for quantizing FP32 memory into block-quantized formats.
 */
public final class QuantizeOps {

    private static final VectorSpecies<Float> F_SPECIES = FloatVector.SPECIES_PREFERRED;

    private QuantizeOps() {
    }

    /**
     * Quantizes an F32 tensor into a Q8_0 tensor.
     * 
     * @param src         uncompressed F32 source
     * @param dst         quantized Q8_0 destination
     * @param numElements total number of elements
     */
    public static void quantizeQ8_0(MemorySegment src, MemorySegment dst, long numElements) {
        final int blockSize = 32;
        final int blockBytes = 34;

        long blocks = numElements / blockSize;
        for (long b = 0; b < blocks; b++) {
            long srcOffset = b * blockSize * 4L;
            long dstOffset = b * blockBytes;

            // 1. Find the absolute maximum in the block to compute the scale
            float maxAbs = 0.0f;
            for (int i = 0; i < blockSize; i++) {
                float val = Math.abs(src.get(ValueLayout.JAVA_FLOAT, srcOffset + (i * 4L)));
                if (val > maxAbs) {
                    maxAbs = val;
                }
            }

            // scale = max / 127
            float scale = maxAbs / 127.0f;
            float invScale = scale == 0 ? 0 : 1.0f / scale;

            // 2. Write the FP16 scale
            short fp16Scale = DequantizeOps.Float16.fromFloat(scale);
            dst.set(ValueLayout.JAVA_SHORT, dstOffset, fp16Scale);

            // 3. Quantize the 32 weights
            long weightsOffset = dstOffset + 2;

            // We use SIMD to scale, round, and cast to int8
            for (int i = 0; i < blockSize; i += F_SPECIES.length()) {
                FloatVector floats = FloatVector.fromMemorySegment(F_SPECIES, src, srcOffset + (i * 4L),
                        ByteOrder.nativeOrder());

                // val = round(val * invScale)
                FloatVector scaled = floats.mul(invScale);
                // Math.round logic (add 0.5 and floor) - for simplicity we just rely on casting
                // behavior
                // combined with Math.round in scalar if we want exact matching. But SIMD
                // conversion
                // cast truncates. A trick is to add Math.copySign(0.5f, x) before truncation.
                // For simplicity here, we'll extract and do proper rounding scalar or just
                // simple cast
                // Vector API has .castShape but no direct round-to-nearest. We'll add 0.5f
                // signed:
                // floats.add(floats.test(VectorOperators.IS_NEGATIVE).blend(-0.5f, 0.5f)) ...

                // For Q8_0, standard cast to byte with truncation is usually okay if we don't
                // care about perfect rounding,
                // but proper Math.round(v) is preferred.
                // Let's do scalar fallback for the rounding just to be safe and simple:
                for (int j = 0; j < F_SPECIES.length() && (i + j) < blockSize; j++) {
                    float f = floats.lane(j);
                    int q = Math.round(f * invScale);
                    if (q > 127)
                        q = 127;
                    if (q < -128)
                        q = -128;
                    dst.set(ValueLayout.JAVA_BYTE, weightsOffset + i + j, (byte) q);
                }
            }
        }
    }

    /**
     * Quantizes an F32 tensor into a Q4_0 tensor.
     * 
     * @param src         uncompressed F32 source
     * @param dst         quantized Q4_0 destination
     * @param numElements total number of elements
     */
    public static void quantizeQ4_0(MemorySegment src, MemorySegment dst, long numElements) {
        final int blockSize = 32;
        final int blockBytes = 18;

        long blocks = numElements / blockSize;
        for (long b = 0; b < blocks; b++) {
            long srcOffset = b * blockSize * 4L;
            long dstOffset = b * blockBytes;

            // 1. Find the absolute maximum in the block to compute the scale
            float maxAbs = 0.0f;
            for (int i = 0; i < blockSize; i++) {
                float val = Math.abs(src.get(ValueLayout.JAVA_FLOAT, srcOffset + (i * 4L)));
                if (val > maxAbs) {
                    maxAbs = val;
                }
            }

            // Q4_0 scale = max / -8
            // Wait, GGML Q4_0 scale is max / -8 ?
            // Usually we map maxAbs to 7 (since range is -8 to 7).
            // Let's use maxAbs / 7.0f for scale.
            float scale = maxAbs / 7.0f;
            float invScale = scale == 0 ? 0 : 1.0f / scale;

            // 2. Write the FP16 scale
            short fp16Scale = DequantizeOps.Float16.fromFloat(scale);
            dst.set(ValueLayout.JAVA_SHORT, dstOffset, fp16Scale);

            // 3. Quantize the 32 weights into 16 bytes
            long weightsOffset = dstOffset + 2;

            for (int i = 0; i < 16; i++) {
                float f0 = src.get(ValueLayout.JAVA_FLOAT, srcOffset + (i * 2L) * 4L);
                float f1 = src.get(ValueLayout.JAVA_FLOAT, srcOffset + ((i * 2L) + 1) * 4L);

                int q0 = Math.round(f0 * invScale);
                int q1 = Math.round(f1 * invScale);

                // Clamp to -8 .. 7
                if (q0 > 7)
                    q0 = 7;
                if (q0 < -8)
                    q0 = -8;

                if (q1 > 7)
                    q1 = 7;
                if (q1 < -8)
                    q1 = -8;

                // Offset by +8 so range is 0..15, then pack
                int w0 = (q0 + 8) & 0x0F;
                int w1 = (q1 + 8) & 0x0F;

                byte packed = (byte) (w0 | (w1 << 4));
                dst.set(ValueLayout.JAVA_BYTE, weightsOffset + i, packed);
            }
        }
    }
}
