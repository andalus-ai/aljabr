package tech.kayys.aljabr.safetensor.quantization.quantizer;

import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;
import tech.kayys.aljabr.safetensor.quantization.QuantConfig;
import tech.kayys.tafkir.quantizer.turboquant.BnBDequantizer;
import org.jboss.logging.Logger;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Adapter for BitsAndBytes (BnB) quantization formats (NF4, INT8).
 *
 * <p>Key design: {@link #quantizeTensor} reads weights in fixed-size chunks directly
 * from the tensor's MemorySegment to avoid allocating a full F32 copy of the entire
 * weight matrix at once. For large tensors like lm_head (~623M elements / ~2.5GB F32),
 * the old approach of {@code tensor.toFloatArray()} would exhaust JVM direct buffer
 * memory. Chunked streaming quantization caps peak allocation at ~4MB per chunk
 * regardless of tensor size.
 * @author bhangun
 */
public class BnBQuantizerAdapter implements Quantizer {

    private static final Logger log = Logger.getLogger(BnBQuantizerAdapter.class);

    /** Number of float elements processed per chunk. 1M floats ≈ 4MB per chunk. */
    private static final int CHUNK_ELEMENTS = 1 << 20; // 1,048,576

    private final BnBDequantizer engine = new BnBDequantizer();

    @Override
    public AccelTensor quantizeTensor(AccelTensor tensor, QuantConfig config) {
        if (tensor == null) throw new IllegalArgumentException("Tensor cannot be null");

        int numElements = (int) tensor.numel();
        int blockSize = 64; // NF4 default
        int numBlocks = (numElements + blockSize - 1) / blockSize;

        byte[] packedOut = new byte[(numElements + 1) / 2];
        float[] absmaxOut = new float[numBlocks];

        AccelTensor.QuantType dtype = tensor.quantType();
        MemorySegment seg = tensor.dataPtr();

        // Process in CHUNK_ELEMENTS-sized slices to avoid OOM on huge tensors.
        // Each chunk reads BF16/F32 values directly from the MemorySegment.
        int srcOffset = 0;
        while (srcOffset < numElements) {
            int chunkLen = Math.min(CHUNK_ELEMENTS, numElements - srcOffset);
            float[] chunkFloats = readChunk(seg, dtype, srcOffset, chunkLen);
            engine.quantizeNF4Chunk(chunkFloats, packedOut, absmaxOut, srcOffset);
            srcOffset += chunkLen;
        }

        // Build quantized AccelTensor and attach NF4 metadata
        AccelTensor quantized = AccelTensor.fromByteArray(packedOut, tensor.shape());

        java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofAuto();
        MemorySegment scaleSeg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, absmaxOut);

        quantized.withQuantization(
                AccelTensor.QuantType.NF4,
                scaleSeg,
                null,   // no zeros for NF4
                blockSize
        );

        int chunks = (numElements + CHUNK_ELEMENTS - 1) / CHUNK_ELEMENTS;
        log.debugf("BnB: quantized %d elements to NF4 (blockSize=%d, chunks=%d, dtype=%s)",
                numElements, blockSize, chunks, dtype);
        return quantized;
    }

    /**
     * Reads a contiguous slice of float values from the tensor's backing MemorySegment.
     *
     * @param seg       the tensor's raw data segment
     * @param dtype     the on-disk dtype (F32 or BF16 are the common cases)
     * @param elemStart the element index to start reading from
     * @param count     the number of elements to read
     * @return a fresh float[] of length {@code count} with the decoded values
     */
    private static float[] readChunk(MemorySegment seg, AccelTensor.QuantType dtype, int elemStart, int count) {
        float[] out = new float[count];
        switch (dtype) {
            case F32 -> {
                long byteStart = (long) elemStart * Float.BYTES;
                MemorySegment.copy(seg, ValueLayout.JAVA_FLOAT, byteStart, out, 0, count);
            }
            case BF16 -> {
                for (int i = 0; i < count; i++) {
                    short raw = seg.getAtIndex(ValueLayout.JAVA_SHORT, (long) (elemStart + i));
                    // BF16 → F32: shift left 16 bits (BF16 is the top 16 bits of F32)
                    out[i] = Float.intBitsToFloat((raw & 0xFFFF) << 16);
                }
            }
            case F16 -> {
                for (int i = 0; i < count; i++) {
                    short raw = seg.getAtIndex(ValueLayout.JAVA_SHORT, (long) (elemStart + i));
                    out[i] = fp16ToFloat(raw);
                }
            }
            default -> {
                // For other types, fall back to a partial dequantize via the segment directly
                // (should rarely be hit; weights reaching here should be BF16 or F32)
                for (int i = 0; i < count; i++) {
                    out[i] = seg.getAtIndex(ValueLayout.JAVA_FLOAT, elemStart + i);
                }
            }
        }
        return out;
    }

    /** IEEE 754 FP16 → float32 conversion. */
    private static float fp16ToFloat(short h) {
        int bits = h & 0xFFFF;
        int sign  = (bits >> 15) & 1;
        int exp   = (bits >> 10) & 0x1F;
        int mant  = bits & 0x3FF;
        if (exp == 0) {
            // Subnormal
            return (sign == 0 ? 1 : -1) * (float) Math.scalb(mant / 1024.0, -14);
        }
        if (exp == 31) {
            return mant == 0 ? (sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY) : Float.NaN;
        }
        return Float.intBitsToFloat((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13));
    }

    @Override
    public AccelTensor dequantizeTensor(AccelTensor quantizedTensor, QuantConfig config) {
        log.debugf("BnB: dequantizing NF4/INT8 tensor %s", java.util.Arrays.toString(quantizedTensor.shape()));

        MemorySegment dataSeg   = quantizedTensor.dataSegment();
        MemorySegment scalesSeg = quantizedTensor.scales();

        if (scalesSeg == null) {
            log.warn("BnB dequantization requested but no scales found on tensor");
            return quantizedTensor;
        }

        int numElements = (int) quantizedTensor.numel();
        float[] output = new float[numElements];

        byte[]  packedBytes = dataSeg.toArray(ValueLayout.JAVA_BYTE);
        float[] absmax      = scalesSeg.toArray(ValueLayout.JAVA_FLOAT);

        engine.dequantNF4(packedBytes, absmax, numElements, output);

        return AccelTensor.fromFloatArray(output, quantizedTensor.shape());
    }

    @Override
    public String getName() {
        return "BitsAndBytes";
    }

    @Override
    public boolean supports(QuantConfig config) {
        return config.getStrategy() == tech.kayys.aljabr.safetensor.quantization.QuantizationEngine.QuantStrategy.INT8
                || config.getBits() == 4; // NF4
    }
}
