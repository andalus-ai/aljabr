package tech.kayys.aljabr.backend.cpu.ops;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tensor.DType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertTrue;

class DequantizeOpsTest {

    @Test
    void testQ8_0QuantizeAndDequantize() {
        try (Arena arena = Arena.ofConfined()) {
            int numElements = 256; // 8 blocks of 32

            MemorySegment original = arena.allocate(numElements * 4L);
            MemorySegment quantized = arena.allocate(DType.Q8_0.memoryFootprintBytes(numElements));
            MemorySegment dequantized = arena.allocate(numElements * 4L);

            // Populate original with random floats between -1 and 1
            Random r = new Random(42);
            for (int i = 0; i < numElements; i++) {
                original.set(ValueLayout.JAVA_FLOAT, i * 4L, (r.nextFloat() * 2) - 1.0f);
            }

            // Quantize F32 -> Q8_0
            QuantizeOps.quantizeQ8_0(original, quantized, numElements);

            // Dequantize Q8_0 -> F32
            DequantizeOps.dequantizeQ8_0(quantized, dequantized, numElements);

            // Measure Cosine Similarity and MSE
            double dot = 0;
            double normA = 0;
            double normB = 0;
            double mse = 0;

            for (int i = 0; i < numElements; i++) {
                float a = original.get(ValueLayout.JAVA_FLOAT, i * 4L);
                float b = dequantized.get(ValueLayout.JAVA_FLOAT, i * 4L);

                dot += (a * b);
                normA += (a * a);
                normB += (b * b);

                double diff = a - b;
                mse += diff * diff;
            }

            double cosSim = dot / (Math.sqrt(normA) * Math.sqrt(normB));
            mse = mse / numElements;

            System.out.printf("Q8_0 Cosine Similarity: %.4f\n", cosSim);
            System.out.printf("Q8_0 MSE: %.6f\n", mse);

            assertTrue(cosSim > 0.99, "Cosine similarity too low: " + cosSim);
            assertTrue(mse < 0.001, "MSE too high: " + mse);
        }
    }

    @Test
    void testQ4_0QuantizeAndDequantize() {
        try (Arena arena = Arena.ofConfined()) {
            int numElements = 256; // 8 blocks of 32

            MemorySegment original = arena.allocate(numElements * 4L);
            MemorySegment quantized = arena.allocate(DType.Q4_0.memoryFootprintBytes(numElements));
            MemorySegment dequantized = arena.allocate(numElements * 4L);

            // Populate original with random floats between -1 and 1
            Random r = new Random(42);
            for (int i = 0; i < numElements; i++) {
                original.set(ValueLayout.JAVA_FLOAT, i * 4L, (r.nextFloat() * 2) - 1.0f);
            }

            // Quantize F32 -> Q4_0
            QuantizeOps.quantizeQ4_0(original, quantized, numElements);

            // Dequantize Q4_0 -> F32
            DequantizeOps.dequantizeQ4_0(quantized, dequantized, numElements);

            // Measure Cosine Similarity and MSE
            double dot = 0;
            double normA = 0;
            double normB = 0;
            double mse = 0;

            for (int i = 0; i < numElements; i++) {
                float a = original.get(ValueLayout.JAVA_FLOAT, i * 4L);
                float b = dequantized.get(ValueLayout.JAVA_FLOAT, i * 4L);

                dot += (a * b);
                normA += (a * a);
                normB += (b * b);

                double diff = a - b;
                mse += diff * diff;
            }

            double cosSim = dot / (Math.sqrt(normA) * Math.sqrt(normB));
            mse = mse / numElements;

            System.out.printf("Q4_0 Cosine Similarity: %.4f\n", cosSim);
            System.out.printf("Q4_0 MSE: %.6f\n", mse);

            // Q4_0 has higher error than Q8_0, so thresholds are relaxed
            assertTrue(cosSim > 0.95, "Cosine similarity too low: " + cosSim);
            assertTrue(mse < 0.05, "MSE too high: " + mse);
        }
    }
}
