package tech.kayys.aljabr.metal.binding;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Regression test for the multi-row Q4_K/Q4_K/Q6_K native triple-matvec path.
 *
 * For batches larger than one, Metal can select specialized B4 kernels rather
 * than the single-row MR4/MR8 kernels. Each batched row must still agree with
 * the corresponding single-row invocation.
 */
class MetalTripleBatchCorrectnessTest {
    private static final int K = 3840;
    private static final int N0 = 4096;
    private static final int N1 = 2048;
    private static final int N2 = 2048;
    private static final int TYPE_Q4_K = 12;
    private static final int TYPE_Q6_K = 14;
    private static final int MAX_BATCH = 32;
    // Include sizes around the Q4_K B4 cutoff and representative prefill batches.
    // Batches above 15 must use the correctness fallback until B4 is fixed.
    private static final int[] BATCH_SIZES = {4, 8, 15, 16, 24, 27, 32};
    private static final float RELATIVE_TOLERANCE = 2.0e-3f;

    @Test
    void batchedMixedQuantizationMatchesRepeatedSingleRowCalls() {
        Path dylib = findBuiltDylib();
        assumeTrue(dylib != null, "Native Metal dylib is not built in this checkout");

        MetalBinding.reset();
        assertTrue(MetalBinding.initialize(dylib));
        MetalBinding binding = MetalBinding.getInstance();
        assertEquals(0, binding.init());
        assumeTrue(binding.isRuntimeActive(), "Native Metal runtime is not active");

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate((long) MAX_BATCH * K * Float.BYTES, 64);
            MemorySegment batchOut0 = arena.allocate((long) MAX_BATCH * N0 * Float.BYTES, 64);
            MemorySegment batchOut1 = arena.allocate((long) MAX_BATCH * N1 * Float.BYTES, 64);
            MemorySegment batchOut2 = arena.allocate((long) MAX_BATCH * N2 * Float.BYTES, 64);

            for (int i = 0; i < MAX_BATCH * K; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, ((i % 31) - 15) / 31.0f);
            }

            // These byte patterns encode finite, deterministic scale values and
            // exercise real quantized kernel decoding without loading a model.
            MemorySegment weights0 = arena.allocate(q4KBytes(N0, K), 64);
            MemorySegment weights1 = arena.allocate(q4KBytes(N1, K), 64);
            MemorySegment weights2 = arena.allocate(q6KBytes(N2, K), 64);
            weights0.fill((byte) 0x55);
            weights1.fill((byte) 0x35);
            weights2.fill((byte) 0x45);

            MemorySegment ref0 = arena.allocate((long) N0 * Float.BYTES, 64);
            MemorySegment ref1 = arena.allocate((long) N1 * Float.BYTES, 64);
            MemorySegment ref2 = arena.allocate((long) N2 * Float.BYTES, 64);
            long inputRowBytes = (long) K * Float.BYTES;

            for (int batchSize : BATCH_SIZES) {
                int batchRc = binding.matvecTransposedRightTripleBatch(
                        batchOut0, batchOut1, batchOut2, input,
                        weights0, weights1, weights2,
                        K, N0, N1, N2,
                        TYPE_Q4_K, TYPE_Q4_K, TYPE_Q6_K, batchSize);
                assertEquals(0, batchRc, "Batched triple matvec failed for batch=" + batchSize);

                for (int row = 0; row < batchSize; row++) {
                    MemorySegment inputRow = input.asSlice((long) row * inputRowBytes, inputRowBytes);
                    int singleRc = binding.matvecTransposedRightTriple(
                            ref0, ref1, ref2, inputRow,
                            weights0, weights1, weights2,
                            K, N0, N1, N2,
                            TYPE_Q4_K, TYPE_Q4_K, TYPE_Q6_K);
                    assertEquals(0, singleRc,
                            "Single-row triple matvec failed for batch=" + batchSize + ", row=" + row);

                    assertRowClose("Q4_K output 0", batchOut0, (long) row * N0, ref0, N0, batchSize, row);
                    assertRowClose("Q4_K output 1", batchOut1, (long) row * N1, ref1, N1, batchSize, row);
                    assertRowClose("Q6_K output 2", batchOut2, (long) row * N2, ref2, N2, batchSize, row);
                }
            }
        }
    }


    @Test
    void batchedMixedQuantizationMatchesRepeatedSingleRowCallsWithRepresentativeWeights() {
        Path dylib = findBuiltDylib();
        assumeTrue(dylib != null, "Native Metal dylib is not built in this checkout");

        MetalBinding.reset();
        assertTrue(MetalBinding.initialize(dylib));
        MetalBinding binding = MetalBinding.getInstance();
        assertEquals(0, binding.init());
        assumeTrue(binding.isRuntimeActive(), "Native Metal runtime is not active");

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate((long) MAX_BATCH * K * Float.BYTES, 64);
            MemorySegment batchOut0 = arena.allocate((long) MAX_BATCH * N0 * Float.BYTES, 64);
            MemorySegment batchOut1 = arena.allocate((long) MAX_BATCH * N1 * Float.BYTES, 64);
            MemorySegment batchOut2 = arena.allocate((long) MAX_BATCH * N2 * Float.BYTES, 64);

            Random inputRandom = new Random(0xA1AB4L);
            for (int i = 0; i < MAX_BATCH * K; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, inputRandom.nextFloat() * 2.0f - 1.0f);
            }

            MemorySegment weights0 = arena.allocate(q4KBytes(N0, K), 64);
            MemorySegment weights1 = arena.allocate(q4KBytes(N1, K), 64);
            MemorySegment weights2 = arena.allocate(q6KBytes(N2, K), 64);
            fillQ4KWeights(weights0, N0, K, 0x4A401L);
            fillQ4KWeights(weights1, N1, K, 0x4A402L);
            fillQ6KWeights(weights2, N2, K, 0x6A403L);

            MemorySegment ref0 = arena.allocate((long) N0 * Float.BYTES, 64);
            MemorySegment ref1 = arena.allocate((long) N1 * Float.BYTES, 64);
            MemorySegment ref2 = arena.allocate((long) N2 * Float.BYTES, 64);
            long inputRowBytes = (long) K * Float.BYTES;

            for (int batchSize : BATCH_SIZES) {
                int batchRc = binding.matvecTransposedRightTripleBatch(
                        batchOut0, batchOut1, batchOut2, input,
                        weights0, weights1, weights2,
                        K, N0, N1, N2,
                        TYPE_Q4_K, TYPE_Q4_K, TYPE_Q6_K, batchSize);
                assertEquals(0, batchRc,
                        "Batched triple matvec failed for representative fixture batch=" + batchSize);

                for (int row = 0; row < batchSize; row++) {
                    MemorySegment inputRow = input.asSlice((long) row * inputRowBytes, inputRowBytes);
                    int singleRc = binding.matvecTransposedRightTriple(
                            ref0, ref1, ref2, inputRow,
                            weights0, weights1, weights2,
                            K, N0, N1, N2,
                            TYPE_Q4_K, TYPE_Q4_K, TYPE_Q6_K);
                    assertEquals(0, singleRc,
                            "Single-row triple matvec failed for representative fixture batch="
                                    + batchSize + ", row=" + row);

                    assertRowClose("representative Q4_K output 0", batchOut0,
                            (long) row * N0, ref0, N0, batchSize, row);
                    assertRowClose("representative Q4_K output 1", batchOut1,
                            (long) row * N1, ref1, N1, batchSize, row);
                    assertRowClose("representative Q6_K output 2", batchOut2,
                            (long) row * N2, ref2, N2, batchSize, row);
                }
            }
        }
    }

    private static void fillQ4KWeights(MemorySegment weights, int n, int k, long seed) {
        final int blockBytes = 144;
        final int halfScale = 0x211f; // approximately 0.01 in IEEE-754 binary16
        final int halfMinScale = 0x1d1f; // approximately 0.005 in IEEE-754 binary16
        Random random = new Random(seed);
        long blocks = (long) n * k / 256;

        for (long block = 0; block < blocks; block++) {
            long offset = block * blockBytes;
            for (int i = 0; i < blockBytes; i++) {
                weights.setAtIndex(ValueLayout.JAVA_BYTE, offset + i, (byte) random.nextInt(256));
            }
            // GGML Q4_K layout: d, dmin, 12 packed scale/min bytes, then 128 q bytes.
            setHalfBits(weights, offset, halfScale);
            setHalfBits(weights, offset + 2, halfMinScale);
        }
    }

    private static void fillQ6KWeights(MemorySegment weights, int n, int k, long seed) {
        final int blockBytes = 210;
        final int scalesOffset = 192;
        final int halfScale = 0x211f; // approximately 0.01 in IEEE-754 binary16
        Random random = new Random(seed);
        long blocks = (long) n * k / 256;

        for (long block = 0; block < blocks; block++) {
            long offset = block * blockBytes;
            for (int i = 0; i < blockBytes; i++) {
                weights.setAtIndex(ValueLayout.JAVA_BYTE, offset + i, (byte) random.nextInt(256));
            }
            // Bound signed sub-block scales while varying packed quantized values.
            for (int i = 0; i < 16; i++) {
                weights.setAtIndex(ValueLayout.JAVA_BYTE, offset + scalesOffset + i,
                        (byte) (random.nextInt(17) - 8));
            }
            // GGML Q6_K layout: ql[128], qh[64], scales[16], then d.
            setHalfBits(weights, offset + 208, halfScale);
        }
    }

    private static void setHalfBits(MemorySegment segment, long offset, int bits) {
        // GGML quant blocks are little-endian on the supported Apple Silicon target.
        segment.setAtIndex(ValueLayout.JAVA_BYTE, offset, (byte) (bits & 0xff));
        segment.setAtIndex(ValueLayout.JAVA_BYTE, offset + 1, (byte) ((bits >>> 8) & 0xff));
    }

    private static void assertRowClose(
            String label, MemorySegment batched, long batchedOffset,
            MemorySegment single, int elements, int batchSize, int row) {
        float maxRelativeError = 0.0f;
        int worstIndex = -1;
        float worstExpected = 0.0f;
        float worstActual = 0.0f;

        for (int i = 0; i < elements; i++) {
            float expected = single.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float actual = batched.getAtIndex(ValueLayout.JAVA_FLOAT, batchedOffset + i);
            assertTrue(Float.isFinite(expected) && Float.isFinite(actual),
                    label + " produced a non-finite value at batch=" + batchSize
                            + ", row=" + row + ", index=" + i
                            + " (expected=" + expected + ", actual=" + actual + ")");

            float relativeError = Math.abs(actual - expected) / Math.max(1.0f, Math.abs(expected));
            if (relativeError > maxRelativeError) {
                maxRelativeError = relativeError;
                worstIndex = i;
                worstExpected = expected;
                worstActual = actual;
            }
        }

        assertTrue(maxRelativeError <= RELATIVE_TOLERANCE,
                label + " differs from the single-row result: batch=" + batchSize
                        + ", row=" + row + ", maxRelativeError=" + maxRelativeError
                        + ", index=" + worstIndex + ", expected=" + worstExpected
                        + ", actual=" + worstActual + ", tolerance=" + RELATIVE_TOLERANCE);
    }

    private static long q4KBytes(int n, int k) {
        // GGML Q4_K: 144 bytes per block of 256 elements.
        return (long) n * k / 256 * 144;
    }

    private static long q6KBytes(int n, int k) {
        // GGML Q6_K: 210 bytes per block of 256 elements.
        return (long) n * k / 256 * 210;
    }

    private static Path findBuiltDylib() {
        return List.of(
                Path.of("target/native/darwin-aarch64/libaljabr_metal.dylib"),
                Path.of("backend/metal/aljabr-backend-metal/target/native/darwin-aarch64/libaljabr_metal.dylib"),
                Path.of(System.getProperty("user.home"), ".aljabr", "libs", "libaljabr_metal.dylib"))
                .stream()
                .filter(Files::isRegularFile)
                .findFirst()
                .orElse(null);
    }
}
