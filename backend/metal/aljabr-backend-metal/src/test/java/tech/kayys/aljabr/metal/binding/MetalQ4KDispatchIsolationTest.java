package tech.kayys.aljabr.metal.binding;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;


import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Isolates the Q4_K batch dispatch kernels from the triple-matvec wrapper.
 * Mode 1 forces B4; mode 2 forces the batch-aware MR4 kernel.
 */
class MetalQ4KDispatchIsolationTest {
    private static final int K = 3840;
    private static final int N = 4096;
    private static final int BATCH = 16;
    private static final float TOLERANCE = 2.0e-3f;

    @Test
    void forcedB4AndMr4MatchRepeatedSingleRowQ4K() {
        Path dylib = findBuiltDylib();
        assumeTrue(dylib != null, "Native Metal dylib is not built in this checkout");

        MetalBinding.reset();
        assertTrue(MetalBinding.initialize(dylib));
        MetalBinding binding = MetalBinding.getInstance();
        assertEquals(0, binding.init());
        assumeTrue(binding.isRuntimeActive(), "Native Metal runtime is not active");

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate((long) BATCH * K * Float.BYTES, 64);
            MemorySegment weights = arena.allocate((long) N * K / 256 * 144, 64);
            MemorySegment reference = arena.allocate((long) N * Float.BYTES, 64);
            MemorySegment b4 = arena.allocate((long) BATCH * N * Float.BYTES, 64);
            MemorySegment mr4 = arena.allocate((long) BATCH * N * Float.BYTES, 64);

            for (int i = 0; i < BATCH * K; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, ((i % 31) - 15) / 31.0f);
            }
            // Reproduce the deterministic packed-byte fixture that exposes the
            // batch-16 discrepancy in MetalTripleBatchCorrectnessTest.
            weights.fill((byte) 0x55);

            assertEquals(0, binding.matvecQ4KBatchDispatchForBenchmark(
                    b4, input, weights, K, N, BATCH, 1), "Forced B4 dispatch failed");
            assertEquals(0, binding.matvecQ4KBatchDispatchForBenchmark(
                    mr4, input, weights, K, N, BATCH, 2), "Forced MR4 dispatch failed");

            long rowBytes = (long) K * Float.BYTES;
            for (int row = 0; row < BATCH; row++) {
                MemorySegment inputRow = input.asSlice((long) row * rowBytes, rowBytes);
                assertEquals(0, binding.matvecTransposedRightQ4K(
                        reference, inputRow, weights, K, N),
                        "Single-row reference failed at row=" + row);
                assertRowClose("MR4", mr4, row, reference);
                assertRowClose("B4", b4, row, reference);
            }
        }
    }

    private static void assertRowClose(String label, MemorySegment batch,
            int row, MemorySegment reference) {
        long offset = (long) row * N;
        float maxError = 0.0f;
        int worstIndex = -1;
        float expectedAtWorst = 0.0f;
        float actualAtWorst = 0.0f;
        for (int i = 0; i < N; i++) {
            float expected = reference.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float actual = batch.getAtIndex(ValueLayout.JAVA_FLOAT, offset + i);
            assertTrue(Float.isFinite(expected) && Float.isFinite(actual),
                    label + " produced non-finite output at row=" + row + ", index=" + i);
            float error = Math.abs(actual - expected) / Math.max(1.0f, Math.abs(expected));
            if (error > maxError) {
                maxError = error;
                worstIndex = i;
                expectedAtWorst = expected;
                actualAtWorst = actual;
            }
        }
        assertTrue(maxError <= TOLERANCE,
                label + " mismatch at row=" + row + ", maxError=" + maxError
                        + ", index=" + worstIndex + ", expected=" + expectedAtWorst
                        + ", actual=" + actualAtWorst + ", tolerance=" + TOLERANCE);
    }

    private static Path findBuiltDylib() {
        List<Path> candidates = List.of(
                Path.of("backend/metal/aljabr-backend-metal/target/native/darwin-aarch64/libaljabr_metal.dylib"),
                Path.of("target/native/darwin-aarch64/libaljabr_metal.dylib"),
                Path.of(System.getProperty("user.home"), ".aljabr", "libs", "libaljabr_metal.dylib"));
        for (Path candidate : candidates) {
            if (Files.isRegularFile(candidate)) {
                System.out.println("[MetalQ4KDispatchIsolationTest] Loading native Metal dylib: "
                        + candidate.toAbsolutePath().normalize());
                return candidate;
            }
        }
        return null;
    }
}
