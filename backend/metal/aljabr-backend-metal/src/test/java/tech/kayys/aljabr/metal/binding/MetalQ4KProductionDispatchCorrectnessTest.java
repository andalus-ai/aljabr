package tech.kayys.aljabr.metal.binding;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Exercises the public production Q4_K batch entry point and compares every
 * batched row with an independent production single-row invocation.
 *
 * This intentionally does not use the benchmark-only forced-dispatch hook:
 * AUTO must select B4 through batch 15 and MR4 above that boundary.
 */
class MetalQ4KProductionDispatchCorrectnessTest {
    private static final int TYPE_Q4_K = 12;
    private static final int MAX_BATCH = 32;
    private static final int[] BATCH_SIZES = {2, 4, 8, 15, 16, 17, 24, 27, 28, 31, 32};
    private static final float RELATIVE_TOLERANCE = 2.0e-3f;

    @Test
    void productionBatchDispatchMatchesRepeatedSingleRowCalls() {
        MetalBinding.reset();
        Path dylib = findBuiltDylib();
        assumeTrue(dylib != null, "Native Metal dylib is not built in this checkout");
        assertTrue(MetalBinding.initialize(dylib),
                "Could not load freshly built Aljabr Metal library: " + dylib);
        MetalBinding binding = MetalBinding.getInstance();
        assertEquals(0, binding.init(), "Metal runtime initialization returned an error");
        assumeTrue(binding.isRuntimeActive(),
                "Metal runtime is not active; this test requires a real Metal device");

        verifyShape(binding, 1024, 1024, BATCH_SIZES);
        verifyShape(binding, 3840, 2048, BATCH_SIZES);
    }


    private static Path findBuiltDylib() {
        List<Path> candidates = List.of(
                Path.of("backend/metal/aljabr-backend-metal/target/native/darwin-aarch64/libaljabr_metal.dylib"),
                Path.of("target/native/darwin-aarch64/libaljabr_metal.dylib"));
        for (Path candidate : candidates) {
            if (Files.isRegularFile(candidate)) {
                try {
                    System.out.println("[MetalQ4KProductionDispatchCorrectnessTest] Loading native Metal dylib: "
                            + candidate.toAbsolutePath().normalize()
                            + " (modified=" + Files.getLastModifiedTime(candidate) + ")");
                } catch (java.io.IOException e) {
                    System.out.println("[MetalQ4KProductionDispatchCorrectnessTest] Loading native Metal dylib: "
                            + candidate.toAbsolutePath().normalize()
                            + " (modification time unavailable: " + e.getMessage() + ")");
                }
                return candidate;
            }
        }
        return null;
    }

    private static void verifyShape(MetalBinding binding, int k, int n, int[] batchSizes) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate((long) MAX_BATCH * k * Float.BYTES, 64);
            MemorySegment batched = arena.allocate((long) MAX_BATCH * n * Float.BYTES, 64);
            MemorySegment single = arena.allocate((long) n * Float.BYTES, 64);
            MemorySegment weights = arena.allocate(q4KBytes(n, k), 64);

            Random inputRandom = new Random(0xA11AB4L + k * 31L + n);
            for (long i = 0; i < (long) MAX_BATCH * k; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, inputRandom.nextFloat() * 2.0f - 1.0f);
            }
            fillQ4KWeights(weights, n, k, 0x4A401L + k * 17L + n);

            long inputRowBytes = (long) k * Float.BYTES;
            for (int batchSize : batchSizes) {
                int batchRc = binding.matvecTransposedRightBatch(
                        batched, input, weights, k, n, TYPE_Q4_K, batchSize);
                assertEquals(0, batchRc,
                        "Production Q4_K batch dispatch failed: K=" + k + ", N=" + n
                                + ", batch=" + batchSize);

                for (int row = 0; row < batchSize; row++) {
                    MemorySegment inputRow = input.asSlice((long) row * inputRowBytes, inputRowBytes);
                    int singleRc = binding.matvecTransposedRightQ4K(
                            single, inputRow, weights, k, n);
                    assertEquals(0, singleRc,
                            "Production single-row reference failed: K=" + k + ", N=" + n
                                    + ", batch=" + batchSize + ", row=" + row);
                    assertRowClose(batched, (long) row * n, single, n, k, batchSize, row);
                }
            }
        }
    }

    private static void assertRowClose(
            MemorySegment batched, long batchedOffset, MemorySegment single,
            int elements, int k, int batchSize, int row) {
        float maxRelativeError = 0.0f;
        int worstIndex = -1;
        float worstExpected = 0.0f;
        float worstActual = 0.0f;

        for (int i = 0; i < elements; i++) {
            float expected = single.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float actual = batched.getAtIndex(ValueLayout.JAVA_FLOAT, batchedOffset + i);
            assertTrue(Float.isFinite(expected) && Float.isFinite(actual),
                    "Non-finite Q4_K output: K=" + k + ", batch=" + batchSize
                            + ", row=" + row + ", index=" + i
                            + ", expected=" + expected + ", actual=" + actual);

            float error = Math.abs(actual - expected) / Math.max(1.0f, Math.abs(expected));
            if (error > maxRelativeError) {
                maxRelativeError = error;
                worstIndex = i;
                worstExpected = expected;
                worstActual = actual;
            }
        }

        assertTrue(maxRelativeError <= RELATIVE_TOLERANCE,
                "Production batched Q4_K differs from repeated single-row output: K=" + k
                        + ", batch=" + batchSize + ", row=" + row
                        + ", maxRelativeError=" + maxRelativeError
                        + ", index=" + worstIndex + ", expected=" + worstExpected
                        + ", actual=" + worstActual + ", tolerance=" + RELATIVE_TOLERANCE);
    }

    private static void fillQ4KWeights(MemorySegment weights, int n, int k, long seed) {
        final int blockBytes = 144;
        final int halfScale = 0x211f;
        final int halfMinScale = 0x1d1f;
        Random random = new Random(seed);
        long blocks = (long) n * k / 256;

        for (long block = 0; block < blocks; block++) {
            long offset = block * blockBytes;
            setHalfBits(weights, offset, halfScale);
            setHalfBits(weights, offset + 2, halfMinScale);
            for (int i = 0; i < 12; i++) {
                int low = 1 + random.nextInt(15);
                int packed = i < 4 ? low : low | ((1 + random.nextInt(15)) << 4);
                weights.setAtIndex(ValueLayout.JAVA_BYTE, offset + 4 + i, (byte) packed);
            }
            for (int i = 16; i < blockBytes; i++) {
                weights.setAtIndex(ValueLayout.JAVA_BYTE, offset + i, (byte) random.nextInt(256));
            }
        }
    }

    private static void setHalfBits(MemorySegment segment, long offset, int bits) {
        segment.setAtIndex(ValueLayout.JAVA_BYTE, offset, (byte) (bits & 0xff));
        segment.setAtIndex(ValueLayout.JAVA_BYTE, offset + 1, (byte) ((bits >>> 8) & 0xff));
    }

    private static long q4KBytes(int n, int k) {
        return (long) n * k / 256 * 144;
    }
}
