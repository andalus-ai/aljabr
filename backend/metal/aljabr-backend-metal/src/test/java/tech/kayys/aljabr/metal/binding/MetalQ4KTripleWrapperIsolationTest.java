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
 * Compares the same Q4_K fixture through the standalone batch API and the
 * triple-batch wrapper. This isolates wrapper/scratch interaction from kernel
 * dispatch while preserving the original failing fixture exactly.
 */
class MetalQ4KTripleWrapperIsolationTest {
    private static final int K = 3840;
    private static final int N0 = 4096;
    private static final int N1 = 2048;
    private static final int N2 = 2048;
    private static final int BATCH = 16;
    private static final int Q4_K = 12;
    private static final int Q6_K = 14;
    private static final float TOLERANCE = 2.0e-3f;

    @Test
    void standaloneQ4KBatchAndTripleWrapperMatchSingleRowReference() {
        Path dylib = findBuiltDylib();
        assumeTrue(dylib != null, "Native Metal dylib is not built in this checkout");

        MetalBinding.reset();
        assertTrue(MetalBinding.initialize(dylib));
        MetalBinding binding = MetalBinding.getInstance();
        assertEquals(0, binding.init());
        assumeTrue(binding.isRuntimeActive(), "Native Metal runtime is not active");

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate((long) BATCH * K * Float.BYTES, 64);
            MemorySegment weights0 = arena.allocate((long) N0 * K / 256 * 144, 64);
            MemorySegment weights1 = arena.allocate((long) N1 * K / 256 * 144, 64);
            MemorySegment weights2 = arena.allocate((long) N2 * K / 256 * 210, 64);
            MemorySegment standalone = arena.allocate((long) BATCH * N0 * Float.BYTES, 64);
            MemorySegment triple0 = arena.allocate((long) BATCH * N0 * Float.BYTES, 64);
            MemorySegment triple1 = arena.allocate((long) BATCH * N1 * Float.BYTES, 64);
            MemorySegment triple2 = arena.allocate((long) BATCH * N2 * Float.BYTES, 64);
            MemorySegment reference = arena.allocate((long) N0 * Float.BYTES, 64);

            for (int i = 0; i < BATCH * K; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, ((i % 31) - 15) / 31.0f);
            }
            weights0.fill((byte) 0x55);
            weights1.fill((byte) 0x35);
            weights2.fill((byte) 0x45);

            int standaloneRc = binding.matvecTransposedRightBatch(
                    standalone, input, weights0, K, N0, Q4_K, BATCH);
            assertEquals(0, standaloneRc, "Standalone Q4_K batch API failed");

            int tripleRc = binding.matvecTransposedRightTripleBatch(
                    triple0, triple1, triple2, input,
                    weights0, weights1, weights2,
                    K, N0, N1, N2, Q4_K, Q4_K, Q6_K, BATCH);
            assertEquals(0, tripleRc, "Mixed-quantization triple-batch API failed");

            long rowBytes = (long) K * Float.BYTES;
            StringBuilder mismatches = new StringBuilder();
            for (int row = 0; row < BATCH; row++) {
                MemorySegment inputRow = input.asSlice((long) row * rowBytes, rowBytes);
                int singleRc = binding.matvecTransposedRightQ4K(
                        reference, inputRow, weights0, K, N0);
                assertEquals(0, singleRc, "Single-row Q4_K reference failed at row=" + row);

                appendMismatch(mismatches, "standalone", standalone, row, reference);
                appendMismatch(mismatches, "triple", triple0, row, reference);
                appendCrossMismatch(mismatches, standalone, triple0, row);
            }

            assertTrue(mismatches.length() == 0,
                    "Q4_K path comparison failed (tolerance=" + TOLERANCE + "):\n" + mismatches);
        }
    }

    private static void appendMismatch(StringBuilder out, String label,
            MemorySegment actual, int row, MemorySegment reference) {
        float max = 0.0f;
        int worst = -1;
        float expectedAtWorst = 0.0f;
        float actualAtWorst = 0.0f;
        long base = (long) row * N0;
        for (int i = 0; i < N0; i++) {
            float expected = reference.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float value = actual.getAtIndex(ValueLayout.JAVA_FLOAT, base + i);
            float error = Math.abs(value - expected) / Math.max(1.0f, Math.abs(expected));
            if (!Float.isFinite(value) || !Float.isFinite(expected) || error > max) {
                max = error;
                worst = i;
                expectedAtWorst = expected;
                actualAtWorst = value;
            }
        }
        if (max > TOLERANCE || worst >= 0 && (!Float.isFinite(expectedAtWorst) || !Float.isFinite(actualAtWorst))) {
            out.append(label).append(" row=").append(row)
                    .append(" maxError=").append(max).append(" index=").append(worst)
                    .append(" expected=").append(expectedAtWorst)
                    .append(" actual=").append(actualAtWorst).append('\n');
        }
    }

    private static void appendCrossMismatch(StringBuilder out,
            MemorySegment standalone, MemorySegment triple, int row) {
        float max = 0.0f;
        int worst = -1;
        float aWorst = 0.0f;
        float bWorst = 0.0f;
        long base = (long) row * N0;
        for (int i = 0; i < N0; i++) {
            float a = standalone.getAtIndex(ValueLayout.JAVA_FLOAT, base + i);
            float b = triple.getAtIndex(ValueLayout.JAVA_FLOAT, base + i);
            float error = Math.abs(a - b) / Math.max(1.0f, Math.abs(a));
            if (!Float.isFinite(a) || !Float.isFinite(b) || error > max) {
                max = error;
                worst = i;
                aWorst = a;
                bWorst = b;
            }
        }
        if (max > TOLERANCE || worst >= 0 && (!Float.isFinite(aWorst) || !Float.isFinite(bWorst))) {
            out.append("standalone-vs-triple row=").append(row)
                    .append(" maxError=").append(max).append(" index=").append(worst)
                    .append(" standalone=").append(aWorst)
                    .append(" triple=").append(bWorst).append('\n');
        }
    }

    private static Path findBuiltDylib() {
        List<Path> candidates = List.of(
                Path.of("backend/metal/aljabr-backend-metal/target/native/darwin-aarch64/libaljabr_metal.dylib"),
                Path.of("target/native/darwin-aarch64/libaljabr_metal.dylib"),
                Path.of(System.getProperty("user.home"), ".aljabr", "libs", "libaljabr_metal.dylib"));
        for (Path candidate : candidates) {
            if (Files.isRegularFile(candidate)) {
                System.out.println("[MetalQ4KTripleWrapperIsolationTest] Loading native Metal dylib: "
                        + candidate.toAbsolutePath().normalize());
                return candidate;
            }
        }
        return null;
    }
}
