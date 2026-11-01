package tech.kayys.aljabr.metal.binding;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;


import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Isolated Q4_K B4-versus-MR4 benchmark; correctness is checked before timings. */
class MetalQ4KDispatchBenchmarkTest {
    private static final int K = 3840;
    private static final int N = 4096;
    private static final int MODE_B4 = 1;
    private static final int MODE_MR4 = 2;
    private static final int MAX_BATCH = 32;
    private static final int WARMUP_ITERATIONS = 5;
    private static final int MEASURED_ITERATIONS = 20;
    private static final int[] B4_BATCH_SIZES = {4, 8, 12, 15};
    private static final int[] MR4_BATCH_SIZES = {4, 8, 12, 15, 16, 17, 20, 24, 27, 28, 32};
    private static final float RELATIVE_TOLERANCE = 2.0e-3f;

    @Test
    void checksCorrectnessThenReportsIsolatedDispatchTimings() throws IOException {
        assumeTrue(benchmarkEnabled(),
                "Set ALJABR_METAL_BENCH=1 to run the native Q4_K dispatch benchmark");
        assertTrue(MetalBinding.initialize(),
                "Could not load Aljabr Metal library; rebuild/install libaljabr_metal.dylib");
        MetalBinding binding = MetalBinding.getInstance();
        assertEquals(0, binding.init(), "Metal runtime initialization returned an error");
        assertTrue(binding.isRuntimeActive(),
                "Metal runtime is not active; refusing to benchmark a CPU fallback");

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate((long) MAX_BATCH * K * Float.BYTES, 64);
            MemorySegment b4Output = arena.allocate((long) MAX_BATCH * N * Float.BYTES, 64);
            MemorySegment mr4Output = arena.allocate((long) MAX_BATCH * N * Float.BYTES, 64);
            MemorySegment weights = arena.allocate(q4KBytes(N, K), 64);
            for (int i = 0; i < MAX_BATCH * K; i++)
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, ((i % 31) - 15) / 31.0f);
            weights.fill((byte) 0x55);

            for (int batch : B4_BATCH_SIZES) {
                assertEquals(0, invoke(binding, b4Output, input, weights, batch, MODE_B4),
                        "Forced B4 failed for batch=" + batch);
                assertEquals(0, invoke(binding, mr4Output, input, weights, batch, MODE_MR4),
                        "Forced MR4 failed for batch=" + batch);
                assertRowsClose(b4Output, mr4Output, batch);
            }

            String header = "path,batch,mean_ms,median_ms,p90_ms,mean_ms_per_item,items_per_second";
            StringBuilder csv = new StringBuilder(header).append(System.lineSeparator());
            System.out.println("ALJABR_METAL_Q4K_DISPATCH_BENCH K=" + K + " N=" + N
                    + " warmup=" + WARMUP_ITERATIONS + " measured=" + MEASURED_ITERATIONS);
            System.out.println(header);
            for (int batch : B4_BATCH_SIZES)
                appendTiming(csv, binding, b4Output, input, weights, batch, MODE_B4, "B4");
            for (int batch : MR4_BATCH_SIZES)
                appendTiming(csv, binding, mr4Output, input, weights, batch, MODE_MR4, "MR4");

            Path reportPath = repositoryRoot().resolve(
                    "backend/metal/aljabr-backend-metal/build/reports/metal-q4k-dispatch-benchmark.csv");
            Files.createDirectories(reportPath.getParent());
            Files.writeString(reportPath, csv.toString(), StandardCharsets.UTF_8);
            System.err.println("Aljabr Q4_K dispatch benchmark CSV: " + reportPath);
        }
    }

    private static void appendTiming(StringBuilder csv, MetalBinding binding,
            MemorySegment output, MemorySegment input, MemorySegment weights,
            int batch, int mode, String path) {
        for (int i = 0; i < WARMUP_ITERATIONS; i++)
            assertEquals(0, invoke(binding, output, input, weights, batch, mode),
                    path + " warmup failed for batch=" + batch);
        long[] samples = new long[MEASURED_ITERATIONS];
        long totalNanos = 0L;
        for (int i = 0; i < MEASURED_ITERATIONS; i++) {
            long start = System.nanoTime();
            int rc = invoke(binding, output, input, weights, batch, mode);
            long elapsed = System.nanoTime() - start;
            assertEquals(0, rc, path + " call failed for batch=" + batch);
            samples[i] = elapsed;
            totalNanos += elapsed;
        }
        Arrays.sort(samples);
        double meanMs = totalNanos / (double) MEASURED_ITERATIONS / 1_000_000.0;
        double medianMs = percentileNanos(samples, 0.50) / 1_000_000.0;
        double p90Ms = percentileNanos(samples, 0.90) / 1_000_000.0;
        String row = String.format(Locale.ROOT, "%s,%d,%.3f,%.3f,%.3f,%.3f,%.1f",
                path, batch, meanMs, medianMs, p90Ms, meanMs / batch,
                batch * MEASURED_ITERATIONS * 1_000_000_000.0 / totalNanos);
        System.out.println(row);
        csv.append(row).append(System.lineSeparator());
    }

    private static int invoke(MetalBinding binding, MemorySegment output,
            MemorySegment input, MemorySegment weights, int batch, int mode) {
        return binding.matvecQ4KBatchDispatchForBenchmark(output, input, weights, K, N, batch, mode);
    }

    private static void assertRowsClose(MemorySegment b4, MemorySegment mr4, int batch) {
        for (int row = 0; row < batch; row++) {
            for (int i = 0; i < N; i++) {
                long index = (long) row * N + i;
                float expected = mr4.getAtIndex(ValueLayout.JAVA_FLOAT, index);
                float actual = b4.getAtIndex(ValueLayout.JAVA_FLOAT, index);
                assertTrue(Float.isFinite(expected) && Float.isFinite(actual),
                        "Non-finite Q4_K output at batch=" + batch + ", row=" + row + ", index=" + i);
                float error = Math.abs(actual - expected) / Math.max(1.0f, Math.abs(expected));
                assertTrue(error <= RELATIVE_TOLERANCE,
                        "B4 differs from MR4 at batch=" + batch + ", row=" + row + ", index=" + i
                                + ", expected=" + expected + ", actual=" + actual + ", error=" + error);
            }
        }
    }

    private static long q4KBytes(int n, int k) {
        return (long) n * k / 256 * 144;
    }

    private static long percentileNanos(long[] samples, double quantile) {
        int index = (int) Math.ceil(quantile * samples.length) - 1;
        return samples[Math.max(0, Math.min(index, samples.length - 1))];
    }

    private static Path repositoryRoot() {
        Path current = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
        for (Path candidate = current; candidate != null; candidate = candidate.getParent()) {
            if (Files.isRegularFile(candidate.resolve("settings.gradle.kts"))
                    && Files.isRegularFile(candidate.resolve("gradlew")))
                return candidate;
        }
        throw new IllegalStateException("Could not find Aljabr repository root from " + current);
    }

    private static boolean benchmarkEnabled() {
        String value = System.getenv("ALJABR_METAL_BENCH");
        return value != null && ("1".equals(value) || "true".equalsIgnoreCase(value));
    }
}
