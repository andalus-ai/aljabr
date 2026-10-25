package tech.kayys.aljabr.metal.binding;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Opt-in Apple Silicon benchmark for the production native triple-matvec batch path.
 *
 * Run with:
 * ALJABR_METAL_BENCH=1 ./gradlew :backend:metal:aljabr-backend-metal:test \
 *   --tests tech.kayys.aljabr.metal.binding.MetalTripleBatchBenchmarkTest
 *
 * This is deliberately opt-in: regular CI should validate compilation without
 * requiring Apple GPU hardware or allocating benchmark buffers.
 */
class MetalTripleBatchBenchmarkTest {
    private static final int K = 3840;
    private static final int N0 = 4096;
    private static final int N1 = 2048;
    private static final int N2 = 2048;
    private static final int TYPE_Q4_K = 12;
    private static final int TYPE_Q6_K = 14;
    private static final int MAX_BATCH = 15;
    private static final int WARMUP_ITERATIONS = 3;
    private static final int MEASURED_ITERATIONS = 12;
    private static final int[] BATCH_SIZES = {1, 4, 8, 15};

    @Test
    void reportsWarmTripleBatchLatencyAndThroughput() throws IOException {
        assumeTrue(benchmarkEnabled(),
                "Set ALJABR_METAL_BENCH=1 to run the native Metal performance benchmark");

        assertTrue(MetalBinding.initialize(),
                "Could not load the current Aljabr Metal library; rebuild/install libaljabr_metal.dylib");
        MetalBinding binding = MetalBinding.getInstance();
        assertEquals(0, binding.init(), "Metal runtime initialization returned an error");
        assertTrue(binding.isRuntimeActive(),
                "Metal runtime is not active; refusing to benchmark a CPU fallback");

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate((long) MAX_BATCH * K * Float.BYTES, 64);
            MemorySegment output0 = arena.allocate((long) MAX_BATCH * N0 * Float.BYTES, 64);
            MemorySegment output1 = arena.allocate((long) MAX_BATCH * N1 * Float.BYTES, 64);
            MemorySegment output2 = arena.allocate((long) MAX_BATCH * N2 * Float.BYTES, 64);

            // Non-zero input and valid byte patterns prevent benchmarking only a
            // degenerate all-zero input while keeping fixture setup deterministic.
            int inputElements = MAX_BATCH * K;
            for (int i = 0; i < inputElements; i++) {
                input.setAtIndex(ValueLayout.JAVA_FLOAT, i, ((i % 31) - 15) / 31.0f);
            }

            MemorySegment weights0 = arena.allocate(q4KBytes(N0, K), 64);
            MemorySegment weights1 = arena.allocate(q4KBytes(N1, K), 64);
            MemorySegment weights2 = arena.allocate(q6KBytes(N2, K), 64);
            weights0.fill((byte) 0x55);
            weights1.fill((byte) 0x35);
            weights2.fill((byte) 0x45);

            System.out.printf(Locale.ROOT,
                    "ALJABR_METAL_TRIPLE_BENCH shape=K:%d,N:%d/%d/%d types:%d/%d/%d warmup:%d measured:%d%n",
                    K, N0, N1, N2, TYPE_Q4_K, TYPE_Q4_K, TYPE_Q6_K,
                    WARMUP_ITERATIONS, MEASURED_ITERATIONS);
            String header = "batch,mean_ms,median_ms,p90_ms,mean_ms_per_item,items_per_second";
            System.out.println(header);
            StringBuilder csv = new StringBuilder(header).append(System.lineSeparator());

            for (int batchSize : BATCH_SIZES) {
                for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                    invoke(binding, output0, output1, output2, input, weights0,
                            weights1, weights2, batchSize);
                }

                long[] samples = new long[MEASURED_ITERATIONS];
                long totalNanos = 0L;
                for (int i = 0; i < MEASURED_ITERATIONS; i++) {
                    long start = System.nanoTime();
                    int rc = invoke(binding, output0, output1, output2, input, weights0,
                            weights1, weights2, batchSize);
                    long elapsed = System.nanoTime() - start;
                    assertEquals(0, rc, "Native triple-batch call failed for batch=" + batchSize);
                    samples[i] = elapsed;
                    totalNanos += elapsed;
                }

                Arrays.sort(samples);
                double meanMs = totalNanos / (double) MEASURED_ITERATIONS / 1_000_000.0;
                double medianMs = percentileNanos(samples, 0.50) / 1_000_000.0;
                double p90Ms = percentileNanos(samples, 0.90) / 1_000_000.0;
                double perItemMs = meanMs / batchSize;
                double itemsPerSecond = batchSize * MEASURED_ITERATIONS * 1_000_000_000.0
                        / totalNanos;

                String row = String.format(Locale.ROOT, "%d,%.3f,%.3f,%.3f,%.3f,%.1f",
                        batchSize, meanMs, medianMs, p90Ms, perItemMs, itemsPerSecond);
                System.out.println(row);
                csv.append(row).append(System.lineSeparator());
            }

            Path reportPath = repositoryRoot().resolve(
                    "backend/metal/aljabr-backend-metal/build/reports/metal-triple-batch-benchmark.csv");
            Files.createDirectories(reportPath.getParent());
            Files.writeString(reportPath, csv.toString(), StandardCharsets.UTF_8);
            System.err.println("Aljabr Metal benchmark CSV: " + reportPath);
        }
    }

    private static Path repositoryRoot() {
        Path current = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
        for (Path candidate = current; candidate != null; candidate = candidate.getParent()) {
            if (Files.isRegularFile(candidate.resolve("settings.gradle.kts"))
                    && Files.isRegularFile(candidate.resolve("gradlew"))) {
                return candidate;
            }
        }
        throw new IllegalStateException("Could not find the Aljabr repository root from "
                + current);
    }

    private static int invoke(
            MetalBinding binding,
            MemorySegment output0, MemorySegment output1, MemorySegment output2,
            MemorySegment input, MemorySegment weights0, MemorySegment weights1,
            MemorySegment weights2, int batchSize) {
        return binding.matvecTransposedRightTripleBatch(
                output0, output1, output2, input, weights0, weights1, weights2,
                K, N0, N1, N2, TYPE_Q4_K, TYPE_Q4_K, TYPE_Q6_K, batchSize);
    }

    private static long q4KBytes(int n, int k) {
        // GGML Q4_K: 144 bytes per block of 256 elements.
        return (long) n * k / 256 * 144;
    }

    private static long q6KBytes(int n, int k) {
        // GGML Q6_K: 210 bytes per block of 256 elements.
        return (long) n * k / 256 * 210;
    }

    private static long percentileNanos(long[] sortedSamples, double quantile) {
        int index = (int) Math.ceil(quantile * sortedSamples.length) - 1;
        index = Math.max(0, Math.min(index, sortedSamples.length - 1));
        return sortedSamples[index];
    }

    private static boolean benchmarkEnabled() {
        String value = System.getenv("ALJABR_METAL_BENCH");
        return value != null && ("1".equals(value) || "true".equalsIgnoreCase(value));
    }
}
