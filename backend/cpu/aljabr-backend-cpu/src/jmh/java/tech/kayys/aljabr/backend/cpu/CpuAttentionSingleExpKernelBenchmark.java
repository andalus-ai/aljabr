package tech.kayys.aljabr.backend.cpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Compares online-softmax recurrences inside a complete single-query,
 * single-head attention kernel: SIMD Q.K, weighted-value accumulation,
 * and output normalization.
 *
 * <p>This is a focused kernel benchmark, not a replacement for the
 * end-to-end CpuBackend attention benchmark. Inputs are reused and stored
 * off-heap, matching the production CPU backend's MemorySegment access.</p>
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 8, time = 1)
@Fork(2)
public class CpuAttentionSingleExpKernelBenchmark {

    @Param({"64", "256", "1024"})
    public int keyCount;

    @Param({"64", "80", "96", "128"})
    public int headDimension;

    private Arena arena;
    private MemorySegment query;
    private MemorySegment keys;
    private MemorySegment values;
    private float scale;

    @Setup(Level.Trial)
    public void setUp() {
        arena = Arena.ofShared();
        query = arena.allocate((long) headDimension * Float.BYTES, Float.BYTES);
        keys = arena.allocate((long) keyCount * headDimension * Float.BYTES, Float.BYTES);
        values = arena.allocate((long) keyCount * headDimension * Float.BYTES, Float.BYTES);
        scale = (float) (1.0 / Math.sqrt(headDimension));

        for (int d = 0; d < headDimension; d++) {
            query.set(ValueLayout.JAVA_FLOAT, (long) d * Float.BYTES,
                    (float) (Math.sin((d + 1) * 0.13) * 0.5));
        }
        for (int key = 0; key < keyCount; key++) {
            for (int d = 0; d < headDimension; d++) {
                long offset = ((long) key * headDimension + d) * Float.BYTES;
                keys.set(ValueLayout.JAVA_FLOAT, offset,
                        (float) (Math.sin((key + 1) * (d + 3) * 0.0017) * 0.7));
                values.set(ValueLayout.JAVA_FLOAT, offset,
                        (float) (Math.cos((key + 2) * (d + 1) * 0.0023) * 0.8));
            }
        }

        double[] baseline = runTwoExp();
        double[] candidate = runSingleExp();
        for (int d = 0; d < headDimension; d++) {
            double tolerance = Math.max(1.0e-10, Math.abs(baseline[d]) * 2.0e-6);
            if (Math.abs(baseline[d] - candidate[d]) > tolerance) {
                throw new IllegalStateException("Attention output differs at dimension "
                        + d + ": " + baseline[d] + " vs " + candidate[d]);
            }
        }
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        if (arena != null) {
            arena.close();
        }
    }

    @Benchmark
    public double twoExpOnlineAttentionKernel() {
        return checksum(runTwoExp());
    }

    @Benchmark
    public double singleExpOnlineAttentionKernel() {
        return checksum(runSingleExp());
    }

    private double[] runTwoExp() {
        double[] weightedValues = new double[headDimension];
        float runningMax = Float.NEGATIVE_INFINITY;
        double denominator = 0d;

        for (int key = 0; key < keyCount; key++) {
            long keyOffset = (long) key * headDimension * Float.BYTES;
            float score = CpuAttentionOps.dot(query, 0, keys, keyOffset, headDimension) * scale;

            float nextMax = Math.max(runningMax, score);
            double previousScale = runningMax == Float.NEGATIVE_INFINITY
                    ? 0d : Math.exp(runningMax - nextMax);
            double currentScale = Math.exp(score - nextMax);
            denominator = denominator * previousScale + currentScale;

            for (int d = 0; d < headDimension; d++) {
                float value = values.get(ValueLayout.JAVA_FLOAT,
                        keyOffset + (long) d * Float.BYTES);
                weightedValues[d] = weightedValues[d] * previousScale + currentScale * value;
            }
            runningMax = nextMax;
        }

        normalize(weightedValues, denominator);
        return weightedValues;
    }

    private double[] runSingleExp() {
        double[] weightedValues = new double[headDimension];
        float runningMax = Float.NEGATIVE_INFINITY;
        double denominator = 0d;

        for (int key = 0; key < keyCount; key++) {
            long keyOffset = (long) key * headDimension * Float.BYTES;
            float score = CpuAttentionOps.dot(query, 0, keys, keyOffset, headDimension) * scale;

            // Preserve the baseline's exceptional-value behavior. The fast
            // recurrence is used only for finite logits.
            if (!Float.isFinite(score)) {
                float nextMax = Math.max(runningMax, score);
                double previousScale = runningMax == Float.NEGATIVE_INFINITY
                        ? 0d : Math.exp(runningMax - nextMax);
                double currentScale = Math.exp(score - nextMax);
                denominator = denominator * previousScale + currentScale;
                for (int d = 0; d < headDimension; d++) {
                    float value = values.get(ValueLayout.JAVA_FLOAT,
                            keyOffset + (long) d * Float.BYTES);
                    weightedValues[d] = weightedValues[d] * previousScale + currentScale * value;
                }
                runningMax = nextMax;
                continue;
            }

            double previousScale;
            double currentScale;
            if (score > runningMax) {
                previousScale = Math.exp(runningMax - score);
                currentScale = 1d;
                denominator = denominator * previousScale + 1d;
                runningMax = score;
            } else {
                previousScale = 1d;
                currentScale = Math.exp(score - runningMax);
                denominator += currentScale;
            }

            for (int d = 0; d < headDimension; d++) {
                float value = values.get(ValueLayout.JAVA_FLOAT,
                        keyOffset + (long) d * Float.BYTES);
                weightedValues[d] = weightedValues[d] * previousScale + currentScale * value;
            }
        }

        normalize(weightedValues, denominator);
        return weightedValues;
    }

    private static void normalize(double[] values, double denominator) {
        if (denominator == 0d) {
            java.util.Arrays.fill(values, 0d);
            return;
        }
        for (int d = 0; d < values.length; d++) {
            values[d] /= denominator;
        }
    }

    private static double checksum(double[] values) {
        double result = 0d;
        for (int d = 0; d < values.length; d++) {
            result += values[d] * (d + 1);
        }
        return result;
    }
}
