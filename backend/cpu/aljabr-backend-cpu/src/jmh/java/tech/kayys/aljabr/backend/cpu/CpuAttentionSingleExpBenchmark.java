package tech.kayys.aljabr.backend.cpu;

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
import org.openjdk.jmh.annotations.Warmup;

/**
 * Measures whether the online-softmax recurrence can avoid evaluating the
 * exponential whose argument is necessarily zero.
 *
 * <p>The baseline mirrors the two-exponential recurrence currently used by
 * production rank-4 attention. The single-exponential candidate uses the
 * ordering of score and running maximum to replace one exponential with the
 * exact constant 1.0. Both variants consume the same precomputed scores.</p>
 *
 * <p>This benchmark isolates denominator arithmetic. It does not measure the
 * full attention operation or justify a production change by itself.</p>
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 8, time = 1)
@Fork(2)
public class CpuAttentionSingleExpBenchmark {

    @Param({"64", "256", "1024", "4096"})
    public int keyCount;

    private float[] scores;

    @Setup(Level.Trial)
    public void setUp() {
        scores = new float[keyCount];
        for (int i = 0; i < keyCount; i++) {
            // Deterministic, non-constant logits with a bounded dynamic range.
            scores[i] = (float) (Math.sin((i + 1) * 0.017) * 8.0
                    + Math.cos((i + 1) * 0.0031) * 2.0);
        }

        double baseline = twoExpOnlineDenominator();
        double candidate = singleExpOnlineDenominator();
        double tolerance = Math.max(1.0e-12, Math.abs(baseline) * 1.0e-12);
        if (Math.abs(baseline - candidate) > tolerance) {
            throw new IllegalStateException(
                    "Single-exponential recurrence differs from baseline: "
                            + baseline + " vs " + candidate);
        }
    }

    /** Mirrors the current production recurrence, including both exp calls. */
    @Benchmark
    public double twoExpOnlineDenominator() {
        float runningMax = Float.NEGATIVE_INFINITY;
        double denominator = 0d;
        for (float score : scores) {
            float nextMax = Math.max(runningMax, score);
            double previousScale = runningMax == Float.NEGATIVE_INFINITY
                    ? 0d : Math.exp(runningMax - nextMax);
            double currentScale = Math.exp(score - nextMax);
            denominator = denominator * previousScale + currentScale;
            runningMax = nextMax;
        }
        return denominator;
    }

    /**
     * Uses one exponential per score: whichever side of the running maximum
     * is already at the new maximum contributes an exact scale of 1.0.
     */
    @Benchmark
    public double singleExpOnlineDenominator() {
        float runningMax = Float.NEGATIVE_INFINITY;
        double denominator = 0d;
        for (float score : scores) {
            if (score > runningMax) {
                double previousScale = Math.exp(runningMax - score);
                denominator = denominator * previousScale + 1d;
                runningMax = score;
            } else {
                denominator += Math.exp(score - runningMax);
            }
        }
        return denominator;
    }
}
