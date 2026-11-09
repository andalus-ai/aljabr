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
 * Isolates the normalization arithmetic used by rank-4 online-softmax attention.
 *
 * <p>The online variant matches the running-maximum/denominator recurrence used
 * by the CPU attention kernel. The two-pass reference first finds the maximum
 * and then sums exponentials. Scores are prepared once per trial, so this
 * benchmark compares normalization arithmetic rather than score generation,
 * tensor access, or value accumulation.</p>
 *
 * <p>The two-pass variant is a diagnostic baseline, not a proposed production
 * replacement: using it in attention would require retaining or recomputing
 * scores before the value-weighted reduction.</p>
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 8, time = 1)
@Fork(2)
public class CpuAttentionSoftmaxBenchmark {

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
    }

    /** Running-max and running-denominator recurrence used by production attention. */
    @Benchmark
    public double onlineSoftmaxDenominator() {
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

    /** Stable two-pass denominator reference; scores are already materialized. */
    @Benchmark
    public double twoPassSoftmaxDenominator() {
        float maximum = Float.NEGATIVE_INFINITY;
        for (float score : scores) {
            maximum = Math.max(maximum, score);
        }

        double denominator = 0d;
        for (float score : scores) {
            denominator += Math.exp(score - maximum);
        }
        return denominator;
    }
}
