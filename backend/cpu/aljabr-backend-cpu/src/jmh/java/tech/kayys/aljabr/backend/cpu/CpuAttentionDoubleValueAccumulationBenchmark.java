package tech.kayys.aljabr.backend.cpu;

import java.util.Arrays;
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
 * Compares scalar and SIMD double accumulation, matching the production
 * rank-4 attention accumulator precision. Input values are promoted from F32
 * once during setup; MemorySegment access and conversion costs are excluded.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 8, time = 1)
@Fork(2)
public class CpuAttentionDoubleValueAccumulationBenchmark {
    @Param({"64", "80", "96", "128"})
    public int headDimension;

    @Param({"64", "256", "1024"})
    public int keyCount;

    private double[] values;
    private double[] previousScales;
    private double[] currentScales;
    private double[] scalarAccumulator;
    private double[] vectorAccumulator;

    @Setup(Level.Trial)
    public void setUp() {
        values = new double[Math.multiplyExact(keyCount, headDimension)];
        previousScales = new double[keyCount];
        currentScales = new double[keyCount];
        for (int key = 0; key < keyCount; key++) {
            previousScales[key] = 0.997d + (key % 3) * 0.001d;
            currentScales[key] = 0.001d + (key % 5) * 0.0001d;
            for (int d = 0; d < headDimension; d++) {
                // Preserve F32 source-value precision, then promote as production does.
                values[key * headDimension + d] =
                        (float) Math.sin((key + 1) * (d + 1) * 0.013);
            }
        }
        scalarAccumulator = new double[headDimension];
        vectorAccumulator = new double[headDimension];
    }

    @Setup(Level.Invocation)
    public void resetAccumulators() {
        Arrays.fill(scalarAccumulator, 0d);
        Arrays.fill(vectorAccumulator, 0d);
    }

    @Benchmark
    public void scalarDoubleWeightedValueAccumulation() {
        for (int key = 0; key < keyCount; key++) {
            CpuAttentionDoubleValueOps.scalar(scalarAccumulator, values,
                    key * headDimension, previousScales[key], currentScales[key], headDimension);
        }
    }

    @Benchmark
    public void vectorizedDoubleWeightedValueAccumulation() {
        for (int key = 0; key < keyCount; key++) {
            CpuAttentionDoubleValueOps.vectorized(vectorAccumulator, values,
                    key * headDimension, previousScales[key], currentScales[key], headDimension);
        }
    }
}
