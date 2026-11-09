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

/** Isolates scalar and Vector API weighted-value accumulation for attention. */
@State(Scope.Thread)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 8, time = 1)
@Fork(2)
public class CpuAttentionValueAccumulationBenchmark {
    @Param({"64", "80", "96", "128"})
    public int headDimension;

    @Param({"64", "256", "1024"})
    public int keyCount;

    private float[] values;
    private float[] previousScales;
    private float[] currentScales;
    private float[] scalarAccumulator;
    private float[] vectorAccumulator;

    @Setup(Level.Trial)
    public void setUp() {
        values = new float[Math.multiplyExact(keyCount, headDimension)];
        previousScales = new float[keyCount];
        currentScales = new float[keyCount];
        for (int key = 0; key < keyCount; key++) {
            previousScales[key] = 0.997f + (key % 3) * 0.001f;
            currentScales[key] = 0.001f + (key % 5) * 0.0001f;
            for (int d = 0; d < headDimension; d++) {
                values[key * headDimension + d] =
                        (float) Math.sin((key + 1) * (d + 1) * 0.013);
            }
        }
        scalarAccumulator = new float[headDimension];
        vectorAccumulator = new float[headDimension];
    }

    @Setup(Level.Invocation)
    public void resetAccumulators() {
        java.util.Arrays.fill(scalarAccumulator, 0f);
        java.util.Arrays.fill(vectorAccumulator, 0f);
    }

    @Benchmark
    public void scalarWeightedValueAccumulation() {
        for (int key = 0; key < keyCount; key++) {
            CpuAttentionValueOps.scalar(scalarAccumulator, values,
                    key * headDimension, previousScales[key], currentScales[key], headDimension);
        }
    }

    @Benchmark
    public void vectorizedWeightedValueAccumulation() {
        for (int key = 0; key < keyCount; key++) {
            CpuAttentionValueOps.vectorized(vectorAccumulator, values,
                    key * headDimension, previousScales[key], currentScales[key], headDimension);
        }
    }
}
