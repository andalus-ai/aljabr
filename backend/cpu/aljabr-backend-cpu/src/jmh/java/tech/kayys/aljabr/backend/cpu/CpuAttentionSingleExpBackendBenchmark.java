
package tech.kayys.aljabr.backend.cpu;

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

import tech.kayys.aljabr.core.memory.CpuBuffer;
import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * End-to-end rank-4 CPU attention comparison for the current two-exp
 * online-softmax recurrence and a single-exp candidate.
 *
 * <p>Both paths use identical reusable F32 inputs, SIMD Q.K dot products,
 * weighted-value accumulation, output storage allocation, and complete
 * batch/head/query loops. The candidate differs only in the online-softmax
 * recurrence. This benchmark intentionally covers the unmasked, non-causal
 * overload used by the existing CpuAttentionBenchmark.</p>
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 8, time = 1)
@Fork(2)
public class CpuAttentionSingleExpBackendBenchmark {

    @Param({"1", "8"})
    public int batchHeads;

    @Param({"64", "256", "1024"})
    public int sequenceLength;

    @Param({"64", "80", "96", "128"})
    public int headDimension;

    private CpuBackend backend;
    private Tensor query;
    private Tensor key;
    private Tensor value;

    @Setup(Level.Trial)
    public void setUp() {
        backend = new CpuBackend();
        query = TensorFactory.randn(backend, 1, batchHeads, sequenceLength, headDimension);
        key = TensorFactory.randn(backend, 1, batchHeads, sequenceLength, headDimension);
        value = TensorFactory.randn(backend, 1, batchHeads, sequenceLength, headDimension);

        Tensor baseline = backend.attention(query, key, value);
        Tensor candidate = singleExpAttention();
        try {
            float[] expected = baseline.toFloatArray();
            float[] actual = candidate.toFloatArray();
            for (int i = 0; i < expected.length; i++) {
                float tolerance = Math.max(2.0e-6f, Math.abs(expected[i]) * 2.0e-5f);
                if (Math.abs(expected[i] - actual[i]) > tolerance) {
                    throw new IllegalStateException("Attention output differs at flat index "
                            + i + ": " + expected[i] + " vs " + actual[i]);
                }
            }
        } finally {
            candidate.release();
            baseline.release();
        }
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        if (value != null) value.release();
        if (key != null) key.release();
        if (query != null) query.release();
        if (backend != null) backend.close();
    }

    @Benchmark
    public double productionTwoExpAttention() {
        Tensor output = backend.attention(query, key, value);
        try {
            return checksum(output);
        } finally {
            output.release();
        }
    }

    @Benchmark
    public double candidateSingleExpAttention() {
        Tensor output = singleExpAttention();
        try {
            return checksum(output);
        } finally {
            output.release();
        }
    }

    private Tensor singleExpAttention() {
        long batch = 1;
        long heads = batchHeads;
        long queryLength = sequenceLength;
        long keyLength = sequenceLength;
        long headDim = headDimension;
        Shape outputShape = new Shape(batch, heads, queryLength, headDim);
        CpuBuffer output = new CpuBuffer(Math.multiplyExact(outputShape.numel(), Float.BYTES));
        boolean transferred = false;
        try {
            MemorySegment q = segment(query);
            MemorySegment k = segment(key);
            MemorySegment v = segment(value);
            MemorySegment out = output.segment();
            float scale = (float) (1.0 / Math.sqrt(headDim));

            for (long b = 0; b < batch; b++) {
                for (long h = 0; h < heads; h++) {
                    long qBase = (b * heads + h) * queryLength * headDim;
                    long kBase = (b * heads + h) * keyLength * headDim;
                    double[] weightedValues = new double[Math.toIntExact(headDim)];

                    for (long qi = 0; qi < queryLength; qi++) {
                        java.util.Arrays.fill(weightedValues, 0d);
                        float runningMax = Float.NEGATIVE_INFINITY;
                        double denominator = 0d;

                        for (long ki = 0; ki < keyLength; ki++) {
                            float score = CpuAttentionOps.dot(
                                    q, (qBase + qi * headDim) * Float.BYTES,
                                    k, (kBase + ki * headDim) * Float.BYTES,
                                    Math.toIntExact(headDim)) * scale;

                            double previousScale;
                            double currentScale;
                            if (!Float.isFinite(score)) {
                                float nextMax = Math.max(runningMax, score);
                                previousScale = runningMax == Float.NEGATIVE_INFINITY
                                        ? 0d : Math.exp(runningMax - nextMax);
                                currentScale = Math.exp(score - nextMax);
                                denominator = denominator * previousScale + currentScale;
                                runningMax = nextMax;
                            } else if (score > runningMax) {
                                previousScale = Math.exp(runningMax - score);
                                currentScale = 1d;
                                denominator = denominator * previousScale + 1d;
                                runningMax = score;
                            } else {
                                previousScale = 1d;
                                currentScale = Math.exp(score - runningMax);
                                denominator += currentScale;
                            }

                            for (long d = 0; d < headDim; d++) {
                                float valueAtKey = v.get(ValueLayout.JAVA_FLOAT,
                                        (kBase + ki * headDim + d) * Float.BYTES);
                                weightedValues[Math.toIntExact(d)] =
                                        weightedValues[Math.toIntExact(d)] * previousScale
                                                + currentScale * valueAtKey;
                            }
                        }

                        for (long d = 0; d < headDim; d++) {
                            float result = denominator == 0d
                                    ? 0f
                                    : (float) (weightedValues[Math.toIntExact(d)] / denominator);
                            out.set(ValueLayout.JAVA_FLOAT,
                                    (qBase + qi * headDim + d) * Float.BYTES, result);
                        }
                    }
                }
            }

            Tensor result = new DefaultTensor(outputShape, DType.F32, DeviceType.CPU, output, backend);
            transferred = true;
            return result;
        } finally {
            if (!transferred) output.release();
        }
    }

    private static MemorySegment segment(Tensor tensor) {
        return ((HostAccessibleStorage) tensor.storage()).segment();
    }

    private static double checksum(Tensor tensor) {
        MemorySegment data = segment(tensor);
        double checksum = 0d;
        for (long i = 0; i < tensor.numel(); i++) {
            checksum += data.get(ValueLayout.JAVA_FLOAT, i * Float.BYTES) * ((i % 31) + 1);
        }
        return checksum;
    }
}
