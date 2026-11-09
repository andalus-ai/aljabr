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

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * Isolates the scalar and SIMD attention dot-product primitives.
 *
 * <p>Both benchmarks read the same contiguous off-heap inputs. Only the head
 * dimension varies, avoiding unrelated batch-head and sequence-length
 * parameters from the end-to-end attention benchmark.</p>
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 8, time = 1)
@Fork(2)
public class CpuAttentionDotProductBenchmark {

    @Param({"64", "80", "96", "128"})
    public int headDimension;

    private CpuBackend backend;
    private Tensor query;
    private Tensor key;
    private MemorySegment querySegment;
    private MemorySegment keySegment;

    @Setup(Level.Trial)
    public void setUp() {
        backend = new CpuBackend();
        query = TensorFactory.randn(backend, 1, 1, 1, headDimension);
        key = TensorFactory.randn(backend, 1, 1, 1, headDimension);
        querySegment = CpuTensorStorageAccess.segment(query);
        keySegment = CpuTensorStorageAccess.segment(key);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        if (key != null) key.release();
        if (query != null) query.release();
        if (backend != null) backend.close();
    }

    /** Measures the Vector API implementation. */
    @Benchmark
    public float vectorizedDotProduct() {
        return CpuAttentionOps.dot(querySegment, 0, keySegment, 0, headDimension);
    }

    /** Scalar reference over the same input segments. */
    @Benchmark
    public float scalarDotProduct() {
        float result = 0f;
        for (int i = 0; i < headDimension; i++) {
            long offset = (long) i * Float.BYTES;
            float left = querySegment.get(ValueLayout.JAVA_FLOAT, offset);
            float right = keySegment.get(ValueLayout.JAVA_FLOAT, offset);
            result += left * right;
        }
        return result;
    }
}
