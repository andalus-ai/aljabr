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
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * Reproducible baseline for rank-4 CPU scaled dot-product attention.
 *
 * <p>Measures the complete backend operation, including output allocation.
 * Input tensors are allocated once per benchmark thread and reused across
 * invocations. Run with {@code ./gradlew :backend:cpu:aljabr-backend-cpu:jmh}.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 8, time = 1)
@Fork(2)
public class CpuAttentionBenchmark {

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
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        if (value != null) value.release();
        if (key != null) key.release();
        if (query != null) query.release();
        if (backend != null) backend.close();
    }

    @Benchmark
    public void scaledDotProductAttention() {
        Tensor output = backend.attention(query, key, value);
        output.release();
    }
}
