package tech.kayys.aljabr.backend.cpu;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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

import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * Compares an explicit serial reference, the production attention path, and a
 * benchmark-only candidate that partitions independent batch/head groups
 * across a persistent executor. The candidate deliberately includes task
 * submission and output allocation in each measured invocation.
 *
 * <p>The serial reference remains independent of production scheduling, so
 * this benchmark stays meaningful after production attention is parallelized.</p>
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 8, time = 1)
@Fork(2)
public class CpuAttentionParallelismBenchmark {

    @Param({"1", "8"})
    public int batchHeads;

    @Param({"256", "1024"})
    public int sequenceLength;

    @Param({"64", "128"})
    public int headDimension;

    @Param({"1", "2", "4", "8"})
    public int workerCount;

    private CpuBackend backend;
    private Tensor query;
    private Tensor key;
    private Tensor value;
    private ExecutorService executor;

    @Setup(Level.Trial)
    public void setUp() {
        backend = new CpuBackend();
        query = TensorFactory.randn(backend, 1, batchHeads, sequenceLength, headDimension);
        key = TensorFactory.randn(backend, 1, batchHeads, sequenceLength, headDimension);
        value = TensorFactory.randn(backend, 1, batchHeads, sequenceLength, headDimension);
        executor = Executors.newFixedThreadPool(workerCount);
        assertEquivalentOutputs();
    }

    private void assertEquivalentOutputs() {
        Tensor expected = serialReference();
        Tensor production = backend.attention(query, key, value);
        Tensor candidate = parallelCandidate();
        try {
            assertEquivalent(expected, production, "production");
            assertEquivalent(expected, candidate, "parallel candidate");
        } finally {
            candidate.release();
            production.release();
            expected.release();
        }
    }

    private static void assertEquivalent(Tensor expected, Tensor actual, String label) {
        MemorySegment expectedSegment = CpuTensorStorageAccess.segment(expected);
        MemorySegment actualSegment = CpuTensorStorageAccess.segment(actual);
        long elements = expected.numel();
        for (long i = 0; i < elements; i++) {
            float a = expectedSegment.get(ValueLayout.JAVA_FLOAT, i * Float.BYTES);
            float b = actualSegment.get(ValueLayout.JAVA_FLOAT, i * Float.BYTES);
            float tolerance = Math.max(2.0e-6f, Math.abs(a) * 2.0e-5f);
            if (!(Math.abs(a - b) <= tolerance)) {
                throw new IllegalStateException(
                        label + " attention mismatch at element " + i
                                + ": serial=" + a + ", actual=" + b);
            }
        }
    }

    @TearDown(Level.Trial)
    public void tearDown() throws InterruptedException {
        executor.shutdown();
        if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
            executor.shutdownNow();
        }
        if (value != null) value.release();
        if (key != null) key.release();
        if (query != null) query.release();
        if (backend != null) backend.close();
    }

    @Benchmark
    public void serialReferenceAttention() {
        Tensor output = serialReference();
        output.release();
    }

    @Benchmark
    public void productionBackendAttention() {
        Tensor output = backend.attention(query, key, value);
        output.release();
    }

    @Benchmark
    public void parallelCandidateAttention() {
        Tensor output = parallelCandidate();
        output.release();
    }

    private Tensor serialReference() {
        long queryLength = sequenceLength;
        long keyLength = sequenceLength;
        long headDim = headDimension;
        long totalElements = (long) batchHeads * queryLength * headDim;
        HostAccessibleStorage outputStorage = (HostAccessibleStorage) backend.allocateStorage(
                Math.multiplyExact(totalElements, Float.BYTES));
        MemorySegment q = CpuTensorStorageAccess.segment(query);
        MemorySegment k = CpuTensorStorageAccess.segment(key);
        MemorySegment v = CpuTensorStorageAccess.segment(value);
        MemorySegment out = outputStorage.segment();
        float scale = (float) (1.0 / Math.sqrt(headDim));

        try {
            for (int group = 0; group < batchHeads; group++) {
                computeGroup(q, k, v, out, group, queryLength, keyLength, headDim, scale);
            }
            return new DefaultTensor(
                    new Shape(1, batchHeads, sequenceLength, headDimension),
                    DType.F32,
                    DeviceType.CPU,
                    outputStorage,
                    backend);
        } catch (RuntimeException | Error failure) {
            outputStorage.release();
            throw failure;
        }
    }

    private Tensor parallelCandidate() {
        long queryLength = sequenceLength;
        long keyLength = sequenceLength;
        long headDim = headDimension;
        long totalElements = (long) batchHeads * queryLength * headDim;
        HostAccessibleStorage outputStorage = (HostAccessibleStorage) backend.allocateStorage(
                Math.multiplyExact(totalElements, Float.BYTES));
        MemorySegment q = CpuTensorStorageAccess.segment(query);
        MemorySegment k = CpuTensorStorageAccess.segment(key);
        MemorySegment v = CpuTensorStorageAccess.segment(value);
        MemorySegment out = outputStorage.segment();
        float scale = (float) (1.0 / Math.sqrt(headDim));

        List<Callable<Void>> tasks = new ArrayList<>(batchHeads);
        for (int group = 0; group < batchHeads; group++) {
            final int groupIndex = group;
            tasks.add(() -> {
                computeGroup(q, k, v, out, groupIndex, queryLength, keyLength, headDim, scale);
                return null;
            });
        }

        try {
            List<Future<Void>> futures = executor.invokeAll(tasks);
            for (Future<Void> future : futures) {
                future.get();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            outputStorage.release();
            throw new IllegalStateException("Interrupted while executing parallel attention", e);
        } catch (ExecutionException e) {
            outputStorage.release();
            throw new IllegalStateException("Parallel attention worker failed", e.getCause());
        } catch (RuntimeException | Error e) {
            outputStorage.release();
            throw e;
        }
        return new DefaultTensor(
                new Shape(1, batchHeads, sequenceLength, headDimension),
                DType.F32,
                DeviceType.CPU,
                outputStorage,
                backend);
    }

    private static void computeGroup(
            MemorySegment q,
            MemorySegment k,
            MemorySegment v,
            MemorySegment out,
            int group,
            long queryLength,
            long keyLength,
            long headDim,
            float scale) {
        long qBase = (long) group * queryLength * headDim;
        long kBase = (long) group * keyLength * headDim;
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

                float nextMax;
                double previousScale;
                double currentScale;
                if (Float.isFinite(score)) {
                    if (score > runningMax) {
                        previousScale = Math.exp(runningMax - score);
                        currentScale = 1d;
                        nextMax = score;
                    } else {
                        previousScale = 1d;
                        currentScale = Math.exp(score - runningMax);
                        nextMax = runningMax;
                    }
                } else {
                    nextMax = Math.max(runningMax, score);
                    previousScale = runningMax == Float.NEGATIVE_INFINITY
                            ? 0d : Math.exp(runningMax - nextMax);
                    currentScale = Math.exp(score - nextMax);
                }
                denominator = denominator * previousScale + currentScale;

                for (long d = 0; d < headDim; d++) {
                    float valueAtKey = v.get(ValueLayout.JAVA_FLOAT,
                            (kBase + ki * headDim + d) * Float.BYTES);
                    int index = Math.toIntExact(d);
                    weightedValues[index] = weightedValues[index] * previousScale
                            + currentScale * valueAtKey;
                }
                runningMax = nextMax;
            }

            for (long d = 0; d < headDim; d++) {
                float result = denominator == 0d
                        ? 0f : (float) (weightedValues[Math.toIntExact(d)] / denominator);
                out.set(ValueLayout.JAVA_FLOAT,
                        (qBase + qi * headDim + d) * Float.BYTES, result);
            }
        }
    }
}
