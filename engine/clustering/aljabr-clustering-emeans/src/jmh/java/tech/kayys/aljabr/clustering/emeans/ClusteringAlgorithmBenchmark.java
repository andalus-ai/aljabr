package tech.kayys.aljabr.clustering.emeans;

import java.util.SplittableRandom;
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

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.clustering.api.ClusteringDataset;
import tech.kayys.aljabr.clustering.api.ClusteringResult;
import tech.kayys.aljabr.clustering.kmeans.KMeansAlgorithm;
import tech.kayys.aljabr.clustering.kmeans.KMeansConfig;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * End-to-end CPU comparison of e-means and Lloyd k-means.
 *
 * <p>Both methods receive the same deterministic dataset and random-row
 * initialization seed. The benchmark includes fitting and result allocation,
 * but excludes one-time dataset creation. Run with
 * {@code ./gradlew :engine:clustering:aljabr-clustering-emeans:jmh}.</p>
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
public class ClusteringAlgorithmBenchmark {
    @Param({"10000", "50000"})
    public int sampleCount;

    @Param({"32", "128"})
    public int dimensions;

    @Param({"32", "128"})
    public int clusterCount;

    @Param({"4"})
    public int epochs;

    private CpuBackend backend;
    private Tensor vectors;
    private ClusteringDataset dataset;
    private EMeansConfig eMeansConfig;
    private KMeansConfig kMeansConfig;
    private final EMeansAlgorithm eMeans = new EMeansAlgorithm();
    private final KMeansAlgorithm kMeans = new KMeansAlgorithm();

    @Setup(Level.Trial)
    public void setUp() {
        backend = new CpuBackend();
        float[] values = new float[Math.multiplyExact(sampleCount, dimensions)];
        SplittableRandom random = new SplittableRandom(0x5EEDL);
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) random.nextGaussian();
        }
        vectors = TensorFactory.of(values, backend, sampleCount, dimensions);
        dataset = new ClusteringDataset(vectors);

        var recommendation = EMeansParameterScaling.recommend(
                sampleCount, clusterCount, epochs, dimensions);
        eMeansConfig = new EMeansConfig(
                clusterCount, 0x5EEDL, epochs,
                recommendation.batchSize(), recommendation.initialStepSize(), true);
        kMeansConfig = new KMeansConfig(
                clusterCount, 0x5EEDL,
                tech.kayys.aljabr.clustering.api.DistanceMetric.SQUARED_EUCLIDEAN,
                epochs, 0.0,
                tech.kayys.aljabr.clustering.kmeans.KMeansInitialization.RANDOM);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        if (vectors != null) {
            vectors.close();
        }
        if (backend != null) {
            backend.close();
        }
    }

    @Benchmark
    public double eMeans() {
        try (ClusteringResult result = eMeans.fit(dataset, eMeansConfig)) {
            return result.metrics().objectiveValue();
        }
    }

    @Benchmark
    public double lloydKMeans() {
        try (ClusteringResult result = kMeans.fit(dataset, kMeansConfig)) {
            return result.metrics().objectiveValue();
        }
    }
}
