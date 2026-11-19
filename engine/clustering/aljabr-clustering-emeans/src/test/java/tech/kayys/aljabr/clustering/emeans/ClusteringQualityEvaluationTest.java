package tech.kayys.aljabr.clustering.emeans;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.clustering.api.ClusteringDataset;
import tech.kayys.aljabr.clustering.api.ClusteringResult;
import tech.kayys.aljabr.clustering.kmeans.KMeansAlgorithm;
import tech.kayys.aljabr.clustering.kmeans.KMeansConfig;
import tech.kayys.aljabr.clustering.kmeans.KMeansInitialization;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * Deterministic held-out quality harness for comparing Lloyd and e-means
 * finalizers. It deliberately reports measurements rather than asserting that
 * one algorithm or finalizer must win on every synthetic distribution.
 *
 * <p>Run with
 * {@code ./gradlew :engine:clustering:aljabr-clustering-emeans:test
 * --tests '*ClusteringQualityEvaluationTest'}.</p>
 */
class ClusteringQualityEvaluationTest {
    private static final long SEED = 0xC1A57E2L;
    private static final double[] SLOW_MULTIPLIERS = { 1.0 / 16.0, 0.25, 1.0, 4.0 };

    @Test
    void reportsHeldOutQualityAcrossSampleToClusterRatios() {
        System.out.println("scenario,algorithm,finalizer,train_sse,heldout_sse,heldout_mse");

        evaluateScenario("wide-ratio", 2048, 4, 16, 8);
        evaluateScenario("higher-k", 4096, 16, 16, 8);
    }

    private void evaluateScenario(
            String scenario, int sampleCount, int clusterCount, int dimensions, int epochs) {
        CpuBackend backend = new CpuBackend();
        Tensor training = null;
        Tensor heldOut = null;
        try {
            training = dataset(backend, sampleCount, clusterCount, dimensions, SEED);
            heldOut = dataset(backend, Math.max(1024, sampleCount / 2),
                    clusterCount, dimensions, SEED + 1);
            ClusteringDataset trainingSet = new ClusteringDataset(training);

            KMeansConfig lloydConfig = new KMeansConfig(
                    clusterCount, SEED,
                    tech.kayys.aljabr.clustering.api.DistanceMetric.SQUARED_EUCLIDEAN,
                    100, 1.0e-5, KMeansInitialization.RANDOM);
            try (ClusteringResult result = new KMeansAlgorithm().fit(trainingSet, lloydConfig)) {
                report(scenario, "lloyd", "batch-means", result, heldOut, dimensions);
            }

            int batchSize = Math.max(32, Math.min(256, sampleCount / 16));
            EMeansConfig base = EMeansConfig.defaults(clusterCount, SEED, epochs, batchSize);
            try (EMeansResult result = new EMeansAlgorithm().fit(trainingSet, base)) {
                report(scenario, "emeans", "last-epoch-means", result, heldOut, dimensions);
            }

            for (double multiplier : SLOW_MULTIPLIERS) {
                EMeansConfig config = base.withSlowEmaStepMultiplier(multiplier);
                try (EMeansResult result = new EMeansAlgorithm().fit(trainingSet, config)) {
                    report(scenario, "emeans", "slow-ema-x" + multiplier,
                            result, heldOut, dimensions, result.slowEmaCentroids().orElseThrow());
                }
            }
        } finally {
            if (heldOut != null) heldOut.close();
            if (training != null) training.close();
            backend.close();
        }
    }

    private static Tensor dataset(
            CpuBackend backend, int samples, int clusters, int dimensions, long seed) {
        float[] values = new float[Math.multiplyExact(samples, dimensions)];
        SplittableRandom random = new SplittableRandom(seed);
        for (int sample = 0; sample < samples; sample++) {
            int cluster = sample % clusters;
            int offset = sample * dimensions;
            for (int dimension = 0; dimension < dimensions; dimension++) {
                double center = 5.0 * Math.sin((cluster + 1.0) * (dimension + 1.0) * 0.37);
                values[offset + dimension] = (float) (center + 0.65 * random.nextGaussian());
            }
        }
        return TensorFactory.of(values, backend, samples, dimensions);
    }

    private static void report(
            String scenario, String algorithm, String finalizer,
            ClusteringResult result, Tensor heldOut, int dimensions) {
        report(scenario, algorithm, finalizer, result, heldOut, dimensions, result.centroids());
    }

    private static void report(
            String scenario, String algorithm, String finalizer,
            ClusteringResult result, Tensor heldOut, int dimensions, Tensor centroids) {
        double heldOutSse = heldOutSse(heldOut.toFloatArray(), heldOut.shape().dim(0),
                centroids.toFloatArray(), dimensions);
        double heldOutMse = heldOutSse / (heldOut.shape().dim(0) * (double) dimensions);
        double trainSse = result.metrics().objectiveValue();

        assertThat(Double.isFinite(trainSse)).isTrue();
        assertThat(Double.isFinite(heldOutSse)).isTrue();
        assertThat(heldOutSse).isGreaterThanOrEqualTo(0.0);
        System.out.printf("%s,%s,%s,%.6f,%.6f,%.8f%n",
                scenario, algorithm, finalizer, trainSse, heldOutSse, heldOutMse);
    }

    private static double heldOutSse(
            float[] samples, long sampleCount, float[] centroids, int dimensions) {
        int clusters = centroids.length / dimensions;
        double total = 0.0;
        for (int sample = 0; sample < sampleCount; sample++) {
            int sampleOffset = sample * dimensions;
            double nearest = Double.POSITIVE_INFINITY;
            for (int cluster = 0; cluster < clusters; cluster++) {
                int centroidOffset = cluster * dimensions;
                double distance = 0.0;
                for (int d = 0; d < dimensions; d++) {
                    double delta = samples[sampleOffset + d] - centroids[centroidOffset + d];
                    distance += delta * delta;
                }
                nearest = Math.min(nearest, distance);
            }
            total += nearest;
        }
        return total;
    }
}
