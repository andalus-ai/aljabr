package tech.kayys.aljabr.clustering.kmeans;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;

import tech.kayys.aljabr.clustering.api.ClusteringAlgorithm;
import tech.kayys.aljabr.clustering.api.ClusteringDataset;
import tech.kayys.aljabr.clustering.api.ClusteringMetadata;
import tech.kayys.aljabr.clustering.api.ClusteringMetrics;
import tech.kayys.aljabr.clustering.api.ClusteringResult;
import tech.kayys.aljabr.clustering.api.DistanceMetric;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * Deterministic CPU reference implementation of batch Lloyd k-means.
 *
 * <p>This implementation intentionally accepts only F32 CPU tensors. It is a
 * correctness baseline; accelerator execution and large-data streaming are
 * separate implementations behind the same clustering API.</p>
 */
public final class KMeansAlgorithm implements ClusteringAlgorithm<KMeansConfig> {
    public static final String ID = "kmeans";

    @Override
    public String id() {
        return ID;
    }

    @Override
    public ClusteringResult fit(ClusteringDataset dataset, KMeansConfig config) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(config, "config must not be null");

        Tensor vectors = dataset.vectors();
        if (vectors.dtype() != DType.F32 || vectors.device() != DeviceType.CPU) {
            throw new UnsupportedOperationException("CPU k-means currently requires an F32 CPU tensor");
        }

        int sampleCount = checkedArrayLength(dataset.sampleCount(), "sample count");
        int dimensions = checkedArrayLength(dataset.dimensions(), "dimensions");
        if (config.clusterCount() > sampleCount) {
            throw new IllegalArgumentException(
                    "clusterCount must not exceed sample count: " + config.clusterCount() + " > " + sampleCount);
        }
        int elementCount = Math.toIntExact(Math.multiplyExact((long) sampleCount, dimensions));
        float[] data = vectors.toFloatArray();
        if (data.length != elementCount) {
            throw new IllegalArgumentException("tensor data length does not match its declared shape");
        }
        for (int i = 0; i < data.length; i++) {
            if (!Float.isFinite(data[i])) {
                throw new IllegalArgumentException("input contains a non-finite value at flat index " + i);
            }
        }

        if (config.distanceMetric() == DistanceMetric.COSINE) {
            normalizeRows(data, sampleCount, dimensions);
        }

        SplittableRandom random = new SplittableRandom(config.seed());
        float[] centroids = initialize(data, sampleCount, dimensions, config, random);
        int[] assignments = new int[sampleCount];
        Arrays.fill(assignments, -1);

        int iterations = 0;
        boolean converged = false;
        for (; iterations < config.maxIterations(); iterations++) {
            int[] nextAssignments = assign(data, centroids, sampleCount, dimensions, config.distanceMetric());
            boolean assignmentsStable = Arrays.equals(assignments, nextAssignments);
            float[] nextCentroids = updateCentroids(
                    data, nextAssignments, centroids, sampleCount, dimensions, config.distanceMetric());
            double shift = maximumCentroidShift(centroids, nextCentroids, config.distanceMetric());
            assignments = nextAssignments;
            centroids = nextCentroids;
            if (assignmentsStable || shift <= config.tolerance()) {
                converged = true;
                iterations++;
                break;
            }
        }

        // Reassign against the final centroid values so labels and objective agree.
        assignments = assign(data, centroids, sampleCount, dimensions, config.distanceMetric());
        double objective = objective(data, centroids, assignments, sampleCount, dimensions, config.distanceMetric());
        Tensor centroidTensor = TensorFactory.of(
                centroids, vectors.backend(), config.clusterCount(), dimensions);
        String objectiveName = config.distanceMetric() == DistanceMetric.COSINE
                ? "sum_cosine_distance" : "sum_squared_euclidean";
        return new KMeansResult(
                centroidTensor,
                assignments,
                new ClusteringMetrics(objectiveName, objective),
                new ClusteringMetadata(ID, iterations, config.seed(), converged));
    }

    private static float[] initialize(
            float[] data,
            int samples,
            int dimensions,
            KMeansConfig config,
            SplittableRandom random) {
        int clusters = config.clusterCount();
        float[] centroids = new float[Math.multiplyExact(clusters, dimensions)];
        if (config.initialization() == KMeansInitialization.RANDOM) {
            int[] selected = new int[samples];
            for (int i = 0; i < samples; i++) {
                selected[i] = i;
            }
            for (int i = 0; i < clusters; i++) {
                int selectedIndex = i + random.nextInt(samples - i);
                int temporary = selected[i];
                selected[i] = selected[selectedIndex];
                selected[selectedIndex] = temporary;
                copyRow(data, selected[i], centroids, i, dimensions);
            }
            return centroids;
        }

        int first = random.nextInt(samples);
        copyRow(data, first, centroids, 0, dimensions);
        double[] nearestDistances = new double[samples];
        for (int sample = 0; sample < samples; sample++) {
            nearestDistances[sample] = distance(
                    data, sample * dimensions, centroids, 0, dimensions, config.distanceMetric());
        }

        for (int cluster = 1; cluster < clusters; cluster++) {
            double total = 0.0;
            for (double value : nearestDistances) {
                total += value;
            }
            int selected;
            if (total <= 0.0 || !Double.isFinite(total)) {
                selected = random.nextInt(samples);
            } else {
                double target = random.nextDouble(total);
                selected = samples - 1;
                for (int sample = 0; sample < samples; sample++) {
                    target -= nearestDistances[sample];
                    if (target <= 0.0) {
                        selected = sample;
                        break;
                    }
                }
            }
            copyRow(data, selected, centroids, cluster, dimensions);
            for (int sample = 0; sample < samples; sample++) {
                double candidate = distance(
                        data, sample * dimensions, centroids, cluster * dimensions,
                        dimensions, config.distanceMetric());
                nearestDistances[sample] = Math.min(nearestDistances[sample], candidate);
            }
        }
        return centroids;
    }

    private static int[] assign(
            float[] data, float[] centroids, int samples, int dimensions, DistanceMetric metric) {
        int clusters = centroids.length / dimensions;
        int[] assignments = new int[samples];
        for (int sample = 0; sample < samples; sample++) {
            int sampleOffset = sample * dimensions;
            int bestCluster = 0;
            double bestDistance = Double.POSITIVE_INFINITY;
            for (int cluster = 0; cluster < clusters; cluster++) {
                double candidate = distance(
                        data, sampleOffset, centroids, cluster * dimensions, dimensions, metric);
                if (candidate < bestDistance) {
                    bestDistance = candidate;
                    bestCluster = cluster;
                }
            }
            assignments[sample] = bestCluster;
        }
        return assignments;
    }

    private static float[] updateCentroids(
            float[] data,
            int[] assignments,
            float[] previous,
            int samples,
            int dimensions,
            DistanceMetric metric) {
        int clusters = previous.length / dimensions;
        float[] updated = new float[previous.length];
        int[] counts = new int[clusters];
        for (int sample = 0; sample < samples; sample++) {
            int cluster = assignments[sample];
            counts[cluster]++;
            int source = sample * dimensions;
            int target = cluster * dimensions;
            for (int dimension = 0; dimension < dimensions; dimension++) {
                updated[target + dimension] += data[source + dimension];
            }
        }
        for (int cluster = 0; cluster < clusters; cluster++) {
            int offset = cluster * dimensions;
            if (counts[cluster] == 0) {
                // Retain the previous center; empty-cluster recovery is a separate policy.
                System.arraycopy(previous, offset, updated, offset, dimensions);
                continue;
            }
            float inverseCount = 1.0f / counts[cluster];
            for (int dimension = 0; dimension < dimensions; dimension++) {
                updated[offset + dimension] *= inverseCount;
            }
            if (metric == DistanceMetric.COSINE) {
                normalizeRow(updated, offset, dimensions);
            }
        }
        return updated;
    }

    private static double maximumCentroidShift(
            float[] previous, float[] updated, DistanceMetric metric) {
        int dimensions = previous.length;
        double maximum = 0.0;
        for (int i = 0; i < dimensions; i++) {
            double difference = (double) previous[i] - updated[i];
            maximum = Math.max(maximum, difference * difference);
        }
        return metric == DistanceMetric.COSINE ? Math.sqrt(maximum) : Math.sqrt(maximum);
    }

    private static double objective(
            float[] data,
            float[] centroids,
            int[] assignments,
            int samples,
            int dimensions,
            DistanceMetric metric) {
        double total = 0.0;
        for (int sample = 0; sample < samples; sample++) {
            total += distance(
                    data, sample * dimensions,
                    centroids, assignments[sample] * dimensions, dimensions, metric);
        }
        return total;
    }

    private static double distance(
            float[] left,
            int leftOffset,
            float[] right,
            int rightOffset,
            int dimensions,
            DistanceMetric metric) {
        if (metric == DistanceMetric.SQUARED_EUCLIDEAN) {
            double sum = 0.0;
            for (int i = 0; i < dimensions; i++) {
                double difference = (double) left[leftOffset + i] - right[rightOffset + i];
                sum += difference * difference;
            }
            return sum;
        }
        double dot = 0.0;
        double leftNorm = 0.0;
        double rightNorm = 0.0;
        for (int i = 0; i < dimensions; i++) {
            double a = left[leftOffset + i];
            double b = right[rightOffset + i];
            dot += a * b;
            leftNorm += a * a;
            rightNorm += b * b;
        }
        if (leftNorm == 0.0 || rightNorm == 0.0) {
            return leftNorm == 0.0 && rightNorm == 0.0 ? 0.0 : 1.0;
        }
        return 1.0 - Math.max(-1.0, Math.min(1.0, dot / Math.sqrt(leftNorm * rightNorm)));
    }

    private static void normalizeRows(float[] values, int rows, int dimensions) {
        for (int row = 0; row < rows; row++) {
            normalizeRow(values, row * dimensions, dimensions);
        }
    }

    private static void normalizeRow(float[] values, int offset, int dimensions) {
        double normSquared = 0.0;
        for (int i = 0; i < dimensions; i++) {
            double value = values[offset + i];
            normSquared += value * value;
        }
        if (normSquared == 0.0) {
            return;
        }
        float inverseNorm = (float) (1.0 / Math.sqrt(normSquared));
        for (int i = 0; i < dimensions; i++) {
            values[offset + i] *= inverseNorm;
        }
    }

    private static void copyRow(
            float[] source, int sourceRow, float[] target, int targetRow, int dimensions) {
        System.arraycopy(source, sourceRow * dimensions, target, targetRow * dimensions, dimensions);
    }

    private static int checkedArrayLength(long value, String label) {
        if (value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(label + " exceeds the CPU reference implementation limit: " + value);
        }
        return Math.toIntExact(value);
    }
}
