package tech.kayys.aljabr.clustering.emeans;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicBoolean;

import tech.kayys.aljabr.clustering.api.ClusteringAlgorithm;
import tech.kayys.aljabr.clustering.api.ClusteringDataset;
import tech.kayys.aljabr.clustering.api.ClusteringMetadata;
import tech.kayys.aljabr.clustering.api.ClusteringMetrics;
import tech.kayys.aljabr.clustering.api.ClusteringResult;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

/**
 * CPU reference implementation of mini-batch e-means for squared Euclidean
 * clustering. It follows the paper's EMA count/sum update, cosine step
 * schedule, periodic donor-split revival, and last-epoch-mean finalization.
 *
 * <p>This is a correctness-oriented reference, not the large-scale
 * vectorized/GPU implementation used for the paper's performance results.</p>
 */
public final class EMeansAlgorithm implements ClusteringAlgorithm<EMeansConfig> {
    public static final String ID = "emeans";
    private static final int REVIVAL_INTERVAL_BATCHES = 5;
    private static final double DEAD_CLUSTER_FRACTION = 0.1;
    private static final double SPLIT_EPSILON = 1.0e-4;
    private static final double INITIAL_MASS = 1.0e-3;

    @Override
    public String id() {
        return ID;
    }

    @Override
    public ClusteringResult fit(ClusteringDataset dataset, EMeansConfig config) {
        Objects.requireNonNull(dataset, "dataset must not be null");
        Objects.requireNonNull(config, "config must not be null");
        Tensor vectors = dataset.vectors();
        if (vectors.dtype() != DType.F32 || vectors.device() != DeviceType.CPU) {
            throw new UnsupportedOperationException("CPU e-means currently requires an F32 CPU tensor");
        }

        int samples = checkedLength(dataset.sampleCount(), "sample count");
        int dimensions = checkedLength(dataset.dimensions(), "dimensions");
        if (config.clusterCount() > samples) {
            throw new IllegalArgumentException(
                    "clusterCount must not exceed sample count: "
                            + config.clusterCount() + " > " + samples);
        }
        int elements = Math.toIntExact(Math.multiplyExact((long) samples, dimensions));
        float[] data = vectors.toFloatArray();
        if (data.length != elements) {
            throw new IllegalArgumentException("tensor data length does not match its declared shape");
        }
        for (int i = 0; i < data.length; i++) {
            if (!Float.isFinite(data[i])) {
                throw new IllegalArgumentException("input contains a non-finite value at flat index " + i);
            }
        }

        int clusters = config.clusterCount();
        SplittableRandom random = new SplittableRandom(config.seed());
        float[] centers = initialize(data, samples, dimensions, clusters, random);
        double[] masses = new double[clusters];
        double[] sums = new double[centers.length];
        for (int cluster = 0; cluster < clusters; cluster++) {
            masses[cluster] = INITIAL_MASS;
            for (int d = 0; d < dimensions; d++) {
                sums[cluster * dimensions + d] = centers[cluster * dimensions + d] * INITIAL_MASS;
            }
        }

        int batchesPerEpoch = (samples + config.batchSize() - 1) / config.batchSize();
        int totalBatches = Math.multiplyExact(config.epochs(), batchesPerEpoch);
        int[] order = new int[samples];
        int[] finalEpochAssignments = new int[samples];
        double[] finalEpochSums = new double[centers.length];
        int[] finalEpochCounts = new int[clusters];
        int batchNumber = 0;

        for (int epoch = 0; epoch < config.epochs(); epoch++) {
            for (int i = 0; i < samples; i++) {
                order[i] = i;
            }
            shuffle(order, random);
            if (epoch == config.epochs() - 1) {
                Arrays.fill(finalEpochAssignments, -1);
                Arrays.fill(finalEpochSums, 0.0);
                Arrays.fill(finalEpochCounts, 0);
            }

            for (int start = 0; start < samples; start += config.batchSize()) {
                int end = Math.min(samples, start + config.batchSize());
                int batchLength = end - start;
                int[] batchCounts = new int[clusters];
                double[] batchSums = new double[centers.length];
                int[] batchAssignments = new int[batchLength];

                for (int position = start; position < end; position++) {
                    int sample = order[position];
                    int cluster = nearest(data, sample * dimensions, centers, dimensions);
                    batchAssignments[position - start] = cluster;
                    batchCounts[cluster]++;
                    int source = sample * dimensions;
                    int target = cluster * dimensions;
                    for (int d = 0; d < dimensions; d++) {
                        batchSums[target + d] += data[source + d];
                    }
                    if (epoch == config.epochs() - 1) {
                        finalEpochAssignments[sample] = cluster;
                        finalEpochCounts[cluster]++;
                        for (int d = 0; d < dimensions; d++) {
                            finalEpochSums[target + d] += data[source + d];
                        }
                    }
                }

                double progress = (double) batchNumber / totalBatches;
                double alpha = config.initialStepSize()
                        * (0.1 + 0.45 * (1.0 + Math.cos(Math.PI * progress)));
                double decay = 1.0 - alpha;
                for (int cluster = 0; cluster < clusters; cluster++) {
                    masses[cluster] = decay * masses[cluster] + alpha * batchCounts[cluster];
                    int offset = cluster * dimensions;
                    for (int d = 0; d < dimensions; d++) {
                        sums[offset + d] = decay * sums[offset + d]
                                + alpha * batchSums[offset + d];
                    }
                    if (masses[cluster] > 0.0) {
                        for (int d = 0; d < dimensions; d++) {
                            centers[offset + d] = (float) (sums[offset + d] / masses[cluster]);
                        }
                    }
                }

                batchNumber++;
                if (config.reviveDeadClusters()
                        && batchNumber % REVIVAL_INTERVAL_BATCHES == 0) {
                    reviveDeadClusters(centers, masses, sums, clusters, dimensions, random);
                }
            }
        }

        // The paper's last-epoch mean finalizer minimizes the objective for
        // the assignments observed in the final epoch.
        for (int cluster = 0; cluster < clusters; cluster++) {
            if (finalEpochCounts[cluster] == 0) {
                continue;
            }
            int offset = cluster * dimensions;
            for (int d = 0; d < dimensions; d++) {
                centers[offset + d] = (float) (
                        finalEpochSums[offset + d] / finalEpochCounts[cluster]);
            }
        }

        int[] assignments = new int[samples];
        double objective = 0.0;
        for (int sample = 0; sample < samples; sample++) {
            int cluster = nearest(data, sample * dimensions, centers, dimensions);
            assignments[sample] = cluster;
            objective += squaredDistance(data, sample * dimensions, centers, cluster * dimensions, dimensions);
        }

        Tensor centroidTensor = TensorFactory.of(centers, vectors.backend(), clusters, dimensions);
        return new EMeansResult(
                centroidTensor,
                assignments,
                new ClusteringMetrics("sum_squared_euclidean", objective),
                new ClusteringMetadata(ID, config.epochs(), config.seed(), true));
    }

    private static float[] initialize(
            float[] data, int samples, int dimensions, int clusters, SplittableRandom random) {
        int[] selected = new int[samples];
        for (int i = 0; i < samples; i++) {
            selected[i] = i;
        }
        float[] centers = new float[Math.multiplyExact(clusters, dimensions)];
        for (int cluster = 0; cluster < clusters; cluster++) {
            int pick = cluster + random.nextInt(samples - cluster);
            int temporary = selected[cluster];
            selected[cluster] = selected[pick];
            selected[pick] = temporary;
            System.arraycopy(data, selected[cluster] * dimensions,
                    centers, cluster * dimensions, dimensions);
        }
        return centers;
    }

    private static void shuffle(int[] values, SplittableRandom random) {
        for (int i = values.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temporary = values[i];
            values[i] = values[j];
            values[j] = temporary;
        }
    }

    private static int nearest(float[] data, int sampleOffset, float[] centers, int dimensions) {
        int clusters = centers.length / dimensions;
        int best = 0;
        double bestDistance = Double.POSITIVE_INFINITY;
        for (int cluster = 0; cluster < clusters; cluster++) {
            double candidate = squaredDistance(
                    data, sampleOffset, centers, cluster * dimensions, dimensions);
            if (candidate < bestDistance) {
                bestDistance = candidate;
                best = cluster;
            }
        }
        return best;
    }

    private static double squaredDistance(
            float[] left, int leftOffset, float[] right, int rightOffset, int dimensions) {
        double sum = 0.0;
        for (int d = 0; d < dimensions; d++) {
            double difference = (double) left[leftOffset + d] - right[rightOffset + d];
            sum += difference * difference;
        }
        return sum;
    }

    private static void reviveDeadClusters(
            float[] centers,
            double[] masses,
            double[] sums,
            int clusters,
            int dimensions,
            SplittableRandom random) {
        double meanMass = Arrays.stream(masses).sum() / clusters;
        boolean[] candidates = new boolean[clusters];
        int candidateCount = 0;
        for (int cluster = 0; cluster < clusters; cluster++) {
            if (masses[cluster] < DEAD_CLUSTER_FRACTION * meanMass) {
                candidates[cluster] = true;
                candidateCount++;
            }
        }
        if (candidateCount == 0 || candidateCount == clusters) {
            return;
        }

        boolean[] donorUsed = new boolean[clusters];
        for (int candidate = 0; candidate < clusters; candidate++) {
            if (!candidates[candidate]) {
                continue;
            }
            double donorMass = 0.0;
            for (int donor = 0; donor < clusters; donor++) {
                if (!candidates[donor] && !donorUsed[donor]) {
                    donorMass += masses[donor];
                }
            }
            if (donorMass <= 0.0) {
                break;
            }
            double target = random.nextDouble(donorMass);
            int donor = -1;
            for (int index = 0; index < clusters; index++) {
                if (!candidates[index] && !donorUsed[index]) {
                    target -= masses[index];
                    if (target <= 0.0) {
                        donor = index;
                        break;
                    }
                }
            }
            if (donor < 0) {
                continue;
            }
            donorUsed[donor] = true;
            double childMass = masses[donor] * 0.5;
            int donorOffset = donor * dimensions;
            int candidateOffset = candidate * dimensions;
            for (int d = 0; d < dimensions; d++) {
                double sign = (d & 1) == 0 ? 1.0 : -1.0;
                double original = centers[donorOffset + d];
                centers[donorOffset + d] = (float) (original * (1.0 - SPLIT_EPSILON * sign));
                centers[candidateOffset + d] = (float) (original * (1.0 + SPLIT_EPSILON * sign));
                sums[donorOffset + d] = centers[donorOffset + d] * childMass;
                sums[candidateOffset + d] = centers[candidateOffset + d] * childMass;
            }
            masses[donor] = childMass;
            masses[candidate] = childMass;
        }
    }

    private static int checkedLength(long value, String label) {
        if (value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    label + " exceeds the CPU reference implementation limit: " + value);
        }
        return Math.toIntExact(value);
    }

    private static final class EMeansResult implements ClusteringResult {
        private final Tensor centroids;
        private final int[] assignments;
        private final ClusteringMetrics metrics;
        private final ClusteringMetadata metadata;
        private final AtomicBoolean closed = new AtomicBoolean();

        private EMeansResult(
                Tensor centroids, int[] assignments,
                ClusteringMetrics metrics, ClusteringMetadata metadata) {
            this.centroids = Objects.requireNonNull(centroids);
            this.assignments = assignments;
            this.metrics = metrics;
            this.metadata = metadata;
        }

        @Override
        public Tensor centroids() {
            ensureOpen();
            return centroids;
        }

        @Override
        public long sampleCount() {
            return assignments.length;
        }

        @Override
        public int clusterCount() {
            ensureOpen();
            return Math.toIntExact(centroids.shape().dim(0));
        }

        @Override
        public int clusterOf(long sampleIndex) {
            ensureOpen();
            if (sampleIndex < 0 || sampleIndex >= assignments.length) {
                throw new IndexOutOfBoundsException("sample index out of range: " + sampleIndex);
            }
            return assignments[Math.toIntExact(sampleIndex)];
        }

        @Override
        public ClusteringMetrics metrics() {
            ensureOpen();
            return metrics;
        }

        @Override
        public ClusteringMetadata metadata() {
            ensureOpen();
            return metadata;
        }

        @Override
        public void close() {
            if (closed.compareAndSet(false, true)) {
                centroids.close();
            }
        }

        private void ensureOpen() {
            if (closed.get()) {
                throw new IllegalStateException("clustering result is closed");
            }
        }
    }
}
