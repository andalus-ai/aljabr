package tech.kayys.aljabr.cache.quantization;

/**
 * Reference Lloyd-Max scalar codebook for coordinates of a uniformly rotated
 * unit vector.
 *
 * <p>For a vector of dimension {@code d}, the scaled coordinate
 * {@code z = sqrt(d) * x[i]} has density proportional to
 * {@code (1 - z*z/d)^((d-3)/2)} on {@code [-sqrt(d), sqrt(d)]}. This class
 * numerically solves the one-dimensional Lloyd-Max centroid/boundary
 * equations for that spherical-coordinate distribution.</p>
 *
 * <p>The implementation uses deterministic midpoint quadrature and is
 * intended as a correctness/reference implementation. It is not a trained
 * codebook, does not assume a dataset, and is not yet a packed bitstream or
 * complete TurboQuant KV-cache codec. Callers quantize normalized, rotated
 * coordinates scaled by {@code sqrt(d)} and must store the original vector
 * norm separately.</p>
 *
 * <p>Dimensions below three are rejected because the continuous density
 * representation is singular or degenerate there. Production transformer
 * head dimensions are expected to be substantially larger.</p>
 */
public final class TurboQuantScalarCodebook {

    private static final int INTEGRATION_BINS = 32_768;
    private static final int MAX_ITERATIONS = 200;
    private static final double CONVERGENCE_TOLERANCE = 1.0e-10;

    private final int dimension;
    private final int bits;
    private final double[] centroids;
    private final double[] boundaries;

    private TurboQuantScalarCodebook(int dimension, int bits, double[] centroids, double[] boundaries) {
        this.dimension = dimension;
        this.bits = bits;
        this.centroids = centroids;
        this.boundaries = boundaries;
    }

    /**
     * Builds a deterministic codebook using the spherical-coordinate density.
     *
     * @param dimension original vector dimension
     * @param bits number of bits per scalar coordinate, from 1 through 8
     */
    public static TurboQuantScalarCodebook forDimension(int dimension, int bits) {
        if (dimension < 3) {
            throw new IllegalArgumentException("dimension must be at least 3");
        }
        if (bits < 1 || bits > 8) {
            throw new IllegalArgumentException("bits must be between 1 and 8");
        }

        Distribution distribution = new Distribution(dimension, INTEGRATION_BINS);
        int levels = 1 << bits;
        double[] boundaries = new double[levels + 1];
        boundaries[0] = -distribution.limit;
        boundaries[levels] = distribution.limit;

        // Equal-probability initialization is substantially more stable than
        // equally spaced thresholds for the concentrated spherical density.
        for (int i = 1; i < levels; i++) {
            boundaries[i] = distribution.quantile((double) i / levels);
        }

        double[] centroids = new double[levels];
        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            for (int i = 0; i < levels; i++) {
                centroids[i] = distribution.conditionalMean(boundaries[i], boundaries[i + 1]);
            }

            double maxMovement = 0;
            double[] nextBoundaries = boundaries.clone();
            for (int i = 1; i < levels; i++) {
                nextBoundaries[i] = (centroids[i - 1] + centroids[i]) * 0.5;
                maxMovement = Math.max(maxMovement, Math.abs(nextBoundaries[i] - boundaries[i]));
            }
            boundaries = nextBoundaries;
            if (maxMovement < CONVERGENCE_TOLERANCE) {
                break;
            }
        }

        for (int i = 0; i < levels; i++) {
            centroids[i] = distribution.conditionalMean(boundaries[i], boundaries[i + 1]);
        }
        return new TurboQuantScalarCodebook(dimension, bits, centroids, boundaries);
    }

    public int dimension() {
        return dimension;
    }

    public int bits() {
        return bits;
    }

    public int levels() {
        return centroids.length;
    }

    /** Returns a defensive copy of reconstruction values in scaled coordinates. */
    public double[] centroids() {
        return centroids.clone();
    }

    /** Returns a defensive copy of decision boundaries in scaled coordinates. */
    public double[] boundaries() {
        return boundaries.clone();
    }

    /**
     * Maps one scaled coordinate to its nearest-centroid index.
     * Values outside the theoretical support are saturated to the edge levels.
     */
    public int quantize(double scaledCoordinate) {
        if (!Double.isFinite(scaledCoordinate)) {
            throw new IllegalArgumentException("scaledCoordinate must be finite");
        }
        int low = 1;
        int high = boundaries.length - 1;
        while (low < high) {
            int middle = (low + high) >>> 1;
            if (scaledCoordinate < boundaries[middle]) {
                high = middle;
            } else {
                low = middle + 1;
            }
        }
        return low - 1;
    }

    public double reconstruct(int index) {
        if (index < 0 || index >= centroids.length) {
            throw new IndexOutOfBoundsException("codebook index out of range: " + index);
        }
        return centroids[index];
    }

    private static final class Distribution {
        private final double limit;
        private final double step;
        private final double[] cumulativeMass;
        private final double[] cumulativeMoment;

        private Distribution(int dimension, int bins) {
            this.limit = Math.sqrt(dimension);
            this.step = (2.0 * limit) / bins;
            this.cumulativeMass = new double[bins + 1];
            this.cumulativeMoment = new double[bins + 1];

            double exponent = (dimension - 3.0) * 0.5;
            double logNormalizer = logGamma(dimension * 0.5)
                    - 0.5 * Math.log(Math.PI)
                    - logGamma((dimension - 1.0) * 0.5)
                    - Math.log(limit);
            double normalizer = Math.exp(logNormalizer);

            for (int i = 0; i < bins; i++) {
                double z = -limit + (i + 0.5) * step;
                double base = Math.max(0.0, 1.0 - (z * z) / dimension);
                double density = normalizer * Math.pow(base, exponent);
                double mass = density * step;
                cumulativeMass[i + 1] = cumulativeMass[i] + mass;
                cumulativeMoment[i + 1] = cumulativeMoment[i] + z * mass;
            }

            double totalMass = cumulativeMass[bins];
            if (!(totalMass > 0.0) || !Double.isFinite(totalMass)) {
                throw new IllegalStateException("failed to construct spherical coordinate density");
            }
            for (int i = 1; i <= bins; i++) {
                cumulativeMass[i] /= totalMass;
                cumulativeMoment[i] /= totalMass;
            }
        }

        private double quantile(double probability) {
            int low = 0;
            int high = cumulativeMass.length - 1;
            while (low < high) {
                int middle = (low + high) >>> 1;
                if (cumulativeMass[middle] < probability) {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            int index = Math.max(1, low);
            double before = cumulativeMass[index - 1];
            double after = cumulativeMass[index];
            double fraction = after == before ? 0.5 : (probability - before) / (after - before);
            return -limit + ((index - 1) + fraction) * step;
        }

        private double conditionalMean(double lower, double upper) {
            double mass = cumulative(upper, cumulativeMass) - cumulative(lower, cumulativeMass);
            if (!(mass > 1.0e-15)) {
                return (lower + upper) * 0.5;
            }
            double moment = cumulative(upper, cumulativeMoment) - cumulative(lower, cumulativeMoment);
            return moment / mass;
        }

        private double cumulative(double value, double[] prefix) {
            if (value <= -limit) return 0.0;
            if (value >= limit) return prefix[prefix.length - 1];
            double position = (value + limit) / step;
            int index = Math.min(prefix.length - 2, (int) position);
            double fraction = position - index;
            return prefix[index] + fraction * (prefix[index + 1] - prefix[index]);
        }

        private static double logGamma(double value) {
            double[] coefficients = {
                676.5203681218851, -1259.1392167224028, 771.32342877765313,
                -176.61502916214059, 12.507343278686905, -0.13857109526572012,
                9.9843695780195716e-6, 1.5056327351493116e-7
            };
            if (value < 0.5) {
                return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value))
                        - logGamma(1.0 - value);
            }
            double shifted = value - 1.0;
            double sum = 0.99999999999980993;
            for (int i = 0; i < coefficients.length; i++) {
                sum += coefficients[i] / (shifted + i + 1.0);
            }
            double t = shifted + coefficients.length - 0.5;
            return 0.5 * Math.log(2.0 * Math.PI)
                    + (shifted + 0.5) * Math.log(t) - t + Math.log(sum);
        }
    }
}
