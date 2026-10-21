package tech.kayys.aljabr.core.random;

import java.util.Objects;
import java.util.Random;

/**
 * Shared Gaussian noise helpers for model initialization, diffusion latents,
 * and latent-variable training.
 * @author bhangun
 */
public final class GaussianNoise {
    private GaussianNoise() {
    }

    public static float[] floats(int size, long seed) {
        float[] values = new float[requireNonNegativeSize(size)];
        fill(values, seed);
        return values;
    }

    public static float[] floats(int size, Random random) {
        float[] values = new float[requireNonNegativeSize(size)];
        fill(values, random);
        return values;
    }

    public static void fill(float[] values, long seed) {
        fill(values, new Random(seed));
    }

    public static void fill(float[] values, Random random) {
        Objects.requireNonNull(values, "values must not be null");
        Objects.requireNonNull(random, "random must not be null");
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) random.nextGaussian();
        }
    }

    private static int requireNonNegativeSize(int size) {
        if (size < 0) {
            throw new IllegalArgumentException("size must be >= 0 but was " + size);
        }
        return size;
    }
}
