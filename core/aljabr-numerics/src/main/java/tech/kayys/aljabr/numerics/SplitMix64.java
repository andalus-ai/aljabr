package tech.kayys.aljabr.numerics;

/**
 * Small, fast, reproducible SplitMix64 pseudo-random generator.
 *
 * <p>Suitable for simulation, initialization, and randomized numerical
 * algorithms. It is not a cryptographically secure random generator.</p>
 */
public final class SplitMix64 implements RandomSource {
    private long state;

    public SplitMix64(long seed) {
        this.state = seed;
    }

    @Override
    public long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    @Override
    public RandomSource split() {
        return new SplitMix64(nextLong());
    }
}
