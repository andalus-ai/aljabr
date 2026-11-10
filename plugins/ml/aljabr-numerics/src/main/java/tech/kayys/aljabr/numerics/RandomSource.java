package tech.kayys.aljabr.numerics;

/**
 * Small deterministic random-source contract shared by numerical algorithms.
 *
 * <p>Implementations must document their generator and guarantee that equal
 * seeds produce equal sequences. This interface is not intended for
 * cryptographic use.</p>
 */
public interface RandomSource {
    long nextLong();

    /** Returns a uniformly distributed value in [0, 1). */
    default double nextDouble() {
        return (nextLong() >>> 11) * 0x1.0p-53;
    }

    /** Returns a uniformly distributed integer in [0, bound). */
    default int nextInt(int bound) {
        if (bound <= 0) {
            throw new IllegalArgumentException("bound must be positive");
        }
        long mask = bound - 1L;
        long bits = nextLong() >>> 1;
        if ((bound & mask) == 0L) {
            return (int) (bits & mask);
        }
        long value = bits % bound;
        while (bits + mask - value < 0L) {
            bits = nextLong() >>> 1;
            value = bits % bound;
        }
        return (int) value;
    }

    /** Creates an independent deterministic child stream. */
    RandomSource split();
}
