package tech.kayys.aljabr.numerics;

import java.util.Objects;

/** Shared validation helpers for numerical APIs. */
public final class NumericalChecks {
    private NumericalChecks() {
    }

    public static double requireFinite(double value, String name) {
        return requireFinite(value, name, NonFinitePolicy.REJECT);
    }

    public static double requireFinite(double value, String name, NonFinitePolicy policy) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(policy, "policy");
        if (policy == NonFinitePolicy.REJECT && !Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite, but was " + value);
        }
        return value;
    }

    public static double requirePositiveFinite(double value, String name) {
        requireFinite(value, name);
        if (value <= 0.0) {
            throw new IllegalArgumentException(name + " must be greater than zero");
        }
        return value;
    }
}
