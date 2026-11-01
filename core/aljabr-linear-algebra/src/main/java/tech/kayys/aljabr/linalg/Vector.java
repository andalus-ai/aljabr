package tech.kayys.aljabr.linalg;

import java.util.Arrays;
import java.util.Objects;

/** Immutable dense vector backed by a defensive copy of double-precision values. */
public final class Vector {
    private final double[] values;

    private Vector(double[] values) {
        this.values = values;
    }

    public static Vector of(double... values) {
        Objects.requireNonNull(values, "values");
        return new Vector(values.clone());
    }

    public int size() { return values.length; }

    public double get(int index) { return values[index]; }

    public double[] toArray() { return values.clone(); }

    public Vector add(Vector other) {
        requireSameSize(other);
        double[] result = new double[size()];
        for (int i = 0; i < size(); i++) result[i] = values[i] + other.values[i];
        return new Vector(result);
    }

    public Vector subtract(Vector other) {
        requireSameSize(other);
        double[] result = new double[size()];
        for (int i = 0; i < size(); i++) result[i] = values[i] - other.values[i];
        return new Vector(result);
    }

    public Vector scale(double scalar) {
        double[] result = new double[size()];
        for (int i = 0; i < size(); i++) result[i] = values[i] * scalar;
        return new Vector(result);
    }

    public double dot(Vector other) {
        requireSameSize(other);
        double sum = 0.0;
        for (int i = 0; i < size(); i++) sum = Math.fma(values[i], other.values[i], sum);
        return sum;
    }

    public double norm2() {
        double norm = 0.0;
        for (double value : values) norm = Math.hypot(norm, value);
        return norm;
    }

    private void requireSameSize(Vector other) {
        Objects.requireNonNull(other, "other");
        if (size() != other.size()) throw new IllegalArgumentException("vector sizes differ: " + size() + " and " + other.size());
    }

    @Override public String toString() { return Arrays.toString(values); }
}
