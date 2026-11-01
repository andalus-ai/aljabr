package tech.kayys.aljabr.linalg;

import java.util.Objects;

/** Immutable dense row-major matrix with double-precision elements. */
public final class Matrix {
    private final int rows;
    private final int columns;
    private final double[][] values;

    private Matrix(double[][] values, int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        this.values = values;
    }

    public static Matrix of(double[][] values) {
        Objects.requireNonNull(values, "values");
        if (values.length == 0) throw new IllegalArgumentException("matrix must have at least one row");
        Objects.requireNonNull(values[0], "row 0");
        int columns = values[0].length;
        if (columns == 0) throw new IllegalArgumentException("matrix must have at least one column");
        double[][] copy = new double[values.length][columns];
        for (int row = 0; row < values.length; row++) {
            Objects.requireNonNull(values[row], "row " + row);
            if (values[row].length != columns) throw new IllegalArgumentException("matrix rows must have equal length");
            copy[row] = values[row].clone();
        }
        return new Matrix(copy, values.length, columns);
    }

    public static Matrix identity(int size) {
        if (size <= 0) throw new IllegalArgumentException("size must be positive");
        double[][] values = new double[size][size];
        for (int i = 0; i < size; i++) values[i][i] = 1.0;
        return new Matrix(values, size, size);
    }

    public int rows() { return rows; }
    public int columns() { return columns; }
    public double get(int row, int column) { return values[row][column]; }

    public double[][] toArray() {
        double[][] copy = new double[rows][];
        for (int row = 0; row < rows; row++) copy[row] = values[row].clone();
        return copy;
    }

    public Matrix transpose() {
        double[][] result = new double[columns][rows];
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++) result[column][row] = values[row][column];
        return new Matrix(result, columns, rows);
    }

    public Matrix add(Matrix other) { return elementwise(other, false); }
    public Matrix subtract(Matrix other) { return elementwise(other, true); }

    private Matrix elementwise(Matrix other, boolean subtract) {
        requireSameShape(other);
        double[][] result = new double[rows][columns];
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                result[row][column] = subtract ? values[row][column] - other.values[row][column]
                        : values[row][column] + other.values[row][column];
        return new Matrix(result, rows, columns);
    }

    public Matrix scale(double scalar) {
        double[][] result = new double[rows][columns];
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++) result[row][column] = values[row][column] * scalar;
        return new Matrix(result, rows, columns);
    }

    public Matrix multiply(Matrix other) {
        Objects.requireNonNull(other, "other");
        if (columns != other.rows) throw new IllegalArgumentException("matrix dimensions do not conform for multiplication");
        double[][] result = new double[rows][other.columns];
        for (int row = 0; row < rows; row++) {
            for (int k = 0; k < columns; k++) {
                double left = values[row][k];
                for (int column = 0; column < other.columns; column++)
                    result[row][column] = Math.fma(left, other.values[k][column], result[row][column]);
            }
        }
        return new Matrix(result, rows, other.columns);
    }

    public Vector multiply(Vector vector) {
        Objects.requireNonNull(vector, "vector");
        if (columns != vector.size()) throw new IllegalArgumentException("matrix columns must equal vector size");
        double[] result = new double[rows];
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
                result[row] = Math.fma(values[row][column], vector.get(column), result[row]);
        return Vector.of(result);
    }

    /** Maximum absolute row sum, the induced infinity norm. */
    public double normInfinity() {
        double maximum = 0.0;
        for (int row = 0; row < rows; row++) {
            double sum = 0.0;
            for (int column = 0; column < columns; column++) sum += Math.abs(values[row][column]);
            maximum = Math.max(maximum, sum);
        }
        return maximum;
    }

    private void requireSameShape(Matrix other) {
        Objects.requireNonNull(other, "other");
        if (rows != other.rows || columns != other.columns)
            throw new IllegalArgumentException("matrix shapes differ");
    }
}
