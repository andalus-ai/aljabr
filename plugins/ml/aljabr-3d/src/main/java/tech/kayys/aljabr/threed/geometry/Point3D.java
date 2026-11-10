package tech.kayys.aljabr.threed.geometry;

import java.util.Objects;

/**
 * Immutable 3D Cartesian vector / point (x, y, z).
 * @author bhangun
 */
public record Point3D(double x, double y, double z) {

    public static final Point3D ZERO = new Point3D(0, 0, 0);
    public static final Point3D UNIT_X = new Point3D(1, 0, 0);
    public static final Point3D UNIT_Y = new Point3D(0, 1, 0);
    public static final Point3D UNIT_Z = new Point3D(0, 0, 1);

    public static Point3D of(double x, double y, double z) {
        return new Point3D(x, y, z);
    }

    public Point3D add(Point3D other) {
        return new Point3D(this.x + other.x, this.y + other.y, this.z + other.z);
    }

    public Point3D subtract(Point3D other) {
        return new Point3D(this.x - other.x, this.y - other.y, this.z - other.z);
    }

    public Point3D multiply(double scalar) {
        return new Point3D(this.x * scalar, this.y * scalar, this.z * scalar);
    }

    public Point3D divide(double scalar) {
        if (Math.abs(scalar) < 1e-12) return ZERO;
        return new Point3D(this.x / scalar, this.y / scalar, this.z / scalar);
    }

    public double dot(Point3D other) {
        return this.x * other.x + this.y * other.y + this.z * other.z;
    }

    public Point3D cross(Point3D other) {
        return new Point3D(
                this.y * other.z - this.z * other.y,
                this.z * other.x - this.x * other.z,
                this.x * other.y - this.y * other.x
        );
    }

    public double normSquared() {
        return x * x + y * y + z * z;
    }

    public double norm() {
        return Math.sqrt(normSquared());
    }

    public double distanceTo(Point3D other) {
        return subtract(other).norm();
    }

    public Point3D normalize() {
        double n = norm();
        return n > 1e-12 ? divide(n) : ZERO;
    }

    public Point3D lerp(Point3D target, double t) {
        return new Point3D(
                this.x + (target.x - this.x) * t,
                this.y + (target.y - this.y) * t,
                this.z + (target.z - this.z) * t
        );
    }

    public double[] toArray() {
        return new double[]{x, y, z};
    }

    public float[] toFloatArray() {
        return new float[]{(float) x, (float) y, (float) z};
    }
}
