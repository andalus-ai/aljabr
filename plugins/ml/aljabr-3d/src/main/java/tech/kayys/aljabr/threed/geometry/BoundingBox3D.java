package tech.kayys.aljabr.threed.geometry;

import java.util.List;

/**
 * Axis-Aligned Bounding Box (AABB) in 3D space.
 * @author bhangun
 */
public record BoundingBox3D(Point3D min, Point3D max) {

    public static final BoundingBox3D EMPTY = new BoundingBox3D(Point3D.ZERO, Point3D.ZERO);

    public static BoundingBox3D of(Point3D min, Point3D max) {
        return new BoundingBox3D(min, max);
    }

    public static BoundingBox3D fromPoints(List<Point3D> points) {
        if (points == null || points.isEmpty()) return EMPTY;
        double minX = Double.POSITIVE_INFINITY, minY = Double.POSITIVE_INFINITY, minZ = Double.POSITIVE_INFINITY;
        double maxX = Double.NEGATIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY, maxZ = Double.NEGATIVE_INFINITY;

        for (Point3D p : points) {
            minX = Math.min(minX, p.x());
            minY = Math.min(minY, p.y());
            minZ = Math.min(minZ, p.z());
            maxX = Math.max(maxX, p.x());
            maxY = Math.max(maxY, p.y());
            maxZ = Math.max(maxZ, p.z());
        }
        return new BoundingBox3D(Point3D.of(minX, minY, minZ), Point3D.of(maxX, maxY, maxZ));
    }

    public Point3D center() {
        return min.add(max).multiply(0.5);
    }

    public Point3D extents() {
        return max.subtract(min);
    }

    public double diameter() {
        return extents().norm();
    }

    public boolean contains(Point3D p) {
        return p.x() >= min.x() && p.x() <= max.x()
                && p.y() >= min.y() && p.y() <= max.y()
                && p.z() >= min.z() && p.z() <= max.z();
    }
}
