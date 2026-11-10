package tech.kayys.aljabr.threed.geometry;

import java.util.Objects;

/**
 * A 3D vertex with position, normal, UV texture coordinates, and RGB color.
 * @author bhangun
 */
public record Vertex3D(
        Point3D position,
        Point3D normal,
        double u,
        double v,
        float r,
        float g,
        float b
) {
    public Vertex3D {
        Objects.requireNonNull(position, "position must not be null");
        if (normal == null) normal = Point3D.UNIT_Y;
    }

    public static Vertex3D of(Point3D position) {
        return new Vertex3D(position, Point3D.UNIT_Y, 0, 0, 0.8f, 0.8f, 0.8f);
    }

    public static Vertex3D of(Point3D position, Point3D normal) {
        return new Vertex3D(position, normal, 0, 0, 0.8f, 0.8f, 0.8f);
    }

    public static Vertex3D of(double x, double y, double z) {
        return of(Point3D.of(x, y, z));
    }

    public Vertex3D withNormal(Point3D newNormal) {
        return new Vertex3D(position, newNormal, u, v, r, g, b);
    }

    public Vertex3D withUV(double newU, double newV) {
        return new Vertex3D(position, normal, newU, newV, r, g, b);
    }

    public Vertex3D withColor(float newR, float newG, float newB) {
        return new Vertex3D(position, normal, u, v, newR, newG, newB);
    }
}
