package tech.kayys.aljabr.threed.geometry;

import java.util.Arrays;
import java.util.Objects;

/**
 * Polygonal face referencing 3 or 4 vertex indices with surface normal.
 * @author bhangun
 */
public record Face3D(int[] indices, Point3D normal) {

    public Face3D {
        Objects.requireNonNull(indices, "indices must not be null");
        if (indices.length < 3) {
            throw new IllegalArgumentException("A face must contain at least 3 vertices");
        }
    }

    public static Face3D triangle(int i0, int i1, int i2) {
        return new Face3D(new int[]{i0, i1, i2}, Point3D.UNIT_Y);
    }

    public static Face3D triangle(int i0, int i1, int i2, Point3D normal) {
        return new Face3D(new int[]{i0, i1, i2}, normal);
    }

    public static Face3D quad(int i0, int i1, int i2, int i3) {
        return new Face3D(new int[]{i0, i1, i2, i3}, Point3D.UNIT_Y);
    }

    public boolean isTriangle() {
        return indices.length == 3;
    }

    public int size() {
        return indices.length;
    }

    public int get(int idx) {
        return indices[idx];
    }
}
