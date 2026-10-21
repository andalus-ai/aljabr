package tech.kayys.aljabr.threed.geometry;

import java.util.Arrays;
import java.util.Objects;

/**
 * 3D Gaussian Splatting primitive.
 * Contains position, scale (3D), rotation quaternion (4D), opacity, and Spherical Harmonics color coefficients.
 * @author bhangun
 */
public record GaussianSplat3D(
        Point3D position,
        Point3D scale,
        double[] rotationQuaternion,
        double opacity,
        double[] sphericalHarmonics
) {
    public GaussianSplat3D {
        Objects.requireNonNull(position, "position must not be null");
        if (scale == null) scale = Point3D.of(0.01, 0.01, 0.01);
        if (rotationQuaternion == null || rotationQuaternion.length != 4) {
            rotationQuaternion = new double[]{1.0, 0.0, 0.0, 0.0};
        }
        if (sphericalHarmonics == null) {
            sphericalHarmonics = new double[]{0.5, 0.5, 0.5};
        }
    }

    public static GaussianSplat3D of(Point3D position, Point3D scale, double opacity, double r, double g, double b) {
        return new GaussianSplat3D(
                position,
                scale,
                new double[]{1.0, 0.0, 0.0, 0.0},
                opacity,
                new double[]{r, g, b}
        );
    }
}
