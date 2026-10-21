package tech.kayys.aljabr.threed.voxel;

import tech.kayys.aljabr.threed.geometry.BoundingBox3D;
import tech.kayys.aljabr.threed.geometry.Point3D;

/**
 * 3D discrete volumetric grid storing scalar densities or occupancies.
 * @author bhangun
 */
public final class VoxelGrid3D {

    private final int width;
    private final int height;
    private final int depth;
    private final double[] values;
    private final BoundingBox3D bounds;

    public VoxelGrid3D(int width, int height, int depth, BoundingBox3D bounds) {
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.bounds = bounds != null ? bounds : BoundingBox3D.of(Point3D.of(-1, -1, -1), Point3D.of(1, 1, 1));
        this.values = new double[width * height * depth];
    }

    public int width() { return width; }
    public int height() { return height; }
    public int depth() { return depth; }
    public BoundingBox3D bounds() { return bounds; }

    public double get(int x, int y, int z) {
        if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth) return 0.0;
        return values[x + y * width + z * width * height];
    }

    public void set(int x, int y, int z, double value) {
        if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth) {
            values[x + y * width + z * width * height] = value;
        }
    }

    public double sampleTrilinear(double normX, double normY, double normZ) {
        double gx = Math.max(0, Math.min(width - 1, normX * (width - 1)));
        double gy = Math.max(0, Math.min(height - 1, normY * (height - 1)));
        double gz = Math.max(0, Math.min(depth - 1, normZ * (depth - 1)));

        int x0 = (int) gx, x1 = Math.min(width - 1, x0 + 1);
        int y0 = (int) gy, y1 = Math.min(height - 1, y0 + 1);
        int z0 = (int) gz, z1 = Math.min(depth - 1, z0 + 1);

        double tx = gx - x0, ty = gy - y0, tz = gz - z0;

        double c00 = get(x0, y0, z0) * (1 - tx) + get(x1, y0, z0) * tx;
        double c10 = get(x0, y1, z0) * (1 - tx) + get(x1, y1, z0) * tx;
        double c01 = get(x0, y0, z1) * (1 - tx) + get(x1, y0, z1) * tx;
        double c11 = get(x0, y1, z1) * (1 - tx) + get(x1, y1, z1) * tx;

        double c0 = c00 * (1 - ty) + c10 * ty;
        double c1 = c01 * (1 - ty) + c11 * ty;

        return c0 * (1 - tz) + c1 * tz;
    }
}
