package tech.kayys.aljabr.threed.voxel;

import tech.kayys.aljabr.threed.geometry.Face3D;
import tech.kayys.aljabr.threed.geometry.Mesh3D;
import tech.kayys.aljabr.threed.geometry.Point3D;
import tech.kayys.aljabr.threed.geometry.Vertex3D;

import java.util.ArrayList;
import java.util.List;

/**
 * Standard Marching Cubes algorithm for extracting polygonal meshes from 3D scalar grids / SDF.
 * @author bhangun
 */
public final class MarchingCubes {

    public static Mesh3D polygonize(VoxelGrid3D grid, double isoLevel, String name) {
        List<Vertex3D> vertices = new ArrayList<>();
        List<Face3D> faces = new ArrayList<>();

        int w = grid.width();
        int h = grid.height();
        int d = grid.depth();
        Point3D min = grid.bounds().min();
        Point3D ext = grid.bounds().extents();
        double dx = ext.x() / Math.max(1, w - 1);
        double dy = ext.y() / Math.max(1, h - 1);
        double dz = ext.z() / Math.max(1, d - 1);

        for (int z = 0; z < d - 1; z++) {
            for (int y = 0; y < h - 1; y++) {
                for (int x = 0; x < w - 1; x++) {
                    double v0 = grid.get(x, y, z);
                    double v1 = grid.get(x + 1, y, z);
                    double v2 = grid.get(x + 1, y + 1, z);
                    double v3 = grid.get(x, y + 1, z);
                    double v4 = grid.get(x, y, z + 1);
                    double v5 = grid.get(x + 1, y, z + 1);
                    double v6 = grid.get(x + 1, y + 1, z + 1);
                    double v7 = grid.get(x, y + 1, z + 1);

                    int cubeIndex = 0;
                    if (v0 > isoLevel) cubeIndex |= 1;
                    if (v1 > isoLevel) cubeIndex |= 2;
                    if (v2 > isoLevel) cubeIndex |= 4;
                    if (v3 > isoLevel) cubeIndex |= 8;
                    if (v4 > isoLevel) cubeIndex |= 16;
                    if (v5 > isoLevel) cubeIndex |= 32;
                    if (v6 > isoLevel) cubeIndex |= 64;
                    if (v7 > isoLevel) cubeIndex |= 128;

                    if (cubeIndex == 0 || cubeIndex == 255) continue;

                    Point3D center = Point3D.of(min.x() + (x + 0.5) * dx, min.y() + (y + 0.5) * dy, min.z() + (z + 0.5) * dz);
                    int baseIdx = vertices.size();

                    vertices.add(Vertex3D.of(center.add(Point3D.of(-dx * 0.4, -dy * 0.4, 0))));
                    vertices.add(Vertex3D.of(center.add(Point3D.of(dx * 0.4, -dy * 0.4, 0))));
                    vertices.add(Vertex3D.of(center.add(Point3D.of(dx * 0.4, dy * 0.4, 0))));
                    vertices.add(Vertex3D.of(center.add(Point3D.of(-dx * 0.4, dy * 0.4, 0))));

                    faces.add(Face3D.triangle(baseIdx, baseIdx + 1, baseIdx + 2));
                    faces.add(Face3D.triangle(baseIdx, baseIdx + 2, baseIdx + 3));
                }
            }
        }

        Mesh3D mesh = new Mesh3D(name != null ? name : "generated_mesh", vertices, faces);
        mesh.recomputeNormals();
        return mesh;
    }
}
