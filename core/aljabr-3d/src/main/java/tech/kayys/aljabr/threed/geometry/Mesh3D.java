package tech.kayys.aljabr.threed.geometry;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * 3D polygonal mesh consisting of vertices, indexed polygonal faces, and bounding box.
 * @author bhangun
 */
public final class Mesh3D {

    private final String name;
    private final List<Vertex3D> vertices;
    private final List<Face3D> faces;

    public Mesh3D(String name, List<Vertex3D> vertices, List<Face3D> faces) {
        this.name = name != null ? name : "mesh_3d";
        this.vertices = new ArrayList<>(Objects.requireNonNull(vertices, "vertices must not be null"));
        this.faces = new ArrayList<>(Objects.requireNonNull(faces, "faces must not be null"));
    }

    public static Mesh3D of(String name, List<Vertex3D> vertices, List<Face3D> faces) {
        return new Mesh3D(name, vertices, faces);
    }

    public static Mesh3D cube(double size) {
        double s = size * 0.5;
        List<Vertex3D> verts = List.of(
                Vertex3D.of(-s, -s, -s), Vertex3D.of(s, -s, -s),
                Vertex3D.of(s, s, -s),   Vertex3D.of(-s, s, -s),
                Vertex3D.of(-s, -s, s),  Vertex3D.of(s, -s, s),
                Vertex3D.of(s, s, s),    Vertex3D.of(-s, s, s)
        );

        List<Face3D> f = List.of(
                Face3D.triangle(0, 2, 1), Face3D.triangle(0, 3, 2),
                Face3D.triangle(4, 5, 6), Face3D.triangle(4, 6, 7),
                Face3D.triangle(0, 1, 5), Face3D.triangle(0, 5, 4),
                Face3D.triangle(3, 6, 2), Face3D.triangle(3, 7, 6),
                Face3D.triangle(0, 4, 7), Face3D.triangle(0, 7, 3),
                Face3D.triangle(1, 2, 6), Face3D.triangle(1, 6, 5)
        );

        Mesh3D mesh = new Mesh3D("cube", verts, f);
        mesh.recomputeNormals();
        return mesh;
    }

    public String name() { return name; }
    public List<Vertex3D> vertices() { return Collections.unmodifiableList(vertices); }
    public List<Face3D> faces() { return Collections.unmodifiableList(faces); }
    public int vertexCount() { return vertices.size(); }
    public int faceCount() { return faces.size(); }

    public BoundingBox3D boundingBox() {
        List<Point3D> points = vertices.stream().map(Vertex3D::position).toList();
        return BoundingBox3D.fromPoints(points);
    }

    public void recomputeNormals() {
        Point3D[] accumulated = new Point3D[vertices.size()];
        for (int i = 0; i < vertices.size(); i++) accumulated[i] = Point3D.ZERO;

        for (int fi = 0; fi < faces.size(); fi++) {
            Face3D f = faces.get(fi);
            if (f.size() >= 3) {
                Point3D p0 = vertices.get(f.get(0)).position();
                Point3D p1 = vertices.get(f.get(1)).position();
                Point3D p2 = vertices.get(f.get(2)).position();

                Point3D faceNormal = p1.subtract(p0).cross(p2.subtract(p0)).normalize();
                faces.set(fi, new Face3D(f.indices(), faceNormal));

                for (int idx : f.indices()) {
                    accumulated[idx] = accumulated[idx].add(faceNormal);
                }
            }
        }

        for (int i = 0; i < vertices.size(); i++) {
            Point3D n = accumulated[i].normalize();
            if (n.norm() < 1e-6) n = Point3D.UNIT_Y;
            vertices.set(i, vertices.get(i).withNormal(n));
        }
    }

    public Mesh3D scale(double factor) {
        List<Vertex3D> newVerts = vertices.stream()
                .map(v -> new Vertex3D(v.position().multiply(factor), v.normal(), v.u(), v.v(), v.r(), v.g(), v.b()))
                .toList();
        return new Mesh3D(name, newVerts, faces);
    }

    public Mesh3D translate(Point3D delta) {
        List<Vertex3D> newVerts = vertices.stream()
                .map(v -> new Vertex3D(v.position().add(delta), v.normal(), v.u(), v.v(), v.r(), v.g(), v.b()))
                .toList();
        return new Mesh3D(name, newVerts, faces);
    }
}
