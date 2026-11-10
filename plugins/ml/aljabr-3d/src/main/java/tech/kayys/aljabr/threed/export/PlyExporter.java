package tech.kayys.aljabr.threed.export;

import tech.kayys.aljabr.threed.geometry.Face3D;
import tech.kayys.aljabr.threed.geometry.GaussianSplat3D;
import tech.kayys.aljabr.threed.geometry.Mesh3D;
import tech.kayys.aljabr.threed.geometry.Vertex3D;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * Exporter for Stanford PLY (ASCII format for meshes and 3D Gaussian Splats).
 * @author bhangun
 */
public final class PlyExporter {

    public static String exportMesh(Mesh3D mesh) {
        StringBuilder sb = new StringBuilder();
        sb.append("ply\nformat ascii 1.0\n");
        sb.append("comment Aljabr 3D Engine\n");
        sb.append("element vertex ").append(mesh.vertexCount()).append("\n");
        sb.append("property float x\nproperty float y\nproperty float z\n");
        sb.append("property float nx\nproperty float ny\nproperty float nz\n");
        sb.append("property uchar red\nproperty uchar green\nproperty uchar blue\n");
        sb.append("element face ").append(mesh.faceCount()).append("\n");
        sb.append("property list uchar int vertex_indices\n");
        sb.append("end_header\n");

        for (Vertex3D v : mesh.vertices()) {
            int r = Math.min(255, Math.max(0, (int) (v.r() * 255)));
            int g = Math.min(255, Math.max(0, (int) (v.g() * 255)));
            int b = Math.min(255, Math.max(0, (int) (v.b() * 255)));
            sb.append(String.format(Locale.US, "%.6f %.6f %.6f %.6f %.6f %.6f %d %d %d\n",
                    v.position().x(), v.position().y(), v.position().z(),
                    v.normal().x(), v.normal().y(), v.normal().z(),
                    r, g, b));
        }

        for (Face3D f : mesh.faces()) {
            sb.append(f.size());
            for (int idx : f.indices()) {
                sb.append(" ").append(idx);
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public static String exportSplats(List<GaussianSplat3D> splats) {
        StringBuilder sb = new StringBuilder();
        sb.append("ply\nformat ascii 1.0\n");
        sb.append("element vertex ").append(splats.size()).append("\n");
        sb.append("property float x\nproperty float y\nproperty float z\n");
        sb.append("property float scale_0\nproperty float scale_1\nproperty float scale_2\n");
        sb.append("property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n");
        sb.append("property float opacity\n");
        sb.append("property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n");
        sb.append("end_header\n");

        for (GaussianSplat3D s : splats) {
            double[] q = s.rotationQuaternion();
            double[] sh = s.sphericalHarmonics();
            sb.append(String.format(Locale.US, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    s.position().x(), s.position().y(), s.position().z(),
                    s.scale().x(), s.scale().y(), s.scale().z(),
                    q[0], q[1], q[2], q[3],
                    s.opacity(),
                    sh[0], sh[1], sh[2]));
        }
        return sb.toString();
    }

    public static void exportToFile(Mesh3D mesh, Path targetPath) throws IOException {
        Files.createDirectories(targetPath.getParent());
        try (BufferedWriter writer = Files.newBufferedWriter(targetPath)) {
            writer.write(exportMesh(mesh));
        }
    }
}
