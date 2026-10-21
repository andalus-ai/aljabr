package tech.kayys.aljabr.threed.export;

import tech.kayys.aljabr.threed.geometry.Face3D;
import tech.kayys.aljabr.threed.geometry.Mesh3D;
import tech.kayys.aljabr.threed.geometry.Vertex3D;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Exporter for Wavefront OBJ files.
 * @author bhangun
 */
public final class ObjExporter {

    public static String exportToString(Mesh3D mesh) {
        StringBuilder sb = new StringBuilder();
        sb.append("# Aljabr 3D Wavefront OBJ Exporter\n");
        sb.append("# Mesh: ").append(mesh.name()).append("\n");
        sb.append("o ").append(mesh.name()).append("\n\n");

        for (Vertex3D v : mesh.vertices()) {
            sb.append(String.format(Locale.US, "v %.6f %.6f %.6f\n",
                    v.position().x(), v.position().y(), v.position().z()));
        }

        for (Vertex3D v : mesh.vertices()) {
            sb.append(String.format(Locale.US, "vn %.6f %.6f %.6f\n",
                    v.normal().x(), v.normal().y(), v.normal().z()));
        }

        for (Vertex3D v : mesh.vertices()) {
            sb.append(String.format(Locale.US, "vt %.6f %.6f\n", v.u(), v.v()));
        }

        sb.append("\ns 1\n");
        for (Face3D f : mesh.faces()) {
            sb.append("f");
            for (int idx : f.indices()) {
                int oneBased = idx + 1;
                sb.append(" ").append(oneBased).append("/").append(oneBased).append("/").append(oneBased);
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public static void exportToFile(Mesh3D mesh, Path targetPath) throws IOException {
        Files.createDirectories(targetPath.getParent());
        try (BufferedWriter writer = Files.newBufferedWriter(targetPath)) {
            writer.write(exportToString(mesh));
        }
    }
}
