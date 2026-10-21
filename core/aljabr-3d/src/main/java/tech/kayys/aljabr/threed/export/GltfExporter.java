package tech.kayys.aljabr.threed.export;

import tech.kayys.aljabr.threed.geometry.Face3D;
import tech.kayys.aljabr.threed.geometry.Mesh3D;
import tech.kayys.aljabr.threed.geometry.Vertex3D;

import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.Locale;

/**
 * Exporter for glTF 2.0 (Graphics Language Transmission Format) with embedded binary buffer.
 * @author bhangun
 */
public final class GltfExporter {

    public static String exportToGltfJson(Mesh3D mesh) {
        int vCount = mesh.vertexCount();
        int fCount = mesh.faceCount();

        ByteBuffer posBuffer = ByteBuffer.allocate(vCount * 12).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer normBuffer = ByteBuffer.allocate(vCount * 12).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer idxBuffer = ByteBuffer.allocate(fCount * 3 * 2).order(ByteOrder.LITTLE_ENDIAN);

        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE, minZ = Double.MAX_VALUE;
        double maxX = -Double.MAX_VALUE, maxY = -Double.MAX_VALUE, maxZ = -Double.MAX_VALUE;

        for (Vertex3D v : mesh.vertices()) {
            posBuffer.putFloat((float) v.position().x());
            posBuffer.putFloat((float) v.position().y());
            posBuffer.putFloat((float) v.position().z());

            minX = Math.min(minX, v.position().x());
            minY = Math.min(minY, v.position().y());
            minZ = Math.min(minZ, v.position().z());
            maxX = Math.max(maxX, v.position().x());
            maxY = Math.max(maxY, v.position().y());
            maxZ = Math.max(maxZ, v.position().z());

            normBuffer.putFloat((float) v.normal().x());
            normBuffer.putFloat((float) v.normal().y());
            normBuffer.putFloat((float) v.normal().z());
        }

        int indexCount = 0;
        for (Face3D f : mesh.faces()) {
            if (f.size() >= 3) {
                idxBuffer.putShort((short) f.get(0));
                idxBuffer.putShort((short) f.get(1));
                idxBuffer.putShort((short) f.get(2));
                indexCount += 3;
            }
        }

        ByteArrayOutputStream combined = new ByteArrayOutputStream();
        try {
            combined.write(posBuffer.array());
            combined.write(normBuffer.array());
            combined.write(idxBuffer.array());
        } catch (IOException ignored) {}

        String base64Data = Base64.getEncoder().encodeToString(combined.toByteArray());
        int posOffset = 0;
        int normOffset = vCount * 12;
        int idxOffset = normOffset + vCount * 12;
        int totalBytes = combined.size();

        return String.format(Locale.US,
            "{\n" +
            "  \"asset\": { \"version\": \"2.0\", \"generator\": \"Aljabr 3D Engine\" },\n" +
            "  \"scenes\": [{ \"nodes\": [0] }],\n" +
            "  \"nodes\": [{ \"mesh\": 0, \"name\": \"%s\" }],\n" +
            "  \"meshes\": [{\n" +
            "    \"name\": \"%s\",\n" +
            "    \"primitives\": [{\n" +
            "      \"attributes\": { \"POSITION\": 0, \"NORMAL\": 1 },\n" +
            "      \"indices\": 2,\n" +
            "      \"mode\": 4\n" +
            "    }]\n" +
            "  }],\n" +
            "  \"accessors\": [\n" +
            "    { \"bufferView\": 0, \"byteOffset\": 0, \"componentType\": 5126, \"count\": %d, \"type\": \"VEC3\", \"max\": [%.4f, %.4f, %.4f], \"min\": [%.4f, %.4f, %.4f] },\n" +
            "    { \"bufferView\": 1, \"byteOffset\": 0, \"componentType\": 5126, \"count\": %d, \"type\": \"VEC3\" },\n" +
            "    { \"bufferView\": 2, \"byteOffset\": 0, \"componentType\": 5123, \"count\": %d, \"type\": \"SCALAR\" }\n" +
            "  ],\n" +
            "  \"bufferViews\": [\n" +
            "    { \"buffer\": 0, \"byteOffset\": %d, \"byteLength\": %d, \"target\": 34962 },\n" +
            "    { \"buffer\": 0, \"byteOffset\": %d, \"byteLength\": %d, \"target\": 34962 },\n" +
            "    { \"buffer\": 0, \"byteOffset\": %d, \"byteLength\": %d, \"target\": 34963 }\n" +
            "  ],\n" +
            "  \"buffers\": [{\n" +
            "    \"byteLength\": %d,\n" +
            "    \"uri\": \"data:application/octet-stream;base64,%s\"\n" +
            "  }]\n" +
            "}",
            mesh.name(), mesh.name(), vCount, maxX, maxY, maxZ, minX, minY, minZ, vCount, indexCount,
            posOffset, vCount * 12, normOffset, vCount * 12, idxOffset, indexCount * 2, totalBytes, base64Data);
    }

    public static void exportToFile(Mesh3D mesh, Path targetPath) throws IOException {
        Files.createDirectories(targetPath.getParent());
        try (BufferedWriter writer = Files.newBufferedWriter(targetPath)) {
            writer.write(exportToGltfJson(mesh));
        }
    }
}
