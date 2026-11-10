package tech.kayys.aljabr.threed;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.threed.export.GltfExporter;
import tech.kayys.aljabr.threed.export.ObjExporter;
import tech.kayys.aljabr.threed.export.PlyExporter;
import tech.kayys.aljabr.threed.geometry.GaussianSplat3D;
import tech.kayys.aljabr.threed.geometry.Mesh3D;
import tech.kayys.aljabr.threed.geometry.Point3D;
import tech.kayys.aljabr.threed.voxel.MarchingCubes;
import tech.kayys.aljabr.threed.voxel.VoxelGrid3D;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class Mesh3DTest {

    @Test
    void testCubeMeshCreation() {
        Mesh3D cube = Mesh3D.cube(2.0);
        assertEquals(8, cube.vertexCount());
        assertEquals(12, cube.faceCount());
        assertEquals(2.0, cube.boundingBox().extents().x(), 1e-5);
    }

    @Test
    void testObjExporter() {
        Mesh3D cube = Mesh3D.cube(1.0);
        String obj = ObjExporter.exportToString(cube);
        assertTrue(obj.contains("v -0.500000 -0.500000 -0.500000"));
        assertTrue(obj.contains("f 1/1/1 3/3/3 2/2/2"));
    }

    @Test
    void testPlyExporter() {
        Mesh3D cube = Mesh3D.cube(1.0);
        String ply = PlyExporter.exportMesh(cube);
        assertTrue(ply.contains("element vertex 8"));
        assertTrue(ply.contains("element face 12"));

        List<GaussianSplat3D> splats = List.of(
                GaussianSplat3D.of(Point3D.of(0, 0, 0), Point3D.of(0.1, 0.1, 0.1), 0.9, 1.0, 0.5, 0.2)
        );
        String splatPly = PlyExporter.exportSplats(splats);
        assertTrue(splatPly.contains("element vertex 1"));
    }

    @Test
    void testGltfExporter() {
        Mesh3D cube = Mesh3D.cube(1.0);
        String gltf = GltfExporter.exportToGltfJson(cube);
        assertTrue(gltf.contains("\"generator\": \"Aljabr 3D Engine\""));
        assertTrue(gltf.contains("data:application/octet-stream;base64,"));
    }

    @Test
    void testVoxelMarchingCubes() {
        VoxelGrid3D grid = new VoxelGrid3D(8, 8, 8, null);
        grid.set(4, 4, 4, 1.0);
        Mesh3D mesh = MarchingCubes.polygonize(grid, 0.5, "voxel_sphere");
        assertNotNull(mesh);
        assertTrue(mesh.vertexCount() > 0);
    }
}
