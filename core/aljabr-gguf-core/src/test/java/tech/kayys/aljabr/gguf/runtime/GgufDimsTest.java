package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufDimsTest {
    @Test
    void checkedLayoutReturnsColumnsAndValidatedRowBytesTogether() {
        GGUFTensorInfo tensor = new GGUFTensorInfo("f32", new long[]{4, 2}, GgmlType.F32.id, 0, 32);

        GgufTensorShape.MatrixLayout layout = GgufTensorShape.checkedLayout(tensor, 4);

        assertEquals(4, layout.columns());
        assertEquals(2L, layout.rows());
        assertEquals(16L, layout.rowBytes());
        assertEquals(4, GgufTensorShape.checkedColumns(tensor, 4));
    }

    @Test
    void checkedLayoutReusesCachedLayoutForSameTensor() {
        GgufTensorShape.clearRecentDimCache();
        assertEquals(0, GgufTensorShape.recentShapeFastCacheSize());
        assertEquals(0, GgufTensorShape.recentLayoutFastCacheSize());
        GGUFTensorInfo tensor = new GGUFTensorInfo("cached.f16", new long[]{4, 3}, GgmlType.F16.id, 0, 24);

        try {
            GgufTensorShape.MatrixLayout first = GgufTensorShape.checkedLayout(tensor, 4);
            assertEquals(1, GgufTensorShape.recentShapeFastCacheSize());
            assertEquals(1, GgufTensorShape.recentLayoutFastCacheSize());
            GgufTensorShape.MatrixLayout second = GgufTensorShape.checkedLayout(tensor, 8);

            assertSame(first, second);
            assertEquals(1, GgufTensorShape.recentShapeFastCacheSize());
            assertEquals(1, GgufTensorShape.recentLayoutFastCacheSize());
            assertThrows(IllegalArgumentException.class, () -> GgufTensorShape.checkedLayout(tensor, 3));
        } finally {
            GgufTensorShape.clearRecentDimCache();
        }
    }

    @Test
    void checkedLayoutReusesRecentLayoutsAcrossAlternatingTensors() {
        GGUFTensorInfo first = new GGUFTensorInfo("cached.first", new long[]{4, 3}, GgmlType.F16.id, 0, 24);
        GGUFTensorInfo second = new GGUFTensorInfo("cached.second", new long[]{8, 2}, GgmlType.F32.id, 24, 64);

        GgufTensorShape.MatrixLayout firstLayout = GgufTensorShape.checkedLayout(first, 4);
        GgufTensorShape.MatrixLayout secondLayout = GgufTensorShape.checkedLayout(second, 8);

        assertSame(firstLayout, GgufTensorShape.checkedLayout(first, 4));
        assertSame(secondLayout, GgufTensorShape.checkedLayout(second, 8));
        assertEquals(3L, GgufTensorShape.matrixRows(first));
        assertEquals(2L, GgufTensorShape.matrixRows(second));
    }

    @Test
    void matrixShapeHelpersReuseShapeWithoutRequiringLoadableType() {
        GGUFTensorInfo tensor =
                new GGUFTensorInfo("shape.only", new long[]{4, 3, 2}, GgmlType.Q4_0_4_4.id, 0, 0);

        assertEquals(4L, GgufTensorShape.matrixColumns(tensor));
        assertEquals(6L, GgufTensorShape.matrixRows(tensor));
        assertEquals(6, GgufTensorShape.checkedRows(tensor));
        assertThrows(IllegalArgumentException.class, () -> GgufTensorShape.checkedLayout(tensor, 4));
    }

    @Test
    void checkedLayoutPreservesVectorAndFormatValidation() {
        GGUFTensorInfo tensor = new GGUFTensorInfo("q4_0", new long[]{33, 1}, GgmlType.Q4_0.id, 0, 18);

        assertThrows(IllegalArgumentException.class, () -> GgufTensorShape.checkedLayout(tensor, 32));
        assertThrows(IllegalArgumentException.class, () -> GgufTensorShape.checkedLayout(tensor, 33));
    }

    @Test
    void rowByteSizeUsesIndexedTypeSizingForLoadableFormats() {
        assertEquals(128L, GgufTensorShape.rowByteSize(GgmlType.F32.id, 32));
        assertEquals(18L, GgufTensorShape.rowByteSize(GgmlType.Q4_0.id, 32));
        assertEquals(144L, GgufTensorShape.rowByteSize(GgmlType.Q4_K.id, 256));
        assertEquals(292L, GgufTensorShape.rowByteSize(GgmlType.Q8_K.id, 256));

        assertThrows(IllegalArgumentException.class, () -> GgufTensorShape.rowByteSize(GgmlType.Q4_0.id, 33));
        assertThrows(IllegalArgumentException.class, () -> GgufTensorShape.rowByteSize(GgmlType.Q4_0_4_4.id, 32));
        assertThrows(IllegalArgumentException.class, () -> GgufTensorShape.rowByteSize(10_000, 32));
    }
}
