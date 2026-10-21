package tech.kayys.aljabr.gguf.core;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgmlTypeTest {
    @Test
    void q8_1UsesGgmlReferenceBlockLayout() {
        assertEquals(32, GgmlType.Q8_1.blockSize);
        assertEquals(36, GgmlType.Q8_1.typeSize);
        assertEquals(36L, GgmlType.Q8_1.bytesFor(32));
        assertEquals(72L, GgmlType.Q8_1.bytesFor(64));
        assertEquals(36L, GGUFConstants.tensorBytes(GGUFConstants.GGML_TYPE_Q8_1, 32));
    }

    @Test
    void modernQuantTypesUseGgmlReferenceBlockLayouts() {
        assertEquals(54L, GgmlType.TQ1_0.bytesFor(256));
        assertEquals(66L, GgmlType.TQ2_0.bytesFor(256));
        assertEquals(17L, GgmlType.MXFP4.bytesFor(32));
        assertEquals(36L, GgmlType.NVFP4.bytesFor(64));
        assertEquals(18L, GgmlType.Q1_0.bytesFor(128));

        assertEquals(54L, GGUFConstants.tensorBytes(GGUFConstants.GGML_TYPE_TQ1_0, 256));
        assertEquals(66L, GGUFConstants.tensorBytes(GGUFConstants.GGML_TYPE_TQ2_0, 256));
        assertEquals(17L, GGUFConstants.tensorBytes(GGUFConstants.GGML_TYPE_MXFP4, 32));
        assertEquals(36L, GGUFConstants.tensorBytes(GGUFConstants.GGML_TYPE_NVFP4, 64));
        assertEquals(18L, GGUFConstants.tensorBytes(GGUFConstants.GGML_TYPE_Q1_0, 128));
    }

    @Test
    void removedRuntimeRepackedLayoutsAreNotLoadableFromGgufData() {
        assertThrows(IllegalArgumentException.class, () -> GgmlType.Q4_0_4_4.bytesFor(32));
        assertThrows(IllegalArgumentException.class, () -> GgmlType.Q4_0_4_8.bytesFor(32));
        assertThrows(IllegalArgumentException.class, () -> GgmlType.Q4_0_8_8.bytesFor(32));
        assertThrows(IllegalArgumentException.class, () -> GgmlType.IQ4_NL_4_4.bytesFor(32));
        assertThrows(IllegalArgumentException.class, () -> GgmlType.IQ4_NL_4_8.bytesFor(32));
        assertThrows(IllegalArgumentException.class, () -> GgmlType.IQ4_NL_8_8.bytesFor(32));
    }

    @Test
    void fromIdResolvesKnownIdsAndRejectsGaps() {
        for (GgmlType type : GgmlType.values()) {
            assertSame(type, GgmlType.fromId(type.id));
        }
        assertThrows(IllegalArgumentException.class, () -> GgmlType.fromId(-1));
        assertThrows(IllegalArgumentException.class, () -> GgmlType.fromId(4));
        assertThrows(IllegalArgumentException.class, () -> GgmlType.fromId(42));
    }
}
