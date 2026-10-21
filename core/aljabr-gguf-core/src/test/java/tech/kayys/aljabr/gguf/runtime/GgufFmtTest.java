package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.core.GgmlType;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufFmtTest {
    @Test
    void answersHotTypePredicatesFromTables() {
        assertTrue(GgufQuantFormats.supportsQ32PreparedType(GgmlType.Q4_0.id));
        assertTrue(GgufQuantFormats.supportsQ32PreparedType(GgmlType.Q5_1.id));
        assertFalse(GgufQuantFormats.supportsQ32PreparedType(GgmlType.Q8_0.id));

        assertTrue(GgufQuantFormats.supportsPreparedMatVecType(GgmlType.Q4_K.id));
        assertTrue(GgufQuantFormats.supportsPreparedMatVecType(GgmlType.IQ4_XS.id));
        assertFalse(GgufQuantFormats.supportsPreparedMatVecType(GgmlType.F32.id));

        assertTrue(GgufQuantFormats.supportsRowDotType(GgmlType.BF16.id));
        assertTrue(GgufQuantFormats.supportsRowDotType(GgmlType.NVFP4.id));
        assertFalse(GgufQuantFormats.supportsRowDotType(GgmlType.I8.id));

        assertTrue(GgufQuantFormats.usesQ8PreparedCache(GgmlType.TQ2_0.id));
        assertFalse(GgufQuantFormats.usesQ8PreparedCache(GgmlType.Q2_K.id));

        assertTrue(GgufQuantFormats.q32PreparedHasBlockBiases(GgmlType.Q4_1.id));
        assertFalse(GgufQuantFormats.q32PreparedHasBlockBiases(GgmlType.Q4_0.id));
        assertFalse(GgufQuantFormats.q32PreparedHasBlockBiases(-1));
    }

    @Test
    void returnsPreparedBlockLayoutFromTables() {
        assertEquals(GgufQuantFormats.Q1_0_BLOCK_SIZE, GgufQuantFormats.q8BlockSize(GgmlType.Q1_0.id));
        assertEquals(GgufQuantFormats.TQ1_0_BLOCK_BYTES, GgufQuantFormats.q8BlockBytes(GgmlType.TQ1_0.id));
        assertEquals(GgufQuantFormats.NVFP4_SUB_BLOCK_SIZE, GgufQuantFormats.q8BlockSize(GgmlType.NVFP4.id));
        assertEquals(GgufQuantFormats.IQ4_XS_BLOCK_BYTES, GgufQuantFormats.q8BlockBytes(GgmlType.IQ4_XS.id));

        assertEquals(GgufQuantFormats.Q4_0_BLOCK_BYTES, GgufQuantFormats.q32BlockBytes(GgmlType.Q4_0.id));
        assertEquals(GgufQuantFormats.Q5_1_BLOCK_BYTES, GgufQuantFormats.q32BlockBytes(GgmlType.Q5_1.id));

        assertThrows(IllegalArgumentException.class, () -> GgufQuantFormats.q8BlockSize(GgmlType.F32.id));
        assertThrows(IllegalArgumentException.class, () -> GgufQuantFormats.q8BlockBytes(-1));
        assertThrows(IllegalArgumentException.class, () -> GgufQuantFormats.q32BlockBytes(GgmlType.Q8_0.id));
    }
}
