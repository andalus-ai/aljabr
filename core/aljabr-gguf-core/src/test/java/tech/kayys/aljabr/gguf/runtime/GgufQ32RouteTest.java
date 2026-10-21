package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.core.GgmlType;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufQ32RouteTest {
    @Test
    void resolvesQ32FamilySubtypeRoutesByTypeId() {
        assertEquals(GgufQ32Route.Q4_0, GgufQ32Route.forType(GgmlType.Q4_0.id));
        assertEquals(GgufQ32Route.Q4_1, GgufQ32Route.forType(GgmlType.Q4_1.id));
        assertEquals(GgufQ32Route.Q5_0, GgufQ32Route.forType(GgmlType.Q5_0.id));
        assertEquals(GgufQ32Route.Q5_1, GgufQ32Route.forType(GgmlType.Q5_1.id));
    }

    @Test
    void returnsUnsupportedForNonQ32FamilyIds() {
        assertEquals(GgufQ32Route.UNSUPPORTED, GgufQ32Route.forType(GgmlType.F32.id));
        assertEquals(GgufQ32Route.UNSUPPORTED, GgufQ32Route.forType(GgmlType.Q8_0.id));
        assertEquals(GgufQ32Route.UNSUPPORTED, GgufQ32Route.forType(-1));
        assertEquals(GgufQ32Route.UNSUPPORTED, GgufQ32Route.forType(10_000));
    }

    @Test
    void reportsOnlyBiasCarryingQ32Routes() {
        assertFalse(GgufQ32Route.hasBlockBiases(GgufQ32Route.Q4_0));
        assertTrue(GgufQ32Route.hasBlockBiases(GgufQ32Route.Q4_1));
        assertFalse(GgufQ32Route.hasBlockBiases(GgufQ32Route.Q5_0));
        assertTrue(GgufQ32Route.hasBlockBiases(GgufQ32Route.Q5_1));
        assertFalse(GgufQ32Route.hasBlockBiases(GgufQ32Route.UNSUPPORTED));
    }
}
