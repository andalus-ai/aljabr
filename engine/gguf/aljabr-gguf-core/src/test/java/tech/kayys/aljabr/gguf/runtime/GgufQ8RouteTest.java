package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.core.GgmlType;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufQ8RouteTest {
    @Test
    void resolvesQ8FamilySubtypeRoutesByTypeId() {
        assertEquals(GgufQ8Route.Q1_0, GgufQ8Route.forType(GgmlType.Q1_0.id));
        assertEquals(GgufQ8Route.TQ1_0, GgufQ8Route.forType(GgmlType.TQ1_0.id));
        assertEquals(GgufQ8Route.TQ2_0, GgufQ8Route.forType(GgmlType.TQ2_0.id));
        assertEquals(GgufQ8Route.MXFP4, GgufQ8Route.forType(GgmlType.MXFP4.id));
        assertEquals(GgufQ8Route.NVFP4, GgufQ8Route.forType(GgmlType.NVFP4.id));
        assertEquals(GgufQ8Route.Q8_0, GgufQ8Route.forType(GgmlType.Q8_0.id));
        assertEquals(GgufQ8Route.Q8_1, GgufQ8Route.forType(GgmlType.Q8_1.id));
        assertEquals(GgufQ8Route.Q8_K, GgufQ8Route.forType(GgmlType.Q8_K.id));
        assertEquals(GgufQ8Route.IQ4_NL, GgufQ8Route.forType(GgmlType.IQ4_NL.id));
        assertEquals(GgufQ8Route.IQ4_XS, GgufQ8Route.forType(GgmlType.IQ4_XS.id));
    }

    @Test
    void returnsUnsupportedForNonQ8FamilyIds() {
        assertEquals(GgufQ8Route.UNSUPPORTED, GgufQ8Route.forType(GgmlType.F32.id));
        assertEquals(GgufQ8Route.UNSUPPORTED, GgufQ8Route.forType(GgmlType.Q2_K.id));
        assertEquals(GgufQ8Route.UNSUPPORTED, GgufQ8Route.forType(-1));
        assertEquals(GgufQ8Route.UNSUPPORTED, GgufQ8Route.forType(10_000));
    }
}
