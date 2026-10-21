package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.core.GgmlType;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GgufDenseRouteTest {
    @Test
    void resolvesDenseSubtypeRoutesByTypeId() {
        assertEquals(GgufDenseRoute.F32, GgufDenseRoute.forType(GgmlType.F32.id));
        assertEquals(GgufDenseRoute.F16, GgufDenseRoute.forType(GgmlType.F16.id));
        assertEquals(GgufDenseRoute.BF16, GgufDenseRoute.forType(GgmlType.BF16.id));
    }

    @Test
    void returnsUnsupportedForNonDenseIds() {
        assertEquals(GgufDenseRoute.UNSUPPORTED, GgufDenseRoute.forType(GgmlType.Q4_0.id));
        assertEquals(GgufDenseRoute.UNSUPPORTED, GgufDenseRoute.forType(GgmlType.Q8_0.id));
        assertEquals(GgufDenseRoute.UNSUPPORTED, GgufDenseRoute.forType(-1));
        assertEquals(GgufDenseRoute.UNSUPPORTED, GgufDenseRoute.forType(10_000));
    }
}
