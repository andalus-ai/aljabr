package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.core.GgmlType;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GgufRawRouteTest {
    @Test
    void resolvesRawMatVecRouteByTypeId() {
        assertRoute(GgufRawRoute.ROUTE_DENSE, GgmlType.F32, GgmlType.F16, GgmlType.BF16);
        assertRoute(GgufRawRoute.ROUTE_Q32, GgmlType.Q4_0, GgmlType.Q4_1, GgmlType.Q5_0, GgmlType.Q5_1);
        assertRoute(GgufRawRoute.ROUTE_Q2K, GgmlType.Q2_K);
        assertRoute(GgufRawRoute.ROUTE_Q3K, GgmlType.Q3_K);
        assertRoute(GgufRawRoute.ROUTE_Q4K, GgmlType.Q4_K);
        assertRoute(GgufRawRoute.ROUTE_Q5K, GgmlType.Q5_K);
        assertRoute(GgufRawRoute.ROUTE_Q6K, GgmlType.Q6_K);
        assertRoute(
                GgufRawRoute.ROUTE_Q8,
                GgmlType.Q1_0,
                GgmlType.TQ1_0,
                GgmlType.TQ2_0,
                GgmlType.MXFP4,
                GgmlType.NVFP4,
                GgmlType.Q8_0,
                GgmlType.Q8_1,
                GgmlType.Q8_K,
                GgmlType.IQ4_NL,
                GgmlType.IQ4_XS);

        assertEquals(GgufRawRoute.ROUTE_FALLBACK, GgufRawRoute.routeFor(GgmlType.I8.id));
        assertEquals(GgufRawRoute.ROUTE_FALLBACK, GgufRawRoute.routeFor(-1));
        assertEquals(GgufRawRoute.ROUTE_FALLBACK, GgufRawRoute.routeFor(10_000));
    }

    @Test
    void exposesRouteAndEstimateHintFromOneEncodedLookup() {
        int q4K = GgufRawRoute.routeInfoFor(GgmlType.Q4_K.id);
        assertEquals(GgufRawRoute.ROUTE_Q4K, GgufRawRoute.routeFromInfo(q4K));
        assertTrue(GgufRawRoute.routeInfoUsesEstimateHints(q4K));

        int q8 = GgufRawRoute.routeInfoFor(GgmlType.Q8_0.id);
        assertEquals(GgufRawRoute.ROUTE_Q8, GgufRawRoute.routeFromInfo(q8));
        assertFalse(GgufRawRoute.routeInfoUsesEstimateHints(q8));

        int missing = GgufRawRoute.routeInfoFor(10_000);
        assertEquals(GgufRawRoute.ROUTE_FALLBACK, GgufRawRoute.routeFromInfo(missing));
        assertFalse(GgufRawRoute.routeInfoUsesEstimateHints(missing));
    }

    @Test
    void exposesFamilySubrouteFromSameEncodedLookup() {
        assertSubroute(GgufDenseRoute.F32, GgmlType.F32);
        assertSubroute(GgufDenseRoute.F16, GgmlType.F16);
        assertSubroute(GgufDenseRoute.BF16, GgmlType.BF16);

        assertSubroute(GgufQ32Route.Q4_0, GgmlType.Q4_0);
        assertSubroute(GgufQ32Route.Q4_1, GgmlType.Q4_1);
        assertSubroute(GgufQ32Route.Q5_0, GgmlType.Q5_0);
        assertSubroute(GgufQ32Route.Q5_1, GgmlType.Q5_1);

        assertSubroute(GgufQ8Route.Q1_0, GgmlType.Q1_0);
        assertSubroute(GgufQ8Route.TQ1_0, GgmlType.TQ1_0);
        assertSubroute(GgufQ8Route.TQ2_0, GgmlType.TQ2_0);
        assertSubroute(GgufQ8Route.MXFP4, GgmlType.MXFP4);
        assertSubroute(GgufQ8Route.NVFP4, GgmlType.NVFP4);
        assertSubroute(GgufQ8Route.Q8_0, GgmlType.Q8_0);
        assertSubroute(GgufQ8Route.Q8_1, GgmlType.Q8_1);
        assertSubroute(GgufQ8Route.Q8_K, GgmlType.Q8_K);
        assertSubroute(GgufQ8Route.IQ4_NL, GgmlType.IQ4_NL);
        assertSubroute(GgufQ8Route.IQ4_XS, GgmlType.IQ4_XS);

        assertEquals(0, GgufRawRoute.subrouteFromInfo(GgufRawRoute.routeInfoFor(GgmlType.Q4_K.id)));
        assertEquals(0, GgufRawRoute.subrouteFromInfo(GgufRawRoute.routeInfoFor(10_000)));
    }

    @Test
    void reportsOnlyRawFormatsThatConsumeEstimateHints() {
        assertTrue(GgufRawRoute.usesEstimateHints(GgmlType.Q2_K.id));
        assertTrue(GgufRawRoute.usesEstimateHints(GgmlType.Q4_K.id));
        assertTrue(GgufRawRoute.usesEstimateHints(GgmlType.Q5_K.id));
        assertTrue(GgufRawRoute.usesEstimateHints(GgmlType.Q4_1.id));
        assertTrue(GgufRawRoute.usesEstimateHints(GgmlType.Q5_1.id));

        assertFalse(GgufRawRoute.usesEstimateHints(GgmlType.Q3_K.id));
        assertFalse(GgufRawRoute.usesEstimateHints(GgmlType.Q6_K.id));
        assertFalse(GgufRawRoute.usesEstimateHints(GgmlType.Q8_0.id));
        assertFalse(GgufRawRoute.usesEstimateHints(GgmlType.F32.id));
        assertFalse(GgufRawRoute.usesEstimateHints(-1));
        assertFalse(GgufRawRoute.usesEstimateHints(10_000));
    }

    private static void assertRoute(int route, GgmlType... types) {
        for (GgmlType type : types) {
            assertEquals(route, GgufRawRoute.routeFor(type.id));
        }
    }

    private static void assertSubroute(int subroute, GgmlType type) {
        assertEquals(subroute, GgufRawRoute.subrouteFromInfo(GgufRawRoute.routeInfoFor(type.id)));
    }
}
