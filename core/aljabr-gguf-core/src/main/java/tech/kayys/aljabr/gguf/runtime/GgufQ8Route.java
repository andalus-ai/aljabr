package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.core.GgmlType;

/**
 * Compact type-id router for Q8-family raw kernels.
 *
 * <p>The outer raw route only knows that a tensor belongs to the broad Q8
 * prepared-cache family. This table keeps the hotter inner subtype dispatch to
 * one array load for both single-row dots and multi-row matvecs.</p>
 * @author bhangun
 */
final class GgufQ8Route {
    static final int UNSUPPORTED = 0;
    static final int Q1_0 = 1;
    static final int TQ1_0 = 2;
    static final int TQ2_0 = 3;
    static final int MXFP4 = 4;
    static final int NVFP4 = 5;
    static final int Q8_0 = 6;
    static final int Q8_1 = 7;
    static final int Q8_K = 8;
    static final int IQ4_NL = 9;
    static final int IQ4_XS = 10;

    private static final byte[] ROUTES = routesByTypeId();

    private GgufQ8Route() {
    }

    static int forType(int typeId) {
        return typeId >= 0 && typeId < ROUTES.length ? ROUTES[typeId] : UNSUPPORTED;
    }

    private static byte[] routesByTypeId() {
        byte[] routes = new byte[maxTypeId() + 1];
        routes[GgmlType.Q1_0.id] = Q1_0;
        routes[GgmlType.TQ1_0.id] = TQ1_0;
        routes[GgmlType.TQ2_0.id] = TQ2_0;
        routes[GgmlType.MXFP4.id] = MXFP4;
        routes[GgmlType.NVFP4.id] = NVFP4;
        routes[GgmlType.Q8_0.id] = Q8_0;
        routes[GgmlType.Q8_1.id] = Q8_1;
        routes[GgmlType.Q8_K.id] = Q8_K;
        routes[GgmlType.IQ4_NL.id] = IQ4_NL;
        routes[GgmlType.IQ4_XS.id] = IQ4_XS;
        return routes;
    }

    private static int maxTypeId() {
        int maxId = 0;
        for (GgmlType type : GgmlType.values()) {
            maxId = Math.max(maxId, type.id);
        }
        return maxId;
    }
}
