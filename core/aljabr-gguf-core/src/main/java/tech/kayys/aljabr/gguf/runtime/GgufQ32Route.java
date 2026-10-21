package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.core.GgmlType;

/**
 * Compact type-id router for Q32-family raw kernels.
 *
 * <p>Q4/Q5 legacy quant formats share the same prepared cache family but need
 * distinct raw row kernels. Keeping that subtype decision in a tiny table keeps
 * decode-time dispatch cheap and centralizes future Q32-family additions.</p>
 * @author bhangun
 */
final class GgufQ32Route {
    static final int UNSUPPORTED = 0;
    static final int Q4_0 = 1;
    static final int Q4_1 = 2;
    static final int Q5_0 = 3;
    static final int Q5_1 = 4;

    private static final byte[] ROUTES = routesByTypeId();

    private GgufQ32Route() {
    }

    static int forType(int typeId) {
        return typeId >= 0 && typeId < ROUTES.length ? ROUTES[typeId] : UNSUPPORTED;
    }

    static boolean hasBlockBiases(int route) {
        return route == Q4_1 || route == Q5_1;
    }

    private static byte[] routesByTypeId() {
        byte[] routes = new byte[maxTypeId() + 1];
        routes[GgmlType.Q4_0.id] = Q4_0;
        routes[GgmlType.Q4_1.id] = Q4_1;
        routes[GgmlType.Q5_0.id] = Q5_0;
        routes[GgmlType.Q5_1.id] = Q5_1;
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
