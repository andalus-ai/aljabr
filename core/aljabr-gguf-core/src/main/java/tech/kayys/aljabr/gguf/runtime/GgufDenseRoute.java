package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.core.GgmlType;

/**
 * Compact type-id router for dense raw kernels.
 *
 * <p>Dense tensors only have three supported scalar formats, but this path is
 * used for both direct row dots and fallback matvecs. Routing once lets callers
 * jump straight to the F32, F16, or BF16 implementation.</p>
 * @author bhangun
 */
final class GgufDenseRoute {
    static final int UNSUPPORTED = 0;
    static final int F32 = 1;
    static final int F16 = 2;
    static final int BF16 = 3;

    private static final byte[] ROUTES = routesByTypeId();

    private GgufDenseRoute() {
    }

    static int forType(int typeId) {
        return typeId >= 0 && typeId < ROUTES.length ? ROUTES[typeId] : UNSUPPORTED;
    }

    private static byte[] routesByTypeId() {
        byte[] routes = new byte[maxTypeId() + 1];
        routes[GgmlType.F32.id] = F32;
        routes[GgmlType.F16.id] = F16;
        routes[GgmlType.BF16.id] = BF16;
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
