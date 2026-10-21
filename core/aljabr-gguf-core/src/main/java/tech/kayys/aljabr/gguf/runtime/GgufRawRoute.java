package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.MemorySegment;

/**
 * Raw GGUF mat-vec fallback routes used after prepared-cache admission misses.
 * @author bhangun
 */
final class GgufRawRoute {
    static final int ROUTE_FALLBACK = 0;
    static final int ROUTE_Q32 = 1;
    static final int ROUTE_Q2K = 2;
    static final int ROUTE_Q3K = 3;
    static final int ROUTE_Q4K = 4;
    static final int ROUTE_Q5K = 5;
    static final int ROUTE_Q6K = 6;
    static final int ROUTE_Q8 = 7;
    static final int ROUTE_DENSE = 8;
    private static final int ROUTE_MASK = 0xff;
    private static final int ESTIMATE_HINT = 1 << 8;
    private static final int SUBROUTE_SHIFT = 9;
    private static final int SUBROUTE_MASK = 0xff << SUBROUTE_SHIFT;

    private static final int[] RAW_ROUTE_INFO = rawRouteInfoByTypeId();

    private GgufRawRoute() {
    }

    static boolean usesEstimateHints(int typeId) {
        return routeInfoUsesEstimateHints(routeInfoFor(typeId));
    }

    static int routeFor(int typeId) {
        return routeFromInfo(routeInfoFor(typeId));
    }

    static int routeInfoFor(int typeId) {
        return typeId >= 0 && typeId < RAW_ROUTE_INFO.length ? RAW_ROUTE_INFO[typeId] : ROUTE_FALLBACK;
    }

    static int routeFromInfo(int routeInfo) {
        return routeInfo & ROUTE_MASK;
    }

    static boolean routeInfoUsesEstimateHints(int routeInfo) {
        return (routeInfo & ESTIMATE_HINT) != 0;
    }

    static int subrouteFromInfo(int routeInfo) {
        return (routeInfo & SUBROUTE_MASK) >>> SUBROUTE_SHIFT;
    }

    static void rows(
            GGUFModel model,
            GGUFTensorInfo tensor,
            GgufKey key,
            long matrixRows,
            int columns,
            long rowBytes,
            int route,
            int subroute,
            int hintBlocks,
            int typeId,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        MemorySegment data = GgufTensorData.tensorData(model, tensor);
        if (rowCount == 1) {
            output[0] = singleRow(
                    model, key, matrixRows, hintBlocks, data, rowBytes, columns, route, subroute, typeId, vector);
            return;
        }
        switch (route) {
            case ROUTE_Q32 -> q32(
                    model,
                    key,
                    matrixRows,
                    hintBlocks,
                    data,
                    rowBytes,
                    columns,
                    subroute,
                    typeId,
                    vector,
                    output,
                    rowCount,
                    parallel);
            case ROUTE_Q2K ->
                    q2K(model, key, matrixRows, hintBlocks, data, rowBytes, columns, vector, output, rowCount, parallel);
            case ROUTE_Q3K -> q3K(data, rowBytes, columns, vector, output, rowCount, parallel);
            case ROUTE_Q4K ->
                    q4K(model, key, matrixRows, hintBlocks, data, rowBytes, columns, vector, output, rowCount, parallel);
            case ROUTE_Q5K ->
                    q5K(model, key, matrixRows, hintBlocks, data, rowBytes, columns, vector, output, rowCount, parallel);
            case ROUTE_Q6K -> q6K(data, rowBytes, columns, vector, output, rowCount, parallel);
            case ROUTE_Q8 -> q8(data, rowBytes, columns, subroute, typeId, vector, output, rowCount, parallel);
            case ROUTE_DENSE ->
                    dense(data, rowBytes, columns, subroute, typeId, vector, output, rowCount, parallel);
            default -> GgufRawMatVec.fallback(data, typeId, columns, rowBytes, vector, output, rowCount, parallel);
        }
    }

    private static float singleRow(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int hintBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            int route,
            int subroute,
            int typeId,
            float[] vector) {
        return switch (route) {
            case ROUTE_Q32 -> q32Single(
                    model, key, matrixRows, hintBlocks, data, rowBytes, columns, subroute, typeId, vector);
            case ROUTE_Q2K ->
                    GgufRawDot.q2K(
                            data,
                            columns,
                            vector,
                            q2KHasMins(model, key, matrixRows, hintBlocks, data, rowBytes, columns, 1));
            case ROUTE_Q3K -> GgufRawDot.q3K(data, columns, vector);
            case ROUTE_Q4K ->
                    GgufRawDot.q4K(
                            data,
                            columns,
                            vector,
                            q4KHasMins(model, key, matrixRows, hintBlocks, data, rowBytes, columns, 1));
            case ROUTE_Q5K ->
                    GgufRawDot.q5K(
                            data,
                            columns,
                            vector,
                            q5KHasMins(model, key, matrixRows, hintBlocks, data, rowBytes, columns, 1));
            case ROUTE_Q6K -> GgufRawDot.q6K(data, columns, vector);
            case ROUTE_Q8 -> GgufRawDot.q8(data, subroute, typeId, columns, vector);
            case ROUTE_DENSE -> GgufRawDot.dense(data, subroute, typeId, columns, vector);
            default -> GgufRawDot.fallback(data, typeId, columns, vector);
        };
    }

    private static void q32(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int hintBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            int q32Route,
            int typeId,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        if (!GgufQ32Route.hasBlockBiases(q32Route)) {
            GgufRawMatVec.q32Raw(data, q32Route, typeId, columns, rowBytes, vector, output, rowCount, parallel, false);
            return;
        }
        boolean hasBlockBiases =
                q32HasBlockBias(model, key, matrixRows, hintBlocks, data, rowBytes, columns, rowCount, q32Route);
        GgufRawMatVec.q32Raw(
                data, q32Route, typeId, columns, rowBytes, vector, output, rowCount, parallel, hasBlockBiases);
    }

    private static float q32Single(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int hintBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            int q32Route,
            int typeId,
            float[] vector) {
        if (!GgufQ32Route.hasBlockBiases(q32Route)) {
            return GgufRawDot.q32(data, q32Route, typeId, columns, vector, false);
        }
        boolean hasBlockBiases =
                q32HasBlockBias(model, key, matrixRows, hintBlocks, data, rowBytes, columns, 1, q32Route);
        return GgufRawDot.q32(data, q32Route, typeId, columns, vector, hasBlockBiases);
    }

    private static boolean q32HasBlockBias(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int hintBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            int rowCount,
            int q32Route) {
        return GgufRawPathHints.q32HasBlockBias(
                model, key, matrixRows, hintBlocks, data, rowBytes, columns, rowCount, q32Route);
    }

    private static void q2K(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int hintBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        boolean hasMins = q2KHasMins(model, key, matrixRows, hintBlocks, data, rowBytes, columns, rowCount);
        GgufRawMatVec.q2K(
                data,
                columns,
                rowBytes,
                vector,
                output,
                rowCount,
                parallel,
                hasMins);
    }

    private static void q3K(
            MemorySegment data,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        GgufRawMatVec.q3K(data, columns, rowBytes, vector, output, rowCount, parallel);
    }

    private static void q4K(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int hintBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        boolean hasMins = q4KHasMins(model, key, matrixRows, hintBlocks, data, rowBytes, columns, rowCount);
        GgufRawMatVec.q4K(
                data,
                columns,
                rowBytes,
                vector,
                output,
                rowCount,
                parallel,
                hasMins);
    }

    private static void q5K(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int hintBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        boolean hasMins = q5KHasMins(model, key, matrixRows, hintBlocks, data, rowBytes, columns, rowCount);
        GgufRawMatVec.q5K(
                data,
                columns,
                rowBytes,
                vector,
                output,
                rowCount,
                parallel,
                hasMins);
    }

    private static boolean q2KHasMins(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int hintBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            int rowCount) {
        return GgufRawPathHints.q2KHasMins(model, key, matrixRows, hintBlocks, data, rowBytes, columns, rowCount);
    }

    private static boolean q4KHasMins(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int hintBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            int rowCount) {
        return GgufRawPathHints.q4KHasMins(model, key, matrixRows, hintBlocks, data, rowBytes, columns, rowCount);
    }

    private static boolean q5KHasMins(
            GGUFModel model,
            GgufKey key,
            long matrixRows,
            int hintBlocks,
            MemorySegment data,
            long rowBytes,
            int columns,
            int rowCount) {
        return GgufRawPathHints.q5KHasMins(model, key, matrixRows, hintBlocks, data, rowBytes, columns, rowCount);
    }

    private static void q6K(
            MemorySegment data,
            long rowBytes,
            int columns,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        GgufRawMatVec.q6K(data, columns, rowBytes, vector, output, rowCount, parallel);
    }

    private static void q8(
            MemorySegment data,
            long rowBytes,
            int columns,
            int q8Route,
            int typeId,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        GgufRawMatVec.q8Raw(data, q8Route, typeId, columns, rowBytes, vector, output, rowCount, parallel);
    }

    private static void dense(
            MemorySegment data,
            long rowBytes,
            int columns,
            int denseRoute,
            int typeId,
            float[] vector,
            float[] output,
            int rowCount,
            boolean parallel) {
        GgufRawMatVec.denseRaw(
                data,
                denseRoute,
                typeId,
                columns,
                rowBytes,
                vector,
                output,
                rowCount,
                parallel);
    }

    private static int[] rawRouteInfoByTypeId() {
        int[] info = newRouteInfoTable();
        markRoute(info, ROUTE_DENSE, GgufDenseRoute.F32, GgmlType.F32);
        markRoute(info, ROUTE_DENSE, GgufDenseRoute.F16, GgmlType.F16);
        markRoute(info, ROUTE_DENSE, GgufDenseRoute.BF16, GgmlType.BF16);
        markRoute(info, ROUTE_Q32, GgufQ32Route.Q4_0, GgmlType.Q4_0);
        markRoute(info, ROUTE_Q32, GgufQ32Route.Q4_1, GgmlType.Q4_1);
        markRoute(info, ROUTE_Q32, GgufQ32Route.Q5_0, GgmlType.Q5_0);
        markRoute(info, ROUTE_Q32, GgufQ32Route.Q5_1, GgmlType.Q5_1);
        markRoute(info, ROUTE_Q2K, GgmlType.Q2_K);
        markRoute(info, ROUTE_Q3K, GgmlType.Q3_K);
        markRoute(info, ROUTE_Q4K, GgmlType.Q4_K);
        markRoute(info, ROUTE_Q5K, GgmlType.Q5_K);
        markRoute(info, ROUTE_Q6K, GgmlType.Q6_K);
        markRoute(info, ROUTE_Q8, GgufQ8Route.Q1_0, GgmlType.Q1_0);
        markRoute(info, ROUTE_Q8, GgufQ8Route.TQ1_0, GgmlType.TQ1_0);
        markRoute(info, ROUTE_Q8, GgufQ8Route.TQ2_0, GgmlType.TQ2_0);
        markRoute(info, ROUTE_Q8, GgufQ8Route.MXFP4, GgmlType.MXFP4);
        markRoute(info, ROUTE_Q8, GgufQ8Route.NVFP4, GgmlType.NVFP4);
        markRoute(info, ROUTE_Q8, GgufQ8Route.Q8_0, GgmlType.Q8_0);
        markRoute(info, ROUTE_Q8, GgufQ8Route.Q8_1, GgmlType.Q8_1);
        markRoute(info, ROUTE_Q8, GgufQ8Route.Q8_K, GgmlType.Q8_K);
        markRoute(info, ROUTE_Q8, GgufQ8Route.IQ4_NL, GgmlType.IQ4_NL);
        markRoute(info, ROUTE_Q8, GgufQ8Route.IQ4_XS, GgmlType.IQ4_XS);
        markEstimateHint(info, GgmlType.Q2_K, GgmlType.Q4_K, GgmlType.Q5_K, GgmlType.Q4_1, GgmlType.Q5_1);
        return info;
    }

    private static int[] newRouteInfoTable() {
        int maxId = 0;
        for (GgmlType type : GgmlType.values()) {
            maxId = Math.max(maxId, type.id);
        }
        return new int[maxId + 1];
    }

    private static void markRoute(int[] info, int route, GgmlType... types) {
        for (GgmlType type : types) {
            info[type.id] = (info[type.id] & ~ROUTE_MASK) | route;
        }
    }

    private static void markRoute(int[] info, int route, int subroute, GgmlType type) {
        info[type.id] = route | (subroute << SUBROUTE_SHIFT);
    }

    private static void markEstimateHint(int[] info, GgmlType... types) {
        for (GgmlType type : types) {
            info[type.id] |= ESTIMATE_HINT;
        }
    }
}
