package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFParser;
import tech.kayys.aljabr.gguf.loader.GGUFReader;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.PreparedMatrix;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

/**
 * Executes small Java-native GGUF tensor primitives against a model.
 *
 * <p>The probe measures row-dot and prepared mat-vec paths, giving the CLI and
 * tests a cheap signal for whether the Java runtime is ready for a model.</p>
 * @author bhangun
 */
public record GgufRuntimeProbe(
        GgufRuntimeProfile profile,
        String tensorName,
        String tensorType,
        long rows,
        long columns,
        int sampledRows,
        long rowDotNanos,
        float rowDotChecksum,
        int matVecRows,
        boolean preparedMatVecProbe,
        long matrixCacheNanos,
        long matVecNanos,
        float matVecChecksum,
        long cachedMatVecNanos,
        float cachedMatVecChecksum,
        PreparedMatrixCacheDecision preparedMatrixCacheDecision
) {
    private static final List<String> PREFERRED_PROBE_TENSORS = List.of(
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "token_embd.weight"
    );

    public static GgufRuntimeProbe load(Path modelPath, int requestedRows) throws IOException {
        return load(modelPath, requestedRows, requestedRows);
    }

    public static GgufRuntimeProbe load(
            Path modelPath,
            int requestedDotRows,
            int requestedMatVecRows) throws IOException {
        long startNanos = System.nanoTime();
        try (Arena arena = Arena.ofShared(); GGUFReader reader = new GGUFReader(modelPath, arena)) {
            GGUFModel model = new GGUFParser().parse(reader.segment(), null);
            long loadMillis = Duration.ofNanos(System.nanoTime() - startNanos).toMillis();
            return fromModel(
                    model,
                    Files.size(modelPath),
                    loadMillis,
                    requestedDotRows,
                    requestedMatVecRows,
                    selectProbeDecoderPreparedMatrixCache(model));
        }
    }

    public static GgufRuntimeProbe load(
            Path modelPath,
            int requestedDotRows,
            int requestedMatVecRows,
            int prepareMinRows) throws IOException {
        long startNanos = System.nanoTime();
        try (Arena arena = Arena.ofShared(); GGUFReader reader = new GGUFReader(modelPath, arena)) {
            GGUFModel model = new GGUFParser().parse(reader.segment(), null);
            long loadMillis = Duration.ofNanos(System.nanoTime() - startNanos).toMillis();
            return fromModel(
                    model,
                    Files.size(modelPath),
                    loadMillis,
                    requestedDotRows,
                    requestedMatVecRows,
                    selectDecoderPreparedMatrixCache(model, prepareMinRows, false, 1, 0L));
        }
    }

    public static GgufRuntimeProbe fromModel(
            GGUFModel model,
            long modelBytes,
            long loadMillis,
            int requestedRows) {
        return fromModel(model, modelBytes, loadMillis, requestedRows, requestedRows);
    }

    public static GgufRuntimeProbe fromModel(
            GGUFModel model,
            long modelBytes,
            long loadMillis,
            int requestedDotRows,
            int requestedMatVecRows) {
        return fromModel(model, modelBytes, loadMillis, requestedDotRows, requestedMatVecRows, 0);
    }

    public static GgufRuntimeProbe fromModel(
            GGUFModel model,
            long modelBytes,
            long loadMillis,
            int requestedDotRows,
            int requestedMatVecRows,
            int prepareMinRows) {
        return fromModel(
                model,
                modelBytes,
                loadMillis,
                requestedDotRows,
                requestedMatVecRows,
                selectDecoderPreparedMatrixCache(model, prepareMinRows, false, 1, 0L));
    }

    public static GgufRuntimeProbe fromModel(
            GGUFModel model,
            long modelBytes,
            long loadMillis,
            int requestedDotRows,
            int requestedMatVecRows,
            PreparedMatrixCacheSelection cacheSelection) {
        GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model, modelBytes, loadMillis);
        Optional<GGUFTensorInfo> tensor = selectProbeTensor(model);
        if (tensor.isEmpty()) {
            PreparedMatrixCacheDecision decision = prepareDecoderMatrixCaches(model, cacheSelection);
            return new GgufRuntimeProbe(
                    profile,
                    "",
                    "",
                    0,
                    0,
                    0,
                    0,
                    0.0f,
                    0,
                    false,
                    0,
                    0,
                    0.0f,
                    0,
                    0.0f,
                    decision);
        }
        GgufRuntimeProbe probe = probeTensor(model, tensor.get(), requestedDotRows, requestedMatVecRows, profile);
        return probe.withPreparedMatrixCaches(prepareDecoderMatrixCaches(model, cacheSelection));
    }

    public boolean hasTensorProbe() {
        return tensorName != null && !tensorName.isBlank();
    }

    public double rowDotMillis() {
        return rowDotNanos / 1_000_000.0d;
    }

    public double matVecMillis() {
        return matVecNanos / 1_000_000.0d;
    }

    public double matrixCacheMillis() {
        return matrixCacheNanos / 1_000_000.0d;
    }

    public double cachedMatVecMillis() {
        return cachedMatVecNanos / 1_000_000.0d;
    }

    public boolean preparedMatVecReady() {
        return preparedMatVecProbe && matVecChecksumsAgree();
    }

    public boolean hasPreparedMatrixCacheProbe() {
        return preparedMatrixCacheStats() != null && preparedMatrixCacheStats().scannedTensors() > 0;
    }

    public boolean hasPreparedMatrixCachePlan() {
        return preparedMatrixCachePlan() != null && preparedMatrixCachePlan().scannedTensors() > 0;
    }

    public GgufTensorOps.PreparedMatrixCachePlan preparedMatrixCachePlan() {
        return preparedMatrixCacheDecision == null
                ? GgufTensorOps.PreparedMatrixCachePlan.empty()
                : preparedMatrixCacheDecision.plan();
    }

    public GgufTensorOps.PreparedMatrixCacheStats preparedMatrixCacheStats() {
        return preparedMatrixCacheDecision == null
                ? GgufTensorOps.PreparedMatrixCacheStats.empty()
                : preparedMatrixCacheDecision.stats();
    }

    public boolean matVecChecksumsAgree() {
        float tolerance = Math.max(1.0e-4f, Math.abs(matVecChecksum) * 1.0e-4f);
        return Float.isFinite(matVecChecksum)
                && Float.isFinite(cachedMatVecChecksum)
                && Math.abs(matVecChecksum - cachedMatVecChecksum) <= tolerance;
    }

    public String compactSummary() {
        if (!hasTensorProbe()) {
            return "unavailable, no supported matrix tensor";
        }
        return String.format(
                Locale.ROOT,
                "tensor=%s, type=%s, rows=%d, cols=%d, sampledRows=%d, "
                        + "dot=%.3fms, dotChecksum=%.6g, matVecRows=%d, cache=%.3fms, "
                        + "preparedMatVecReady=%s, parallelMatVec=%.3fms, matVecChecksum=%.6g, "
                        + "cachedGenericMatVec=%.3fms, cachedChecksum=%.6g",
                tensorName,
                tensorType,
                rows,
                columns,
                sampledRows,
                rowDotMillis(),
                rowDotChecksum,
                matVecRows,
                matrixCacheMillis(),
                preparedMatVecReady(),
                matVecMillis(),
                matVecChecksum,
                cachedMatVecMillis(),
                cachedMatVecChecksum);
    }

    public static GgufTensorOps.PreparedMatrixCacheStats prepareDecoderMatrixCaches(
            GGUFModel model,
            int minRows) {
        if (minRows <= 0) {
            return GgufTensorOps.PreparedMatrixCacheStats.empty();
        }
        GgufTensorOps.clearPreparedMatrixCaches(model);
        return GgufTensorOps.prepareMatrixCaches(
                model,
                decoderMatVecWeights(model),
                minRows);
    }

    public static PreparedMatrixCacheDecision prepareDecoderMatrixCaches(
            GGUFModel model,
            PreparedMatrixCacheSelection selection) {
        if (selection == null || !selection.prepare()) {
            return selection == null
                    ? PreparedMatrixCacheDecision.skipped(
                            "disabled",
                            1,
                            0L,
                            GgufTensorOps.PreparedMatrixCachePlan.empty())
                    : selection.toDecision(GgufTensorOps.PreparedMatrixCacheStats.empty());
        }
        GgufTensorOps.clearPreparedMatrixCaches(model);
        Iterable<GGUFTensorInfo> tensors = selection.selectedTensors().isEmpty()
                ? decoderMatVecWeights(model)
                : selection.selectedTensors();
        GgufTensorOps.PreparedMatrixCacheStats stats = GgufTensorOps.prepareMatrixCaches(
                model,
                tensors,
                Math.max(1, selection.minRows()));
        return selection.toDecision(stats);
    }

    public static GgufTensorOps.PreparedMatrixCachePlan planDecoderMatrixCaches(
            GGUFModel model,
            int minRows) {
        return GgufTensorOps.planPreparedMatrixCaches(
                model,
                decoderMatVecWeights(model),
                Math.max(1, minRows));
    }

    public static PreparedMatrixCacheSelection selectDecoderPreparedMatrixCache(
            GGUFModel model,
            int explicitMinRows,
            boolean autoPrepare,
            int autoMinRows,
            long budgetBytes) {
        int explicitRows = Math.max(0, explicitMinRows);
        if (explicitRows > 0) {
            return PreparedMatrixCacheSelection.prepared(
                    "explicit",
                    explicitRows,
                    0L,
                    planDecoderMatrixCaches(model, explicitRows));
        }

        int rows = Math.max(1, autoMinRows);
        long budget = Math.max(0L, budgetBytes);
        GgufTensorOps.PreparedMatrixCachePlan plan = planDecoderMatrixCaches(model, rows);
        if (!autoPrepare) {
            return PreparedMatrixCacheSelection.skipped("disabled", rows, budget, plan);
        }
        if (plan.failedTensors() > 0) {
            return PreparedMatrixCacheSelection.skipped("plan-failed", rows, budget, plan);
        }
        if (!plan.hasCandidates() && plan.skippedCacheTooSmallTensors() > 0) {
            return PreparedMatrixCacheSelection.skipped("cache-too-small", rows, budget, plan);
        }
        if (!plan.hasCandidates()) {
            return PreparedMatrixCacheSelection.skipped("no-candidates", rows, budget, plan);
        }
        if (budget > 0 && plan.estimatedPreparedBytes() > budget) {
            BudgetedMatrixCacheSelection budgeted = planBudgetedDecoderMatrixCaches(model, rows, budget);
            if (budgeted.plan().hasCandidates() && budgeted.plan().failedTensors() == 0) {
                return PreparedMatrixCacheSelection.prepared(
                        "auto-budget",
                        rows,
                        budget,
                        budgeted.plan(),
                        budgeted.tensors());
            }
            return PreparedMatrixCacheSelection.skipped("budget", rows, budget, plan);
        }
        return PreparedMatrixCacheSelection.prepared("auto", rows, budget, plan);
    }

    private static Iterable<GGUFTensorInfo> decoderMatVecWeights(GGUFModel model) {
        return () -> new Iterator<>() {
            private final Iterator<GGUFTensorInfo> tensors = model.tensors().iterator();
            private GGUFTensorInfo next;
            private boolean nextReady;

            @Override
            public boolean hasNext() {
                if (nextReady) {
                    return true;
                }
                while (tensors.hasNext()) {
                    GGUFTensorInfo candidate = tensors.next();
                    if (isDecoderMatVecWeight(candidate)) {
                        next = candidate;
                        nextReady = true;
                        return true;
                    }
                }
                return false;
            }

            @Override
            public GGUFTensorInfo next() {
                if (!hasNext()) {
                    throw new java.util.NoSuchElementException();
                }
                GGUFTensorInfo result = next;
                next = null;
                nextReady = false;
                return result;
            }
        };
    }

    private static BudgetedMatrixCacheSelection planBudgetedDecoderMatrixCaches(
            GGUFModel model,
            int minRows,
            long budgetBytes) {
        List<GGUFTensorInfo> selected = selectBudgetedDecoderMatrixCacheTensors(model, minRows, budgetBytes);
        if (selected.isEmpty()) {
            return new BudgetedMatrixCacheSelection(List.of(), GgufTensorOps.PreparedMatrixCachePlan.empty());
        }
        return new BudgetedMatrixCacheSelection(
                selected,
                GgufTensorOps.planPreparedMatrixCaches(model, selected, Math.max(1, minRows)));
    }

    private static List<GGUFTensorInfo> selectBudgetedDecoderMatrixCacheTensors(
            GGUFModel model,
            int minRows,
            long budgetBytes) {
        int rowFloor = Math.max(1, minRows);
        long budget = Math.max(0L, budgetBytes);
        List<GGUFTensorInfo> selected = new ArrayList<>();
        long selectedBytes = 0L;
        long[] reservedCacheBytes = new long[GgufTensorOps.preparedMatrixCacheBucketCount()];

        for (GGUFTensorInfo tensor : decoderMatVecWeights(model)) {
            try {
                if (tensor.shape().length < 2) {
                    continue;
                }
                long[] trialReservedCacheBytes = reservedCacheBytes.clone();
                GgufScanCandidate candidate =
                        GgufPreparationPlan.admitPreparedMatrixCandidate(model, tensor, rowFloor, trialReservedCacheBytes);
                if (candidate.status() != GgufScanStatus.READY) {
                    continue;
                }
                long estimatedBytes = candidate.estimatedBytes();
                if (budget > 0L && selectedBytes > budget - estimatedBytes) {
                    if (selectedBytes >= budget) {
                        break;
                    }
                    continue;
                }
                System.arraycopy(
                        trialReservedCacheBytes,
                        0,
                        reservedCacheBytes,
                        0,
                        reservedCacheBytes.length);
                selectedBytes += estimatedBytes;
                selected.add(tensor);
            } catch (RuntimeException ignored) {
                // Keep budget selection best-effort; invalid tensors are skipped like non-candidates.
            }
        }
        return List.copyOf(selected);
    }

    private record BudgetedMatrixCacheSelection(
            List<GGUFTensorInfo> tensors,
            GgufTensorOps.PreparedMatrixCachePlan plan) {
    }

    private static PreparedMatrixCacheSelection selectProbeDecoderPreparedMatrixCache(GGUFModel model) {
        int explicitMinRows = Math.max(0, Integer.getInteger("aljabr.gguf.java_probe_prepare_min_rows", 0));
        if (explicitMinRows > 0) {
            return selectDecoderPreparedMatrixCache(model, explicitMinRows, false, 1, 0L);
        }
        boolean autoPrepare = Boolean.parseBoolean(System.getProperty("aljabr.gguf.java_probe_auto_prepare", "true"));
        int autoMinRows = Math.max(1, Integer.getInteger("aljabr.gguf.java_probe_auto_prepare_min_rows", 32));
        long budgetBytes = GgufBudget.byteSizeProperty(
                "aljabr.gguf.java_probe_auto_prepare_budget_bytes",
                GgufBudget.defaultAutoPrepareBytes());
        return selectDecoderPreparedMatrixCache(model, 0, autoPrepare, autoMinRows, budgetBytes);
    }

    public static boolean isDecoderMatVecWeight(GGUFTensorInfo tensor) {
        String name = tensor.name();
        return name.equals("output.weight")
                || (name.startsWith("blk.")
                        && (name.endsWith("attn_q.weight")
                                || name.endsWith("attn_k.weight")
                                || name.endsWith("attn_v.weight")
                                || name.endsWith("attn_output.weight")
                                || name.endsWith("ffn_gate.weight")
                                || name.endsWith("ffn_up.weight")
                                || name.endsWith("ffn_down.weight")));
    }

    private static Optional<GGUFTensorInfo> selectProbeTensor(GGUFModel model) {
        for (String name : PREFERRED_PROBE_TENSORS) {
            for (GGUFTensorInfo candidate : model.tensors()) {
                if (name.equals(candidate.name()) && canProbe(candidate)) {
                    return Optional.of(candidate);
                }
            }
        }
        GGUFTensorInfo largest = null;
        for (GGUFTensorInfo candidate : model.tensors()) {
            if (canProbe(candidate)
                    && (largest == null || candidate.sizeInBytes() > largest.sizeInBytes())) {
                largest = candidate;
            }
        }
        return Optional.ofNullable(largest);
    }

    private static boolean canProbe(GGUFTensorInfo tensor) {
        return tensor.shape().length >= 2
                && GgufTensorOps.supportsRowDotType(tensor.typeId())
                && GgufTensorOps.matrixColumns(tensor) <= Integer.MAX_VALUE;
    }

    private static GgufRuntimeProbe probeTensor(
            GGUFModel model,
            GGUFTensorInfo tensor,
            int requestedDotRows,
            int requestedMatVecRows,
            GgufRuntimeProfile profile) {
        long rows = GgufTensorOps.matrixRows(tensor);
        long columns = GgufTensorOps.matrixColumns(tensor);
        int sampledRows = (int) Math.max(1, Math.min(rows, Math.max(1, requestedDotRows)));
        int matVecRows = (int) Math.max(1, Math.min(rows, Math.max(1, requestedMatVecRows)));
        float[] vector = deterministicProbeVector((int) columns);

        float rowDotChecksum = 0.0f;
        long startNanos = System.nanoTime();
        for (int row = 0; row < sampledRows; row++) {
            rowDotChecksum += GgufTensorOps.dotRow(model, tensor, row, vector);
        }
        long rowDotNanos = System.nanoTime() - startNanos;

        float[] output = new float[matVecRows];
        long matrixCacheNanos = 0L;
        long matVecNanos;
        boolean preparedMatVecProbe = false;
        long cachedMatVecNanos = 0L;
        float cachedMatVecChecksum = 0.0f;
        PreparedMatVecResult preparedMatVec = preparedMatVecProbe(model, tensor, vector, matVecRows);
        if (preparedMatVec != null) {
            preparedMatVecProbe = true;
            matrixCacheNanos = preparedMatVec.matrixCacheNanos();
            matVecNanos = preparedMatVec.matVecNanos();
            output = preparedMatVec.output();
            cachedMatVecNanos = preparedMatVec.cachedMatVecNanos();
            cachedMatVecChecksum = preparedMatVec.cachedMatVecChecksum();
        } else {
            startNanos = System.nanoTime();
            GgufTensorOps.matVecRows(model, tensor, vector, output, matVecRows, true);
            matVecNanos = System.nanoTime() - startNanos;
        }
        float matVecChecksum = checksum(output);

        return new GgufRuntimeProbe(
                profile,
                tensor.name(),
                typeLabel(tensor.typeId()),
                rows,
                columns,
                sampledRows,
                rowDotNanos,
                rowDotChecksum,
                matVecRows,
                preparedMatVecProbe,
                matrixCacheNanos,
                matVecNanos,
                matVecChecksum,
                cachedMatVecNanos,
                cachedMatVecChecksum,
                PreparedMatrixCacheDecision.skipped(
                        "disabled",
                        1,
                        0L,
                        GgufTensorOps.PreparedMatrixCachePlan.empty()));
    }

    private GgufRuntimeProbe withPreparedMatrixCaches(PreparedMatrixCacheDecision decision) {
        return new GgufRuntimeProbe(
                profile,
                tensorName,
                tensorType,
                rows,
                columns,
                sampledRows,
                rowDotNanos,
                rowDotChecksum,
                matVecRows,
                preparedMatVecProbe,
                matrixCacheNanos,
                matVecNanos,
                matVecChecksum,
                cachedMatVecNanos,
                cachedMatVecChecksum,
                decision);
    }

    private static PreparedMatVecResult preparedMatVecProbe(
            GGUFModel model,
            GGUFTensorInfo tensor,
            float[] vector,
            int matVecRows) {
        GgufPreparedCachePolicy.Family family = GgufPreparedCachePolicy.preparedMatrixCacheFamily(tensor.typeId());
        return family == null
                ? null
                : runPreparedMatVecProbe(model, tensor, vector, matVecRows, family);
    }

    private static PreparedMatVecResult runPreparedMatVecProbe(
            GGUFModel model,
            GGUFTensorInfo tensor,
            float[] vector,
            int matVecRows,
            GgufPreparedCachePolicy.Family family) {
        GgufPreparedMatrixStore.clearMatrixCache(model, family);
        long cacheStartNanos = System.nanoTime();
        PreparedMatrix matrix = GgufPreparedMatrixStore.matrixCached(model, tensor, family);
        long matrixCacheNanos = System.nanoTime() - cacheStartNanos;

        float[] output = new float[matVecRows];
        long startNanos = System.nanoTime();
        GgufPrepRows.rowsTrusted(matrix, family, vector.length, vector, output, matVecRows, true);
        long matVecNanos = System.nanoTime() - startNanos;

        float[] cachedOutput = new float[matVecRows];
        long cachedStartNanos = System.nanoTime();
        GgufTensorOps.matVecRows(model, tensor, vector, cachedOutput, matVecRows, true);
        long cachedMatVecNanos = System.nanoTime() - cachedStartNanos;

        return new PreparedMatVecResult(
                matrixCacheNanos,
                matVecNanos,
                output,
                cachedMatVecNanos,
                checksum(cachedOutput));
    }

    private record PreparedMatVecResult(
            long matrixCacheNanos,
            long matVecNanos,
            float[] output,
            long cachedMatVecNanos,
            float cachedMatVecChecksum) {
    }

    private static float[] deterministicProbeVector(int columns) {
        float[] vector = new float[columns];
        for (int i = 0; i < vector.length; i++) {
            vector[i] = ((i % 17) - 8) / 17.0f;
        }
        return vector;
    }

    private static String typeLabel(int typeId) {
        try {
            return GgmlType.fromId(typeId).label;
        } catch (IllegalArgumentException ignored) {
            return "TYPE_" + typeId;
        }
    }

    private static float checksum(float[] values) {
        float total = 0.0f;
        for (float value : values) {
            total += value;
        }
        return total;
    }

    public record PreparedMatrixCacheSelection(
            String mode,
            int minRows,
            long budgetBytes,
            GgufTensorOps.PreparedMatrixCachePlan plan,
            List<GGUFTensorInfo> selectedTensors,
            boolean prepare) {
        static PreparedMatrixCacheSelection prepared(
                String mode,
                int minRows,
                long budgetBytes,
                GgufTensorOps.PreparedMatrixCachePlan plan) {
            return prepared(mode, minRows, budgetBytes, plan, List.of());
        }

        static PreparedMatrixCacheSelection prepared(
                String mode,
                int minRows,
                long budgetBytes,
                GgufTensorOps.PreparedMatrixCachePlan plan,
                List<GGUFTensorInfo> selectedTensors) {
            return new PreparedMatrixCacheSelection(
                    mode + "-prepared",
                    minRows,
                    budgetBytes,
                    plan,
                    List.copyOf(selectedTensors),
                    true);
        }

        static PreparedMatrixCacheSelection skipped(
                String reason,
                int minRows,
                long budgetBytes,
                GgufTensorOps.PreparedMatrixCachePlan plan) {
            return new PreparedMatrixCacheSelection(
                    "auto-skipped-" + reason,
                    minRows,
                    budgetBytes,
                    plan,
                    List.of(),
                    false);
        }

        PreparedMatrixCacheDecision toDecision(GgufTensorOps.PreparedMatrixCacheStats stats) {
            return new PreparedMatrixCacheDecision(mode, minRows, budgetBytes, plan, stats);
        }
    }

    public record PreparedMatrixCacheDecision(
            String mode,
            int minRows,
            long budgetBytes,
            GgufTensorOps.PreparedMatrixCachePlan plan,
            GgufTensorOps.PreparedMatrixCacheStats stats) {
        static PreparedMatrixCacheDecision skipped(
                String reason,
                int minRows,
                long budgetBytes,
                GgufTensorOps.PreparedMatrixCachePlan plan) {
            return new PreparedMatrixCacheDecision(
                    "auto-skipped-" + reason,
                    minRows,
                    budgetBytes,
                    plan,
                    GgufTensorOps.PreparedMatrixCacheStats.empty());
        }

        public String selectionSummary() {
            return String.format(
                    Locale.ROOT,
                    "mode=%s, minRows=%d, budget=%s",
                    mode,
                    minRows,
                    formatBytes(budgetBytes));
        }

        public String compactSummary() {
            return String.format(
                    Locale.ROOT,
                    "%s, plan={%s}, stats={%s}",
                    selectionSummary(),
                    plan.compactSummary(),
                    stats.compactSummary());
        }
    }

    private static String formatBytes(long bytes) {
        if (bytes <= 0) {
            return "unbounded";
        }
        return String.format(Locale.ROOT, "%.2fMiB", bytes / 1024.0d / 1024.0d);
    }
}
