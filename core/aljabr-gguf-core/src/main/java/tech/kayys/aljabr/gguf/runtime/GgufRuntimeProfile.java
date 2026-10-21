package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFParser;
import tech.kayys.aljabr.gguf.loader.GGUFReader;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.mapper.GgufMetadataMapper;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Buildable Java-native GGUF readiness profile.
 *
 * <p>This is intentionally stricter than the llama.cpp fallback: Java can parse
 * and inspect many models today, but generation should only be enabled after we
 * can prove the tensor layout, quant families, and decoder tensor set are
 * complete.</p>
 * @author bhangun
 */
public record GgufRuntimeProfile(
        String architecture,
        int ggufVersion,
        int metadataCount,
        int tensorCount,
        long modelBytes,
        long tensorBytes,
        long knownTypeTensorBytes,
        int requiredDecoderTensorCount,
        int presentDecoderTensorCount,
        int malformedDecoderTensorCount,
        List<String> missingDecoderTensorExamples,
        List<String> malformedDecoderTensorExamples,
        List<TensorTypeSummary> tensorTypes,
        List<String> unknownTensorTypeIds,
        List<String> unsupportedRowDotTypeIds,
        ModelConfig modelConfig,
        long loadMillis
) {
    private static final Set<Integer> JAVA_KNOWN_LAYOUT_TYPE_IDS = Set.of(
            GgmlType.F32.id,
            GgmlType.F16.id,
            GgmlType.BF16.id,
            GgmlType.Q4_0.id,
            GgmlType.Q4_1.id,
            GgmlType.Q5_0.id,
            GgmlType.Q5_1.id,
            GgmlType.Q8_0.id,
            GgmlType.Q8_1.id,
            GgmlType.Q2_K.id,
            GgmlType.Q3_K.id,
            GgmlType.Q4_K.id,
            GgmlType.Q5_K.id,
            GgmlType.Q6_K.id,
            GgmlType.Q8_K.id,
            GgmlType.IQ4_NL.id,
            GgmlType.IQ4_XS.id,
            GgmlType.TQ1_0.id,
            GgmlType.TQ2_0.id,
            GgmlType.MXFP4.id,
            GgmlType.NVFP4.id,
            GgmlType.Q1_0.id
    );

    public static GgufRuntimeProfile load(Path modelPath) throws IOException {
        long startNanos = System.nanoTime();
        try (Arena arena = Arena.ofConfined(); GGUFReader reader = new GGUFReader(modelPath, arena)) {
            GGUFModel model = new GGUFParser().parse(reader.segment(), null);
            long loadMillis = Duration.ofNanos(System.nanoTime() - startNanos).toMillis();
            return fromModel(model, Files.size(modelPath), loadMillis);
        }
    }

    public static GgufRuntimeProfile fromModel(GGUFModel model, long modelBytes, long loadMillis) {
        Map<String, Object> metadata = model.metadata();
        List<GGUFTensorInfo> tensors = model.tensors();
        String architecture = String.valueOf(metadata.getOrDefault("general.architecture", "unknown"))
                .toLowerCase(Locale.ROOT);

        Map<String, TypeAccumulator> byType = new HashMap<>();
        List<String> unknownTypeIds = new ArrayList<>();
        List<String> unsupportedRowDotTypeIds = new ArrayList<>();
        long tensorBytes = 0;
        long knownTypeBytes = 0;

        for (GGUFTensorInfo tensor : tensors) {
            long bytes = Math.max(0, tensor.sizeInBytes());
            tensorBytes += bytes;

            String label = "TYPE_" + tensor.typeId();
            boolean knownForJava = JAVA_KNOWN_LAYOUT_TYPE_IDS.contains(tensor.typeId());
            boolean rowDotSupported = GgufTensorOps.supportsRowDotType(tensor.typeId());
            try {
                label = GgmlType.fromId(tensor.typeId()).label;
            } catch (IllegalArgumentException ignored) {
                unknownTypeIds.add(String.valueOf(tensor.typeId()));
                knownForJava = false;
                rowDotSupported = false;
            }

            byType.computeIfAbsent(label, ignored -> new TypeAccumulator())
                    .add(tensor, bytes, knownForJava);
            if (knownForJava) {
                knownTypeBytes += bytes;
            }
            if (!rowDotSupported) {
                unsupportedRowDotTypeIds.add(String.valueOf(tensor.typeId()));
            }
        }

        List<TensorTypeSummary> summaries = byType.entrySet().stream()
                .map(entry -> entry.getValue().summary(entry.getKey()))
                .sorted(Comparator.comparingLong(TensorTypeSummary::bytes).reversed())
                .toList();

        Map<String, GGUFTensorInfo> tensorsByName = new HashMap<>();
        for (GGUFTensorInfo tensor : tensors) {
            tensorsByName.put(tensor.name(), tensor);
        }

        List<String> required = requiredDecoderTensorNames(metadata, architecture);
        DecoderHiddenSizeHint shapeHints = DecoderHiddenSizeHint.from(metadata, architecture);
        int presentRequired = 0;
        int malformedRequired = 0;
        List<String> missingExamples = new ArrayList<>();
        List<String> malformedExamples = new ArrayList<>();
        for (String name : required) {
            GGUFTensorInfo tensor = tensorsByName.get(name);
            if (tensor == null) {
                if (missingExamples.size() < 8) {
                    missingExamples.add(name);
                }
            } else if (decoderTensorShapeValid(name, tensor, shapeHints)) {
                presentRequired++;
            } else {
                malformedRequired++;
                if (malformedExamples.size() < 8) {
                    malformedExamples.add(name + " shape=" + Arrays.toString(tensor.shape()));
                }
            }
        }

        return new GgufRuntimeProfile(
                architecture,
                model.version(),
                metadata.size(),
                tensors.size(),
                modelBytes,
                tensorBytes,
                knownTypeBytes,
                required.size(),
                presentRequired,
                malformedRequired,
                List.copyOf(missingExamples),
                List.copyOf(malformedExamples),
                List.copyOf(summaries),
                unknownTypeIds.stream().distinct().sorted().toList(),
                unsupportedRowDotTypeIds.stream().distinct().sorted().toList(),
                new GgufMetadataMapper().fromGgufMetadata(metadata),
                loadMillis);
    }

    public double knownTensorTypeRatio() {
        if (tensorBytes <= 0) {
            return 1.0d;
        }
        return Math.min(1.0d, Math.max(0.0d, knownTypeTensorBytes / (double) tensorBytes));
    }

    public double decoderTensorRatio() {
        if (requiredDecoderTensorCount <= 0) {
            return 0.0d;
        }
        return presentDecoderTensorCount / (double) requiredDecoderTensorCount;
    }

    public int missingDecoderTensorCount() {
        return Math.max(0, requiredDecoderTensorCount - presentDecoderTensorCount - malformedDecoderTensorCount);
    }

    public boolean decoderTensorSetComplete() {
        return requiredDecoderTensorCount > 0
                && presentDecoderTensorCount == requiredDecoderTensorCount
                && malformedDecoderTensorCount == 0
                && unknownTensorTypeIds.isEmpty();
    }

    public boolean rowDotPrimitivesReady() {
        return decoderTensorSetComplete() && unsupportedRowDotTypeIds.isEmpty();
    }

    public String javaStatus() {
        if (!unknownTensorTypeIds.isEmpty()) {
            return "loader-ready; unknown-tensor-types=" + String.join(",", unknownTensorTypeIds)
                    + "; generation-disabled";
        }
        if (requiredDecoderTensorCount == 0) {
            return "loader-ready; decoder-shape-unknown; generation-disabled";
        }
        int missingTensorCount = missingDecoderTensorCount();
        if (missingTensorCount > 0 || malformedDecoderTensorCount > 0) {
            List<String> readinessProblems = new ArrayList<>(2);
            if (missingTensorCount > 0) {
                readinessProblems.add("decoder-tensors-missing=" + missingTensorCount);
            }
            if (malformedDecoderTensorCount > 0) {
                readinessProblems.add("decoder-tensor-shapes-invalid=" + malformedDecoderTensorCount);
            }
            return "loader-ready; " + String.join("; ", readinessProblems) + "; generation-disabled";
        }
        if (!unsupportedRowDotTypeIds.isEmpty()) {
            return "loader-ready; decoder-tensors-ready; row-dot-unsupported-types="
                    + String.join(",", unsupportedRowDotTypeIds)
                    + "; generation-disabled";
        }
        return "loader-ready; decoder-tensors-ready; row-dot-primitives-ready; generation-disabled";
    }

    public String compactTypeSummary(int limit) {
        if (tensorTypes.isEmpty() || limit <= 0) {
            return "none";
        }
        List<String> parts = new ArrayList<>();
        for (int i = 0; i < Math.min(limit, tensorTypes.size()); i++) {
            TensorTypeSummary summary = tensorTypes.get(i);
            parts.add(summary.label() + ":" + summary.count());
        }
        if (tensorTypes.size() > limit) {
            parts.add("+" + (tensorTypes.size() - limit));
        }
        return String.join(", ", parts);
    }

    private static List<String> requiredDecoderTensorNames(Map<String, Object> metadata, String architecture) {
        int blockCount = metadataInt(metadata,
                architecture + ".block_count",
                "llama.block_count",
                "gemma.block_count",
                "gemma2.block_count",
                "gemma3.block_count",
                "gemma4.block_count",
                "qwen2.block_count",
                "mistral.block_count",
                "phi3.block_count",
                "general.block_count");
        if (blockCount <= 0) {
            return List.of();
        }

        boolean requiresQkNorms = requiresGemmaQkNorms(architecture);
        List<String> required = new ArrayList<>(2 + blockCount * (requiresQkNorms ? 11 : 9));
        required.add("token_embd.weight");
        required.add("output_norm.weight");

        for (int layer = 0; layer < blockCount; layer++) {
            String prefix = "blk." + layer + ".";
            required.add(prefix + "attn_norm.weight");
            required.add(prefix + "attn_q.weight");
            required.add(prefix + "attn_k.weight");
            required.add(prefix + "attn_v.weight");
            if (requiresQkNorms) {
                required.add(prefix + "attn_q_norm.weight");
                required.add(prefix + "attn_k_norm.weight");
            }
            required.add(prefix + "attn_output.weight");
            required.add(prefix + "ffn_norm.weight");
            required.add(prefix + "ffn_gate.weight");
            required.add(prefix + "ffn_up.weight");
            required.add(prefix + "ffn_down.weight");
        }
        return required;
    }

    private static boolean requiresGemmaQkNorms(String architecture) {
        return architecture.equals("gemma3")
                || architecture.equals("gemma3_text")
                || architecture.equals("gemma4")
                || architecture.equals("gemma4_text");
    }

    private static boolean decoderTensorShapeValid(
            String name,
            GGUFTensorInfo tensor,
            DecoderHiddenSizeHint hints
    ) {
        if (!hints.hasHiddenSize()) {
            return true;
        }
        if (name.equals("token_embd.weight")) {
            return true;
        }
        if (name.equals("output_norm.weight")) {
            return true;
        }

        String suffix = decoderTensorSuffix(name);
        if (suffix == null) {
            return true;
        }
        if (suffix.equals("attn_norm.weight") || suffix.equals("ffn_norm.weight")) {
            return true;
        }
        if (suffix.equals("attn_q_norm.weight") || suffix.equals("attn_k_norm.weight")) {
            return true;
        }
        if (suffix.equals("attn_q.weight")
                || suffix.equals("attn_k.weight")
                || suffix.equals("attn_v.weight")
                || suffix.equals("ffn_gate.weight")
                || suffix.equals("ffn_up.weight")) {
            return matrixColumnsEqual(tensor, hints.hiddenSize());
        }
        return true;
    }

    private static String decoderTensorSuffix(String name) {
        if (!name.startsWith("blk.")) {
            return null;
        }
        int layerEnd = name.indexOf('.', 4);
        if (layerEnd < 0 || layerEnd + 1 >= name.length()) {
            return null;
        }
        return name.substring(layerEnd + 1);
    }

    private static boolean matrixColumnsEqual(GGUFTensorInfo tensor, long expected) {
        try {
            return GgufTensorOps.matrixColumns(tensor) == expected;
        } catch (IllegalArgumentException ignored) {
            return false;
        }
    }

    private static int metadataInt(Map<String, Object> metadata, String... keys) {
        for (String key : keys) {
            Object value = metadata.get(key);
            if (value instanceof Number number) {
                return number.intValue();
            }
        }
        return 0;
    }

    private record DecoderHiddenSizeHint(int hiddenSize) {
        private static DecoderHiddenSizeHint from(Map<String, Object> metadata, String architecture) {
            int hiddenSize = metadataInt(metadata,
                    architecture + ".embedding_length",
                    "llama.embedding_length",
                    "gemma.embedding_length",
                    "gemma2.embedding_length",
                    "gemma3.embedding_length",
                    "gemma4.embedding_length",
                    "qwen2.embedding_length",
                    "mistral.embedding_length",
                    "phi3.embedding_length",
                    "general.embedding_length");
            return new DecoderHiddenSizeHint(hiddenSize);
        }

        private boolean hasHiddenSize() {
            return hiddenSize > 0;
        }
    }

    public record TensorTypeSummary(
            String label,
            int typeId,
            int count,
            long bytes,
            boolean knownForJava
    ) {
    }

    private static final class TypeAccumulator {
        private int typeId = -1;
        private int count;
        private long bytes;
        private boolean knownForJava;

        private void add(GGUFTensorInfo tensor, long tensorBytes, boolean tensorKnownForJava) {
            if (typeId < 0) {
                typeId = tensor.typeId();
            }
            count++;
            bytes += tensorBytes;
            knownForJava |= tensorKnownForJava;
        }

        private TensorTypeSummary summary(String label) {
            return new TensorTypeSummary(label, typeId, count, bytes, knownForJava);
        }
    }
}
