package tech.kayys.aljabr.safetensor.functiongemma;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.models.functiongemma.FunctionGemmaArchitecture;
import tech.kayys.aljabr.spi.model.ModelConfig;

class FunctionGemmaSafetensorCheckpointLoaderTest {

    private static final List<String> FUNCTION_TOKENS = List.of(
            "<start_function_declaration>",
            "<end_function_declaration>",
            "<start_function_call>",
            "<end_function_call>",
            "<start_function_response>",
            "<end_function_response>",
            "<escape>");

    @TempDir
    Path tempDir;

    @Test
    void acceptsTheCompleteGemma3TextTensorLayout() throws Exception {
        ModelConfig config = config();
        FunctionGemmaArchitecture architecture = new FunctionGemmaArchitecture();
        Set<String> tensors = requiredTensorNames(config, architecture);

        FunctionGemmaSafetensorCheckpointLoader.validateTensorLayout(
                tensors, config, architecture);
    }

    @Test
    void reportsMissingRequiredWeightsBeforeReturningCheckpoint() throws Exception {
        ModelConfig config = config();
        IllegalArgumentException failure = org.junit.jupiter.api.Assertions.assertThrows(
                IllegalArgumentException.class,
                () -> FunctionGemmaSafetensorCheckpointLoader.validateTensorLayout(
                        Set.of(), config, new FunctionGemmaArchitecture()));
        assertTrue(failure.getMessage().contains("missing 15 required tensors"));
    }

    @Test
    void rejectsOrdinaryGemma3BeforeOpeningSafetensors() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), configJson());
        Files.writeString(tempDir.resolve("tokenizer_config.json"), """
                {"added_tokens_decoder": {}}
                """);

        var mustNotOpen = new tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader() {
            @Override
            public SafetensorShardSession open(Path modelPath) {
                throw new AssertionError("weights must not be opened for a non-FunctionGemma checkpoint");
            }
        };
        FunctionGemmaSafetensorCheckpointLoader loader =
                new FunctionGemmaSafetensorCheckpointLoader(mustNotOpen, new ObjectMapper());

        IllegalArgumentException failure = org.junit.jupiter.api.Assertions.assertThrows(
                IllegalArgumentException.class, () -> loader.load(tempDir));
        assertTrue(failure.getMessage().contains("not a recognized FunctionGemma checkpoint"));
    }

    @Test
    void loadsSyntheticSafetensorsCheckpointIntoExecutableModel() throws Exception {
        Files.writeString(tempDir.resolve("config.json"), configJson());
        Files.writeString(tempDir.resolve("tokenizer.json"), tokenizerJson());
        writeSyntheticCheckpoint(tempDir.resolve("model.safetensors"));

        FunctionGemmaSafetensorCheckpointLoader loader = newLoaderWithRealFfmStack();
        FunctionGemmaModel model = loader.loadModel(tempDir);

        assertNotNull(model);
        assertEquals(1, model.layerCount());
        assertEquals(4, model.hiddenSize());
        assertEquals(5, model.vocabularySize());
        assertEquals(FUNCTION_TOKENS.size(), FUNCTION_TOKENS.stream()
                .filter(model.tokenizer().specialTokens()::containsKey).count());

        float[][] logits = model.forward(new long[] {0, 1, 2});
        assertEquals(3, logits.length);
        for (float[] tokenLogits : logits) {
            assertEquals(5, tokenLogits.length);
            for (float logit : tokenLogits) {
                assertTrue(Float.isFinite(logit), "logits must be finite");
            }
        }
    }

    private FunctionGemmaSafetensorCheckpointLoader newLoaderWithRealFfmStack() throws Exception {
        ObjectMapper mapper = new ObjectMapper();

        Class<?> configType = Class.forName(
                "tech.kayys.aljabr.safetensor.loader.SafetensorLoaderConfig");
        Object config = configProxy(configType);

        Class<?> metricsType = Class.forName(
                "tech.kayys.aljabr.safetensor.loader.SafetensorMetrics");
        Object metrics = metricsType.getConstructor().newInstance();
        Class<?> registryType = Class.forName("io.micrometer.core.instrument.MeterRegistry");
        Object registry = Class.forName(
                "io.micrometer.core.instrument.simple.SimpleMeterRegistry")
                .getConstructor().newInstance();
        setField(metrics, "registry", registry);
        var registerGauge = metricsType.getDeclaredMethod("registerGauge");
        registerGauge.setAccessible(true);
        registerGauge.invoke(metrics);

        Class<?> ffmType = Class.forName(
                "tech.kayys.aljabr.safetensor.loader.SafetensorFFMLoader");
        Object ffmLoader = ffmType.getConstructor().newInstance();
        setField(ffmLoader, "config", config);
        setField(ffmLoader, "objectMapper", mapper);
        setField(ffmLoader, "metrics", metrics);

        var shardLoader = new tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader();
        setField(shardLoader, "ffmLoader", ffmLoader);
        setField(shardLoader, "objectMapper", mapper);
        return new FunctionGemmaSafetensorCheckpointLoader(shardLoader, mapper);
    }

    private static Object configProxy(Class<?> configType) {
        InvocationHandler handler = (proxy, method, args) -> {
            return switch (method.getName()) {
                case "validation", "cache" -> configProxy(method.getReturnType());
                case "preferMmap", "strict", "warnOnEmptyTensors",
                        "enabled", "logLoadSummary" -> true;
                case "readChunkBytes", "maxSize" -> 8192;
                case "maxHeaderBytes" -> 1_048_576L;
                case "ttlSeconds" -> 300L;
                case "toString" -> "SyntheticSafetensorLoaderConfig";
                default -> method.getReturnType().isPrimitive()
                        ? (method.getReturnType() == boolean.class ? false : 0)
                        : null;
            };
        };
        return Proxy.newProxyInstance(
                configType.getClassLoader(), new Class<?>[] {configType}, handler);
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(target, value);
    }

    private ModelConfig config() throws Exception {
        Path path = tempDir.resolve("config.json");
        Files.writeString(path, configJson());
        return ModelConfig.load(path, new ObjectMapper());
    }

    private static String configJson() {
        return """
                {
                  "model_type": "gemma3_text",
                  "architectures": ["Gemma3ForCausalLM"],
                  "hidden_size": 4,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "head_dim": 2,
                  "intermediate_size": 6,
                  "vocab_size": 5,
                  "max_position_embeddings": 16,
                  "tie_word_embeddings": true,
                  "layer_types": ["full_attention"]
                }
                """;
    }

    private static String tokenizerJson() throws IOException {
        Map<String, Object> root = new LinkedHashMap<>();
        Map<String, Object> model = new LinkedHashMap<>();
        model.put("type", "BPE");
        model.put("vocab", Map.of("a", 0, "b", 1, "c", 2, "d", 3, "e", 4));
        model.put("merges", List.of());
        root.put("model", model);
        List<Map<String, Object>> added = new ArrayList<>();
        int id = 5;
        for (String token : FUNCTION_TOKENS) {
            added.add(Map.of("id", id++, "content", token, "special", true));
        }
        root.put("added_tokens", added);
        return new ObjectMapper().writeValueAsString(root);
    }

    private static Set<String> requiredTensorNames(
            ModelConfig config, FunctionGemmaArchitecture architecture) {
        Set<String> names = new LinkedHashSet<>();
        names.add(architecture.embedTokensWeight());
        names.add(architecture.finalNormWeight());
        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
            names.add(architecture.layerQueryWeight(layer));
            names.add(architecture.layerKeyWeight(layer));
            names.add(architecture.layerValueWeight(layer));
            names.add(architecture.layerOutputWeight(layer));
            names.add(architecture.layerAttentionNormWeight(layer));
            names.add(architecture.layerPostAttnNormWeight(layer));
            names.add(architecture.layerPreFfnNormWeight(layer));
            names.add(architecture.layerPostFfnNormWeight(layer));
            names.add(architecture.layerQueryNormWeight(layer));
            names.add(architecture.layerKeyNormWeight(layer));
            names.add(architecture.layerFfnGateWeight(layer));
            names.add(architecture.layerFfnUpWeight(layer));
            names.add(architecture.layerFfnDownWeight(layer));
        }
        return names;
    }

    private static void writeSyntheticCheckpoint(Path path) throws IOException {
        FunctionGemmaArchitecture architecture = new FunctionGemmaArchitecture();
        Map<String, TensorFixture> tensors = new LinkedHashMap<>();
        addMatrix(tensors, architecture.embedTokensWeight(), 5, 4, 0.025f);
        addVector(tensors, architecture.finalNormWeight(), 4, 0.0f);
        addMatrix(tensors, architecture.layerQueryWeight(0), 4, 4, 0.02f);
        addMatrix(tensors, architecture.layerKeyWeight(0), 2, 4, 0.015f);
        addMatrix(tensors, architecture.layerValueWeight(0), 2, 4, 0.03f);
        addMatrix(tensors, architecture.layerOutputWeight(0), 4, 4, 0.025f);
        addVector(tensors, architecture.layerAttentionNormWeight(0), 4, 0.0f);
        addVector(tensors, architecture.layerPostAttnNormWeight(0), 4, 0.0f);
        addVector(tensors, architecture.layerPreFfnNormWeight(0), 4, 0.0f);
        addVector(tensors, architecture.layerPostFfnNormWeight(0), 4, 0.0f);
        addVector(tensors, architecture.layerQueryNormWeight(0), 2, 0.0f);
        addVector(tensors, architecture.layerKeyNormWeight(0), 2, 0.0f);
        addMatrix(tensors, architecture.layerFfnGateWeight(0), 6, 4, 0.01f);
        addMatrix(tensors, architecture.layerFfnUpWeight(0), 6, 4, 0.012f);
        addMatrix(tensors, architecture.layerFfnDownWeight(0), 4, 6, 0.008f);

        ObjectMapper mapper = new ObjectMapper();
        Map<String, Object> header = new LinkedHashMap<>();
        long offset = 0;
        for (TensorFixture tensor : tensors.values()) {
            long bytes = tensor.values().length * Float.BYTES;
            header.put(tensor.name(), Map.of(
                    "dtype", "F32",
                    "shape", tensor.shape(),
                    "data_offsets", List.of(offset, offset + bytes)));
            offset += bytes;
        }
        byte[] json = mapper.writeValueAsBytes(header);
        int padding = (int) ((8 - ((8 + json.length) % 8)) % 8);
        byte[] paddedHeader = java.util.Arrays.copyOf(json, json.length + padding);
        java.util.Arrays.fill(paddedHeader, json.length, paddedHeader.length, (byte) ' ');

        int dataBytes = Math.toIntExact(offset);
        ByteBuffer file = ByteBuffer.allocate(8 + paddedHeader.length + dataBytes)
                .order(ByteOrder.LITTLE_ENDIAN);
        file.putLong(paddedHeader.length);
        file.put(paddedHeader);
        for (TensorFixture tensor : tensors.values()) {
            for (float value : tensor.values()) {
                file.putFloat(value);
            }
        }
        Files.write(path, file.array());
    }

    private static void addMatrix(
            Map<String, TensorFixture> tensors, String name,
            int rows, int columns, float scale) {
        float[] values = new float[rows * columns];
        for (int index = 0; index < values.length; index++) {
            values[index] = (index + 1) * scale;
        }
        tensors.put(name, new TensorFixture(name, new long[] {rows, columns}, values));
    }

    private static void addVector(
            Map<String, TensorFixture> tensors, String name, int length, float value) {
        float[] values = new float[length];
        java.util.Arrays.fill(values, value);
        tensors.put(name, new TensorFixture(name, new long[] {length}, values));
    }

    private record TensorFixture(String name, long[] shape, float[] values) {}
}
