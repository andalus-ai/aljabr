package tech.kayys.aljabr.safetensor.bert;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.lang.reflect.Field;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.safetensor.loader.SafetensorFFMLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorLoaderConfig;
import tech.kayys.aljabr.safetensor.loader.SafetensorMetrics;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;

/**
 * Opt-in integration test against a locally downloaded official
 * sentence-transformers/all-MiniLM-L6-v2 snapshot and reference vectors.
 *
 * <p>Set ALJABR_MINILM_MODEL_DIR and ALJABR_MINILM_REFERENCE_FILE to run.
 * The reference JSON schema is {"sentences":["..."],"embeddings":[[...]]}.
 * Reference vectors should be generated with the official Sentence Transformers
 * implementation, not from this Java implementation.</p>
 */
class MiniLmOfficialCheckpointParityTest {
    private static final String MODEL_ENV = "ALJABR_MINILM_MODEL_DIR";
    private static final String REFERENCE_ENV = "ALJABR_MINILM_REFERENCE_FILE";

    @Test
    void officialCheckpointEmbeddingsMatchIndependentReferenceVectors() throws Exception {
        String modelValue = System.getenv(MODEL_ENV);
        String referenceValue = System.getenv(REFERENCE_ENV);
        Assumptions.assumeTrue(modelValue != null && !modelValue.isBlank(),
                MODEL_ENV + " is not configured");
        Assumptions.assumeTrue(referenceValue != null && !referenceValue.isBlank(),
                REFERENCE_ENV + " is not configured");

        Path modelDirectory = Path.of(modelValue).toAbsolutePath().normalize();
        Path referenceFile = Path.of(referenceValue).toAbsolutePath().normalize();
        Assumptions.assumeTrue(Files.isDirectory(modelDirectory), "MiniLM model directory does not exist");
        Assumptions.assumeTrue(Files.isRegularFile(referenceFile), "MiniLM reference JSON does not exist");

        ObjectMapper mapper = new ObjectMapper();
        JsonNode reference = mapper.readTree(referenceFile.toFile());
        List<String> sentences = new ArrayList<>();
        reference.path("sentences").forEach(node -> sentences.add(node.asText()));
        JsonNode expected = reference.path("embeddings");
        assertTrue(!sentences.isEmpty(), "Reference must contain at least one sentence");
        assertEquals(sentences.size(), expected.size(), "Sentence/reference row count mismatch");

        CpuBackend backend = new CpuBackend();
        try (MiniLmL6V2Bundle bundle = loader(mapper).load(modelDirectory, backend)) {
            float[][] actual = bundle.encode(sentences);
            assertEquals(expected.size(), actual.length);
            double minCosine = Double.parseDouble(
                    System.getenv().getOrDefault("ALJABR_MINILM_MIN_COSINE", "0.999"));
            double maxAbsError = Double.parseDouble(
                    System.getenv().getOrDefault("ALJABR_MINILM_MAX_ABS_ERROR", "0.005"));

            for (int row = 0; row < actual.length; row++) {
                assertEquals(384, actual[row].length, "Unexpected embedding dimension at row " + row);
                assertEquals(384, expected.get(row).size(), "Reference dimension mismatch at row " + row);
                double dot = 0.0;
                double actualNorm = 0.0;
                double expectedNorm = 0.0;
                double largestError = 0.0;
                for (int column = 0; column < 384; column++) {
                    double wanted = expected.get(row).get(column).asDouble();
                    double observed = actual[row][column];
                    assertTrue(Double.isFinite(observed), "Non-finite embedding value");
                    dot += observed * wanted;
                    actualNorm += observed * observed;
                    expectedNorm += wanted * wanted;
                    largestError = Math.max(largestError, Math.abs(observed - wanted));
                }
                double cosine = dot / Math.sqrt(actualNorm * expectedNorm);
                assertTrue(cosine >= minCosine,
                        "Embedding cosine parity failed at row " + row + ": " + cosine);
                assertTrue(largestError <= maxAbsError,
                        "Embedding max absolute error failed at row " + row + ": " + largestError);
            }
        } finally {
            backend.close();
        }
    }

    private static MiniLmL6V2BundleLoader loader(ObjectMapper mapper) throws Exception {
        SafetensorLoaderConfig config = configProxy();
        SafetensorMetrics metrics = new SafetensorMetrics() {
            @Override
            public void recordLoad(Path path,
                    tech.kayys.aljabr.safetensor.loader.SafetensorLoadResult.LoadMode mode,
                    long byteCount, Duration elapsed) { }
            @Override
            public void recordClose() { }
        };
        SafetensorFFMLoader ffm = new SafetensorFFMLoader();
        setField(ffm, "config", config);
        setField(ffm, "objectMapper", mapper);
        setField(ffm, "metrics", metrics);

        SafetensorShardLoader shards = new SafetensorShardLoader();
        setField(shards, "ffmLoader", ffm);
        setField(shards, "objectMapper", mapper);
        return new MiniLmL6V2BundleLoader(mapper, new BertSafetensorLoader(shards));
    }

    private static SafetensorLoaderConfig configProxy() {
        return proxy(SafetensorLoaderConfig.class);
    }

    @SuppressWarnings("unchecked")
    private static <T> T proxy(Class<T> type) {
        return (T) Proxy.newProxyInstance(type.getClassLoader(), new Class<?>[] {type},
                (instance, method, arguments) -> {
                    Class<?> result = method.getReturnType();
                    if (result == boolean.class) return true;
                    if (result == int.class) return 8_388_608;
                    if (result == long.class) return 300L;
                    if (result.isInterface()) return proxy(result);
                    return null;
                });
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(target, value);
    }
}
