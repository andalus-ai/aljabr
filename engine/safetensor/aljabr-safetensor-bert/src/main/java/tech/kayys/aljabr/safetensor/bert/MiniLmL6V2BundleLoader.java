package tech.kayys.aljabr.safetensor.bert;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.models.bert.BertConfig;
import tech.kayys.aljabr.models.bert.BertEncoder;
import tech.kayys.aljabr.models.bert.MiniLmL6V2;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;
import tech.kayys.aljabr.spi.tokenizer.TokenizerFactory;

/** Loads the official Sentence Transformers MiniLM directory layout without Python dependencies. */
@ApplicationScoped
public final class MiniLmL6V2BundleLoader {
    private final ObjectMapper objectMapper;
    private final BertSafetensorLoader safetensorLoader;

    @Inject
    public MiniLmL6V2BundleLoader(ObjectMapper objectMapper, BertSafetensorLoader safetensorLoader) {
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper");
        this.safetensorLoader = Objects.requireNonNull(safetensorLoader, "safetensorLoader");
    }

    /**
     * Loads config.json, Sentence Transformers pooling/sequence metadata,
     * tokenizer.json and model.safetensors from a local Hugging Face snapshot.
     */
    public MiniLmL6V2Bundle load(Path modelDirectory, ComputeBackend backend) throws IOException {
        Objects.requireNonNull(modelDirectory, "modelDirectory");
        Objects.requireNonNull(backend, "backend");
        Path root = modelDirectory.toAbsolutePath().normalize();
        if (!Files.isDirectory(root)) {
            throw new IOException("MiniLM model path must be a directory: " + root);
        }

        Map<String, Object> configJson = readObject(root.resolve("config.json"));
        BertConfig config = BertConfig.from(configJson);
        MiniLmL6V2.validateCompatible(config);
        validateSentenceTransformerLayout(root);

        Tokenizer tokenizer = TokenizerFactory.load(root, null);
        if (tokenizer.vocabSize() != config.vocabularySize()) {
            throw new IOException("MiniLM tokenizer vocabulary size " + tokenizer.vocabSize()
                    + " does not match config vocab_size " + config.vocabularySize());
        }

        BertEncoder encoder = new BertEncoder(backend, config);
        MiniLmL6V2 model = new MiniLmL6V2(encoder);
        try {
            safetensorLoader.load(root, encoder);
            return new MiniLmL6V2Bundle(model, tokenizer);
        } catch (RuntimeException | Error failure) {
            encoder.close();
            throw failure;
        }
    }

    private Map<String, Object> readObject(Path path) throws IOException {
        if (!Files.isRegularFile(path)) throw new IOException("Missing required MiniLM metadata: " + path);
        return objectMapper.readValue(path.toFile(), new TypeReference<>() { });
    }

    private void validateSentenceTransformerLayout(Path root) throws IOException {
        Map<String, Object> sentenceConfig = readObject(root.resolve("sentence_bert_config.json"));
        int maxSequenceLength = integer(sentenceConfig, "max_seq_length");
        if (maxSequenceLength != MiniLmL6V2.MAX_SEQUENCE_LENGTH) {
            throw new IOException("Expected MiniLM max_seq_length "
                    + MiniLmL6V2.MAX_SEQUENCE_LENGTH + " but found " + maxSequenceLength);
        }

        Map<String, Object> pooling = readObject(root.resolve("1_Pooling/config.json"));
        if (integer(pooling, "word_embedding_dimension") != MiniLmL6V2.EMBEDDING_DIMENSION
                || !bool(pooling, "pooling_mode_mean_tokens")
                || bool(pooling, "pooling_mode_cls_token")
                || bool(pooling, "pooling_mode_max_tokens")
                || bool(pooling, "pooling_mode_mean_sqrt_len_tokens")
                || bool(pooling, "pooling_mode_lasttoken")) {
            throw new IOException("Unsupported MiniLM pooling configuration; expected mean pooling at 384 dimensions");
        }

        Path modulesPath = root.resolve("modules.json");
        List<Map<String, Object>> modules = objectMapper.readValue(
                modulesPath.toFile(), new TypeReference<>() { });
        if (modules.size() != 3
                || !moduleType(modules.get(0)).endsWith("Transformer")
                || !moduleType(modules.get(1)).endsWith("Pooling")
                || !moduleType(modules.get(2)).endsWith("Normalize")) {
            throw new IOException("Unsupported Sentence Transformers module sequence; expected Transformer, Pooling, Normalize");
        }
    }

    private static String moduleType(Map<String, Object> module) throws IOException {
        Object value = module.get("type");
        if (!(value instanceof String type)) throw new IOException("Sentence Transformers module is missing type");
        return type;
    }

    private static int integer(Map<String, Object> values, String key) throws IOException {
        Object value = values.get(key);
        if (!(value instanceof Number number) || number.doubleValue() != Math.rint(number.doubleValue())) {
            throw new IOException("MiniLM metadata field '" + key + "' must be an integer");
        }
        return number.intValue();
    }

    private static boolean bool(Map<String, Object> values, String key) throws IOException {
        Object value = values.get(key);
        if (value == null) return false;
        if (!(value instanceof Boolean result)) throw new IOException("MiniLM metadata field '" + key + "' must be boolean");
        return result;
    }
}
