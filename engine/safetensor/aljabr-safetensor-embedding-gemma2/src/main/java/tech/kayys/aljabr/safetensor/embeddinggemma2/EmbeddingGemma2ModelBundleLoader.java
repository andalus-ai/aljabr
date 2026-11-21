package tech.kayys.aljabr.safetensor.embeddinggemma2;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2TextBackbone;
import tech.kayys.aljabr.models.embeddinggemma2.EmbeddingGemma2TextModel;
import tech.kayys.aljabr.models.embeddinggemma2.config.EmbeddingGemma2Config;
import tech.kayys.aljabr.models.embeddinggemma2.input.EmbeddingGemma2Tokenizer;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;
import tech.kayys.aljabr.spi.tokenizer.Tokenizer;
import tech.kayys.aljabr.spi.tokenizer.TokenizerFactory;

/**
 * Assembles a Hugging Face EmbeddingGemma 2 text bundle from its config,
 * shared tokenizer JSON, and single-file or sharded Safetensors checkpoint.
 *
 * <p>The shared {@link TokenizerFactory} remains the sole tokenizer
 * implementation entry point. This adapter adds only the model-specific BOS,
 * task-formatting, mask, and position contract.</p>
 */
@ApplicationScoped
public final class EmbeddingGemma2ModelBundleLoader {
    private final ObjectMapper objectMapper;
    private final SafetensorShardLoader shardLoader;

    @Inject
    public EmbeddingGemma2ModelBundleLoader(
            ObjectMapper objectMapper,
            SafetensorShardLoader shardLoader) {
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
        this.shardLoader = Objects.requireNonNull(shardLoader, "shardLoader must not be null");
    }

    /**
     * Loads config.json, tokenizer.json (or tokenizer/tokenizer.json), and the
     * model's Safetensors weights from one model directory.
     */
    public EmbeddingGemma2TextModel load(Path modelDirectory) throws IOException {
        Objects.requireNonNull(modelDirectory, "modelDirectory must not be null");
        Path directory = modelDirectory.toAbsolutePath().normalize();
        if (!Files.isDirectory(directory)) {
            throw new IOException("EmbeddingGemma 2 model path must be a directory: " + directory);
        }

        Path configPath = directory.resolve("config.json");
        if (!Files.isRegularFile(configPath)) {
            throw new IOException("EmbeddingGemma 2 bundle is missing config.json: " + directory);
        }

        JsonNode rawConfig = objectMapper.readTree(configPath.toFile());
        EmbeddingGemma2Config config = EmbeddingGemma2Config.fromJson(rawConfig);

        Path tokenizerPath = resolveTokenizer(directory);
        Tokenizer sharedTokenizer = TokenizerFactory.load(tokenizerPath, null);
        EmbeddingGemma2Tokenizer tokenizer = new EmbeddingGemma2Tokenizer(config, sharedTokenizer);

        EmbeddingGemma2SafetensorLoader checkpointLoader =
                new EmbeddingGemma2SafetensorLoader(shardLoader);
        EmbeddingGemma2TextBackbone backbone = checkpointLoader.load(directory, config);
        return new EmbeddingGemma2TextModel(config, tokenizer, backbone);
    }

    private static Path resolveTokenizer(Path directory) throws IOException {
        Path rootTokenizer = directory.resolve("tokenizer.json");
        if (Files.isRegularFile(rootTokenizer)) return rootTokenizer;

        Path nestedTokenizer = directory.resolve("tokenizer").resolve("tokenizer.json");
        if (Files.isRegularFile(nestedTokenizer)) return nestedTokenizer;

        throw new IOException("EmbeddingGemma 2 bundle is missing tokenizer.json: " + directory);
    }
}
