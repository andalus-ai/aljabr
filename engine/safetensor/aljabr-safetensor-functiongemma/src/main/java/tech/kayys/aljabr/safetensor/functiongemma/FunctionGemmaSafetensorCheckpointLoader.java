package tech.kayys.aljabr.safetensor.functiongemma;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.aljabr.models.functiongemma.FunctionGemmaArchitecture;
import tech.kayys.aljabr.models.functiongemma.FunctionGemmaModelFamilyPlugin;
import tech.kayys.aljabr.models.functiongemma.FunctionGemmaTokenizer;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Opens a FunctionGemma Hugging Face checkpoint through Aljabr's shared
 * SafeTensors shard loader and selects the family architecture and tokenizer
 * from checkpoint metadata.
 *
 * <p>This is a model-family adapter, not a second tensor storage engine.
 * The returned checkpoint owns the mapped SafeTensors session and must be
 * closed by its caller.</p>
 */
@ApplicationScoped
public final class FunctionGemmaSafetensorCheckpointLoader {

    private final SafetensorShardLoader shardLoader;
    private final ObjectMapper objectMapper;
    private final FunctionGemmaModelFamilyPlugin familyPlugin;

    @Inject
    public FunctionGemmaSafetensorCheckpointLoader(
            SafetensorShardLoader shardLoader,
            ObjectMapper objectMapper) {
        this(shardLoader, objectMapper, new FunctionGemmaModelFamilyPlugin());
    }

    FunctionGemmaSafetensorCheckpointLoader(
            SafetensorShardLoader shardLoader,
            ObjectMapper objectMapper,
            FunctionGemmaModelFamilyPlugin familyPlugin) {
        this.shardLoader = Objects.requireNonNull(shardLoader, "shardLoader");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper");
        this.familyPlugin = Objects.requireNonNull(familyPlugin, "familyPlugin");
    }

    /**
     * Resolves metadata, loads the FunctionGemma tokenizer, opens single-file
     * or sharded SafeTensors, and verifies the model-family tensor layout.
     */
    public LoadedCheckpoint load(Path modelDirectory) throws IOException {
        Objects.requireNonNull(modelDirectory, "modelDirectory");
        Path directory = modelDirectory.toAbsolutePath().normalize();
        if (!Files.isDirectory(directory)) {
            throw new IllegalArgumentException("FunctionGemma checkpoint must be a model directory: " + directory);
        }
        if (!familyPlugin.matchesCheckpoint(directory)) {
            throw new IllegalArgumentException(
                    "Checkpoint is not a recognized FunctionGemma checkpoint: " + directory);
        }

        ModelConfig config = ModelConfig.load(directory.resolve("config.json"), objectMapper);
        FunctionGemmaTokenizer tokenizer = FunctionGemmaTokenizer.load(directory);
        FunctionGemmaArchitecture architecture = selectArchitecture();

        SafetensorShardSession session = shardLoader.open(directory);
        boolean success = false;
        try {
            validateTensorLayout(session, config, architecture);
            LoadedCheckpoint loaded = new LoadedCheckpoint(directory, config, architecture, tokenizer, session);
            success = true;
            return loaded;
        } finally {
            if (!success) {
                session.close();
            }
        }
    }

    private FunctionGemmaArchitecture selectArchitecture() {
        List<ModelArchitecture> adapters = familyPlugin.architectureAdapters();
        if (adapters.size() != 1 || !(adapters.getFirst() instanceof FunctionGemmaArchitecture architecture)) {
            throw new IllegalStateException(
                    "FunctionGemma must resolve to exactly one FunctionGemmaArchitecture adapter");
        }
        return architecture;
    }

    private static void validateTensorLayout(
            SafetensorShardSession session,
            ModelConfig config,
            FunctionGemmaArchitecture architecture) {
        Set<String> available = session.tensorNames();
        List<String> missing = new ArrayList<>();
        requireOne(available, missing, "token embeddings", architecture.embedTokensWeightCandidates());
        requireOne(available, missing, "final norm", architecture.finalNormWeightCandidates());

        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
            requireOne(available, missing, "layer " + layer + " query projection",
                    architecture.layerQueryWeightCandidates(layer));
            requireOne(available, missing, "layer " + layer + " key projection",
                    architecture.layerKeyWeightCandidates(layer));
            requireOne(available, missing, "layer " + layer + " value projection",
                    architecture.layerValueWeightCandidates(layer));
            requireOne(available, missing, "layer " + layer + " attention output",
                    architecture.layerOutputWeightCandidates(layer));
            requireOne(available, missing, "layer " + layer + " FFN gate",
                    architecture.layerFfnGateWeightCandidates(layer));
            requireOne(available, missing, "layer " + layer + " FFN up",
                    architecture.layerFfnUpWeightCandidates(layer));
            requireOne(available, missing, "layer " + layer + " FFN down",
                    architecture.layerFfnDownWeightCandidates(layer));
        }

        if (!missing.isEmpty()) {
            throw new IllegalArgumentException(
                    "SafeTensors checkpoint does not match the FunctionGemma/Gemma 3 text weight layout; missing "
                            + missing.size() + " required tensors: " + String.join(", ", missing));
        }
    }

    private static void requireOne(
            Set<String> available,
            List<String> missing,
            String role,
            List<String> candidates) {
        List<String> usable = candidates == null ? List.of()
                : candidates.stream().filter(Objects::nonNull).filter(name -> !name.isBlank()).toList();
        if (usable.isEmpty() || usable.stream().noneMatch(available::contains)) {
            missing.add(role + " [" + String.join(" | ", usable) + "]");
        }
    }

    /**
     * A live checkpoint handle. Tensor views are valid only until close().
     */
    public record LoadedCheckpoint(
            Path modelDirectory,
            ModelConfig config,
            FunctionGemmaArchitecture architecture,
            FunctionGemmaTokenizer tokenizer,
            SafetensorShardSession tensors) implements AutoCloseable {

        public LoadedCheckpoint {
            Objects.requireNonNull(modelDirectory);
            Objects.requireNonNull(config);
            Objects.requireNonNull(architecture);
            Objects.requireNonNull(tokenizer);
            Objects.requireNonNull(tensors);
        }

        public Set<String> tensorNames() {
            return Set.copyOf(new LinkedHashSet<>(tensors.tensorNames()));
        }

        @Override
        public void close() {
            tensors.close();
        }
    }
}
