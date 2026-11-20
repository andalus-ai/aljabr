package tech.kayys.aljabr.safetensor.functiongemma;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.aljabr.models.functiongemma.FunctionGemmaArchitecture;
import tech.kayys.aljabr.models.functiongemma.FunctionGemmaModelFamilyPlugin;
import tech.kayys.aljabr.models.functiongemma.FunctionGemmaTokenizer;
import tech.kayys.aljabr.models.gemma3.Gemma3TextBackbone;
import tech.kayys.aljabr.models.gemma3.Gemma3TextCheckpointMapper;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
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
            validateTensorLayout(session.tensorNames(), config, architecture);
            LoadedCheckpoint loaded = new LoadedCheckpoint(directory, config, architecture, tokenizer, session);
            success = true;
            return loaded;
        } finally {
            if (!success) {
                session.close();
            }
        }
    }

    /**
     * Loads a FunctionGemma checkpoint into a detached, executable CPU
     * reference model. The SafeTensors session is closed after all required
     * weights have been materialized into the mapper's canonical float32
     * representation.
     */
    public FunctionGemmaModel loadModel(Path modelDirectory) throws IOException {
        try (LoadedCheckpoint checkpoint = load(modelDirectory)) {
            Map<String, Gemma3TextCheckpointMapper.CheckpointTensor> tensors =
                    materializeTensors(checkpoint);
            Gemma3TextCheckpointMapper.Weights weights =
                    Gemma3TextCheckpointMapper.map(
                            checkpoint.config(), checkpoint.architecture(), tensors);
            tensors.clear();
            Gemma3TextBackbone backbone =
                    new Gemma3TextBackbone(checkpoint.config(), weights);
            return new FunctionGemmaModel(checkpoint.config(), checkpoint.tokenizer(), backbone);
        }
    }

    private static Map<String, Gemma3TextCheckpointMapper.CheckpointTensor> materializeTensors(
            LoadedCheckpoint checkpoint) {
        FunctionGemmaArchitecture architecture = checkpoint.architecture();
        ModelConfig config = checkpoint.config();
        List<String> names = new ArrayList<>();
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

        Map<String, Gemma3TextCheckpointMapper.CheckpointTensor> result = new LinkedHashMap<>();
        for (String name : names) {
            var tensor = checkpoint.tensors().tensor(name);
            long[] longShape = tensor.shape();
            int[] shape = new int[longShape.length];
            for (int dimension = 0; dimension < longShape.length; dimension++) {
                shape[dimension] = Math.toIntExact(longShape[dimension]);
            }
            result.put(name, new Gemma3TextCheckpointMapper.CheckpointTensor(
                    shape, tensor.toFloatArray()));
        }
        return result;
    }

    private FunctionGemmaArchitecture selectArchitecture() {
        List<ModelArchitecture> adapters = familyPlugin.architectureAdapters();
        if (adapters.size() != 1 || !(adapters.getFirst() instanceof FunctionGemmaArchitecture architecture)) {
            throw new IllegalStateException(
                    "FunctionGemma must resolve to exactly one FunctionGemmaArchitecture adapter");
        }
        return architecture;
    }

    static void validateTensorLayout(
            Set<String> available,
            ModelConfig config,
            FunctionGemmaArchitecture architecture) {
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
            requireOne(available, missing, "layer " + layer + " attention norm",
                    architecture.layerAttentionNormWeightCandidates(layer));
            requireOne(available, missing, "layer " + layer + " post-attention norm",
                    architecture.layerPostAttnNormWeightCandidates(layer));
            requireOne(available, missing, "layer " + layer + " pre-FFN norm",
                    architecture.layerPreFfnNormWeightCandidates(layer));
            requireOne(available, missing, "layer " + layer + " post-FFN norm",
                    architecture.layerPostFfnNormWeightCandidates(layer));
            requireOne(available, missing, "layer " + layer + " query norm",
                    architecture.layerQueryNormWeightCandidates(layer));
            requireOne(available, missing, "layer " + layer + " key norm",
                    architecture.layerKeyNormWeightCandidates(layer));
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
            return Set.copyOf(tensors.tensorNames());
        }

        @Override
        public void close() {
            tensors.close();
        }
    }
}
