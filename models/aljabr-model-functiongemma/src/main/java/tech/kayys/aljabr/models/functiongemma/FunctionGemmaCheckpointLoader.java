package tech.kayys.aljabr.models.functiongemma;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.model.ModelRuntimeTraits;

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
 * SafeTensors shard loader and resolves its model-specific tokenizer and
 * architecture policy.
 *
 * <p>This class prepares a validated checkpoint session; it does not execute
 * generated function calls or provide a second transformer implementation.</p>
 */
@ApplicationScoped
public class FunctionGemmaCheckpointLoader {

    @Inject
    ObjectMapper objectMapper;

    @Inject
    SafetensorShardLoader safetensorLoader;

    /**
     * Opens a complete local FunctionGemma checkpoint.
     *
     * @param modelDirectory Hugging Face checkpoint directory
     * @return a session owning the mapped SafeTensors resources
     * @throws IOException when config or tokenizer assets cannot be read
     * @throws IllegalArgumentException when the checkpoint is not FunctionGemma
     *         or required architecture tensors are absent
     */
    public FunctionGemmaCheckpointSession open(Path modelDirectory) throws IOException {
        Objects.requireNonNull(modelDirectory, "modelDirectory");
        Path root = modelDirectory.toAbsolutePath().normalize();
        if (!Files.isDirectory(root)) {
            throw new IllegalArgumentException("FunctionGemma checkpoint must be a directory: " + root);
        }

        ModelConfig config = ModelConfig.load(root.resolve("config.json"), objectMapper);
        FunctionGemmaModelFamilyPlugin family = new FunctionGemmaModelFamilyPlugin();
        if (!family.matchesCheckpoint(root)) {
            throw new IllegalArgumentException(
                    "Checkpoint does not match the FunctionGemma config and tokenizer protocol: " + root);
        }

        FunctionGemmaArchitecture architecture = new FunctionGemmaArchitecture();
        FunctionGemmaTokenizer tokenizer = FunctionGemmaTokenizer.load(root);
        SafetensorShardSession weights = safetensorLoader.open(root);
        try {
            validateRequiredTensors(weights.tensorNames(), config, architecture);
            ModelRuntimeTraits runtimeTraits = architecture.runtimeTraits(config);
            return new FunctionGemmaCheckpointSession(root, config, architecture,
                    tokenizer, runtimeTraits, weights);
        } catch (RuntimeException failure) {
            weights.close();
            throw failure;
        }
    }

    /**
     * Produces the minimum tensor manifest implied by the selected architecture
     * and the checkpoint's configured decoder depth.
     */
    static List<String> requiredTensorNames(ModelConfig config, FunctionGemmaArchitecture architecture) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(architecture, "architecture");
        if (config.numHiddenLayers() <= 0) {
            throw new IllegalArgumentException(
                    "FunctionGemma config must declare a positive num_hidden_layers value");
        }

        Set<String> names = new LinkedHashSet<>();
        add(names, architecture.embedTokensWeight());
        add(names, architecture.finalNormWeight());

        for (int layer = 0; layer < config.numHiddenLayers(); layer++) {
            add(names, architecture.layerQueryWeight(layer));
            add(names, architecture.layerKeyWeight(layer));
            add(names, architecture.layerValueWeight(layer));
            add(names, architecture.layerOutputWeight(layer));
            add(names, architecture.layerAttentionNormWeight(layer));
            add(names, architecture.layerPreFfnNormWeight(layer));
            add(names, architecture.layerFfnNormWeight(layer));
            add(names, architecture.layerPostFfnNormWeight(layer));
            add(names, architecture.layerQueryNormWeight(layer));
            add(names, architecture.layerKeyNormWeight(layer));
            add(names, architecture.layerFfnGateWeight(layer));
            add(names, architecture.layerFfnUpWeight(layer));
            add(names, architecture.layerFfnDownWeight(layer));
        }
        return List.copyOf(names);
    }

    private static void add(Set<String> names, String tensorName) {
        if (tensorName != null && !tensorName.isBlank()) {
            names.add(tensorName);
        }
    }

    private static void validateRequiredTensors(
            Set<String> available,
            ModelConfig config,
            FunctionGemmaArchitecture architecture) {
        List<String> missing = new ArrayList<>();
        for (String required : requiredTensorNames(config, architecture)) {
            if (!available.contains(required)) {
                missing.add(required);
            }
        }
        if (!missing.isEmpty()) {
            int shown = Math.min(12, missing.size());
            throw new IllegalArgumentException(
                    "FunctionGemma checkpoint is missing " + missing.size()
                            + " required architecture tensors: "
                            + String.join(", ", missing.subList(0, shown))
                            + (missing.size() > shown ? ", ..." : ""));
        }
    }

    /** Owns the open shard session; callers must close it after inference. */
    public record FunctionGemmaCheckpointSession(
            Path modelDirectory,
            ModelConfig config,
            FunctionGemmaArchitecture architecture,
            FunctionGemmaTokenizer tokenizer,
            ModelRuntimeTraits runtimeTraits,
            SafetensorShardSession weights) implements AutoCloseable {

        public FunctionGemmaCheckpointSession {
            Objects.requireNonNull(modelDirectory, "modelDirectory");
            Objects.requireNonNull(config, "config");
            Objects.requireNonNull(architecture, "architecture");
            Objects.requireNonNull(tokenizer, "tokenizer");
            Objects.requireNonNull(runtimeTraits, "runtimeTraits");
            Objects.requireNonNull(weights, "weights");
        }

        @Override
        public void close() {
            weights.close();
        }
    }
}
