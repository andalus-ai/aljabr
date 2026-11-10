package tech.kayys.aljabr.ml.hub;

import tech.kayys.aljabr.ml.autograd.GradTensor;
import tech.kayys.aljabr.ml.nn.NNModule;
import tech.kayys.aljabr.ml.nn.Parameter;
import tech.kayys.aljabr.model.core.ModelRepository;
import tech.kayys.aljabr.core.model.ModelFormat;
import tech.kayys.aljabr.spi.model.ModelManifest;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Unified Model Hub — the primary entry point for loading pretrained models.
 * 
 * <p>
 * Supports loading from multiple sources including HuggingFace (hf:),
 * local directories (local:), and auto-detects schemes based on identifiers.
 * </p>
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * // Load from HuggingFace
 * ModelHub.loadInto(model, "Qwen/Qwen2.5-0.5B");
 *
 * // Load from local path
 * ModelHub.loadInto(model, "local:/path/to/model");
 * }</pre>
 * @author bhangun
 */
public final class ModelHub {

    private static final Logger LOG = LoggerFactory.getLogger(ModelHub.class);

    private ModelHub() {
    }

    /**
     * Downloads and loads weights into an existing model.
     *
     * @param model   target model
     * @param modelId model identifier (e.g., "bert-base-uncased",
     *                "local:/path/to/model")
     * @throws IOException if loading fails
     */
    public static void loadInto(NNModule model, String modelId) throws IOException {
        loadInto(model, modelId, HubConfig.DEFAULT);
    }

    /**
     * Downloads and loads weights into an existing model with custom config.
     */
    public static void loadInto(NNModule model, String modelId, HubConfig config) throws IOException {
        Map<String, GradTensor> weights = loadWeights(modelId, config);
        applyWeights(model, weights);
    }

    /**
     * Loads weights from a model identifier.
     *
     * @param modelId model identifier
     * @return map of parameter name to tensor
     */
    public static Map<String, GradTensor> loadWeights(String modelId) throws IOException {
        return loadWeights(modelId, HubConfig.DEFAULT);
    }

    /**
     * Loads weights from a model identifier with custom config.
     */
    public static Map<String, GradTensor> loadWeights(String modelId, HubConfig config) throws IOException {
        String scheme = detectScheme(modelId);
        ModelRepository repo = ModelHubFactory.getRepository(scheme, config);

        // Resolve model manifest synchronously (bridge Mutiny Uni)
        ModelManifest manifest = repo.findById(modelId, "sdk-hub")
                .await().atMost(java.time.Duration.ofMinutes(10));

        if (manifest == null) {
            throw new IOException("Could not resolve model: " + modelId);
        }

        // Determine format and path
        Path path = Path.of(manifest.path());
        if (scheme.equalsIgnoreCase("hf") || scheme.equalsIgnoreCase("kaggle")) {
            // Prefer GGUF, then SafeTensors, then TFLite/LiteRT
            if (manifest.artifacts().containsKey(ModelFormat.GGUF)) {
                String uri = manifest.artifacts().get(ModelFormat.GGUF).uri();
                path = Path.of(java.net.URI.create(uri));
            } else if (manifest.artifacts().containsKey(ModelFormat.SAFETENSORS)) {
                String uri = manifest.artifacts().get(ModelFormat.SAFETENSORS).uri();
                path = Path.of(java.net.URI.create(uri));
            } else if (manifest.artifacts().containsKey(ModelFormat.LITERT)) {
                String uri = manifest.artifacts().get(ModelFormat.LITERT).uri();
                path = Path.of(java.net.URI.create(uri));
            }
        }

        if (Files.isDirectory(path)) {
            return loadFromDirectory(path);
        } else {
            return loadFromFile(path);
        }
    }

    /**
     * Applies a map of weights to an NNModule with smart prefix matching.
     */
    public static void applyWeights(NNModule module, Map<String, GradTensor> weights) {
        Map<String, Parameter> params = module.namedParameters();
        int loaded = 0;

        for (var entry : weights.entrySet()) {
            Parameter param = params.get(entry.getKey());
            if (param == null) {
                // Try with prefix stripping (e.g., "model.layers.0.weight" → "layers.0.weight")
                for (var pEntry : params.entrySet()) {
                    if (entry.getKey().endsWith(pEntry.getKey())) {
                        param = pEntry.getValue();
                        break;
                    }
                }
            }
            if (param != null) {
                GradTensor weightTensor = entry.getValue();
                float[] paramData = param.data().data();
                float[] weightData = weightTensor.data();
                int len = Math.min(paramData.length, weightData.length);
                System.arraycopy(weightData, 0, paramData, 0, len);
                loaded++;
            }
        }

        LOG.info("Loaded {}/{} weight tensors into module", loaded, weights.size());
    }

    private static String detectScheme(String modelId) {
        if (modelId.startsWith("hf:") || modelId.startsWith("huggingface:"))
            return "hf";
        if (modelId.startsWith("kaggle:"))
            return "kaggle";
        if (modelId.startsWith("onnx:") || modelId.startsWith("local:"))
            return "local";

        // Explicit local paths (absolute, relative to current dir, or home-relative)
        if (modelId.startsWith("/") || modelId.startsWith("./") || modelId.startsWith("../")
                || modelId.startsWith("~/")) {
            return "local";
        }

        // Windows absolute paths
        if (modelId.length() > 2 && modelId.charAt(1) == ':'
                && (modelId.charAt(2) == '\\' || modelId.charAt(2) == '/')) {
            return "local";
        }

        // Default to HuggingFace for identifiers like "bert-base-uncased" or
        // "Qwen/Qwen2.5"
        // even if a local file accidentally exists in the CWD with that name.
        return "hf";
    }

    private static Map<String, GradTensor> loadFromDirectory(Path dir) throws IOException {
        Map<String, GradTensor> weights = new LinkedHashMap<>();
        try (var stream = Files.list(dir)) {
            stream.filter(p -> p.toString().endsWith(".safetensors") || p.toString().endsWith(".bin")
                    || p.toString().endsWith(".tflite") || p.toString().endsWith(".onnx"))
                    .sorted()
                    .forEach(file -> {
                        try {
                            weights.putAll(loadFromFile(file));
                        } catch (IOException e) {
                            LOG.error("Failed to load {}: {}", file, e.getMessage());
                        }
                    });
        }
        return weights;
    }

    private static Map<String, GradTensor> loadFromFile(Path file) throws IOException {
        String filename = file.getFileName().toString().toLowerCase();
        if (filename.endsWith(".safetensors") || filename.endsWith(".safetensor")) {
            return SafeTensorBridge.load(file);
        }
        if (filename.endsWith(".tflite")) {
            // TFLite uses LiteRT runtime for inference — return empty weights map
            // and let the LiteRT inference engine handle execution directly
            LOG.info("Detected TFLite model: {} — use LiteRT runtime for inference", file);
            return new LinkedHashMap<>();
        }
        if (filename.endsWith(".onnx")) {
            // ONNX uses ONNX Runtime for inference — return empty weights map
            // and let the ONNX inference engine handle execution directly
            LOG.info("Detected ONNX model: {} — use ONNX Runtime for inference", file);
            return new LinkedHashMap<>();
        }
        // Fallback for .bin if supported elsewhere, otherwise empty
        return new LinkedHashMap<>();
    }

    /** Legacy compatibility methods */
    public static void setCacheDir(Path dir) {
        /* Deprecated logic - use HubConfig */ }
}
