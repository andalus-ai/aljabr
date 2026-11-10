package tech.kayys.aljabr.converter.java.gguf;

import tech.kayys.aljabr.gguf.core.*;


import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

/**
 * GGUF file validator - checks spec compliance.
 * @author bhangun
 */
public final class GgufValidator {

    public static record ValidationResult(boolean valid, List<String> errors, List<String> warnings) {
    }

    public static ValidationResult validate(Path path) throws IOException {
        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();

        try (GgufReader reader = new GgufReader(path)) {
            GgufModel model = reader.read(); // Use in-memory for validation

            // Check magic
            // Already handled in reader

            // Check required metadata
            if (!model.getMeta("general.architecture").isPresent()) {
                errors.add("Missing required metadata: general.architecture");
            }

            String arch = model.architecture();

            // Check architecture-specific required keys
            if ("llama".equals(arch)) {
                checkRequiredKey(model, "llama.block_count", errors);
                checkRequiredKey(model, "llama.embedding_length", errors);
                checkRequiredKey(model, "llama.feed_forward_length", errors);
                checkRequiredKey(model, "llama.attention.head_count", errors);
                checkRequiredKey(model, "llama.context_length", errors);
            }

            // Check tensor offsets
            long dataStart = -1;
            Set<String> tensorNames = new HashSet<>();
            Map<Long, GGUFTensorInfo> offsetMap = new HashMap<>();

            for (GGUFTensorInfo ti : model.tensors()) {
                // Check duplicate names
                if (!tensorNames.add(ti.name())) {
                    errors.add("Duplicate tensor name: " + ti.name());
                }

                // Check offset alignment
                if (ti.dataOffset() % model.alignment() != 0) {
                    errors.add("Tensor " + ti.name() + " offset " + ti.dataOffset() +
                            " not aligned to " + model.alignment());
                }

                // Check overlapping tensors
                GGUFTensorInfo overlapping = offsetMap.get(ti.dataOffset());
                if (overlapping != null) {
                    errors.add("Tensor " + ti.name() + " overlaps with " +
                            overlapping.name() + " at offset " + ti.dataOffset());
                }
                offsetMap.put(ti.dataOffset(), ti);

                // Track data start
                if (dataStart < 0 || ti.dataOffset() < dataStart) {
                    dataStart = ti.dataOffset();
                }
            }

            // Check tokenizer presence
            if (!model.getMeta("tokenizer.ggml.model").isPresent()) {
                warnings.add("No tokenizer metadata found");
            } else {
                String tokenizerModel = model.getMeta("tokenizer.ggml.model")
                        .map(GgufMetaValue::asString)
                        .orElse("");
                if ("gpt2".equalsIgnoreCase(tokenizerModel)
                        && !model.getMeta("tokenizer.ggml.merges").isPresent()) {
                    errors.add("Missing required metadata for gpt2 tokenizer: tokenizer.ggml.merges");
                }
            }

            // Check shape consistency
            model.getMeta(arch + ".embedding_length").ifPresent(embLen -> {
                long expected = embLen.asUInt32();
                model.findTensor("token_embd.weight").ifPresent(tensor -> {
                    if (tensor.shape().length > 0 && tensor.shape()[0] != expected) {
                        warnings.add("token_embd.weight dim " + tensor.shape()[0] +
                                " doesn't match embedding_length " + expected);
                    }
                });
            });
        }

        return new ValidationResult(errors.isEmpty(), errors, warnings);
    }

    private static void checkRequiredKey(GgufModel model, String key, List<String> errors) {
        if (!model.getMeta(key).isPresent()) {
            errors.add("Missing required metadata: " + key);
        }
    }
}
