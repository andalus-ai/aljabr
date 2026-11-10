package tech.kayys.aljabr.spi.model;

import java.util.List;
import java.util.Map;

/**
 * Immutable descriptor for a model family.
 *
 * <p>
 * The descriptor is the primary source of truth for a
 * {@link ModelFamilyPlugin}.
 * It names the family, lists supported model types and architecture class
 * names,
 * declares capabilities, and carries freeform metadata for tooling and
 * validation.
 *
 * @param id            lowercase-underscore identifier (e.g. {@code "llama3"})
 * @param displayName   human-readable name (e.g. {@code "LLaMA 3"})
 * @param modelTypes    model-type identifiers supported by this family
 * @param architectures HuggingFace architecture class names (e.g.
 *                      {@code "LlamaForCausalLM"})
 * @param capabilities  set of capability flags declared by this family
 * @param metadata      freeform key-value metadata (e.g.
 *                      {@code "bundle_profile", "optional"})
 * @author bhangun
 */
public record ModelFamilyDescriptor(
        String id,
        String displayName,
        List<String> modelTypes,
        List<String> architectureClassNames,
        List<ModelFamilyCapability> capabilities,
        Map<String, String> metadata) {

    public ModelFamilyDescriptor {
        id = id == null ? "" : id.trim().toLowerCase();
        displayName = displayName == null ? "" : displayName.trim();
        modelTypes = modelTypes == null ? List.of()
                : modelTypes.stream().filter(java.util.Objects::nonNull).map(s -> s.trim().toLowerCase()).distinct()
                        .toList();
        architectureClassNames = architectureClassNames == null ? List.of()
                : architectureClassNames.stream().filter(java.util.Objects::nonNull).map(String::trim).distinct()
                        .toList();
        capabilities = capabilities == null ? List.of()
                : capabilities.stream().filter(java.util.Objects::nonNull).distinct().toList();
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
}
