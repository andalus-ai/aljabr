package tech.kayys.aljabr.spi.model;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * Validates a concrete model-weight directory (a "fixture") against the claims
 * made by a {@link ModelFamilyPlugin}.
 *
 * <p>
 * A fixture is a directory that contains at minimum a {@code config.json} file
 * (in HuggingFace format). This validator cross-checks the fixture's declared
 * {@code model_type} and {@code architectures} against the plugin's descriptor,
 * adapter list, and tokenizer descriptors.
 * @author bhangun
 */
public final class ModelFamilyFixtureValidator {

        private static final ObjectMapper MAPPER = new ObjectMapper();

        private ModelFamilyFixtureValidator() {
        }

        /**
         * Validates the fixture directory against the plugin's contract.
         *
         * @param plugin     the plugin whose contract the fixture should satisfy
         * @param fixtureDir path to a directory containing {@code config.json}
         * @return a (possibly empty) list of contract violations
         * @throws IOException if {@code config.json} cannot be read
         */
        public static List<ModelFamilyContractViolation> validate(
                        ModelFamilyPlugin plugin, Path fixtureDir) throws IOException {

                List<ModelFamilyContractViolation> v = new ArrayList<>();
                ModelFamilyDescriptor desc = plugin.descriptor();

                Path configPath = fixtureDir.resolve("config.json");
                if (!Files.exists(configPath)) {
                        v.add(ModelFamilyContractViolation.of("fixture_config_missing",
                                        "config.json not found in fixture directory: " + fixtureDir));
                        return List.copyOf(v);
                }

                JsonNode config = MAPPER.readTree(configPath.toFile());

                // ── Model type check ──────────────────────────────────────────
                String fixtureModelType = config.path("model_type").asText(null);
                if (fixtureModelType != null && !desc.modelTypes().contains(fixtureModelType)) {
                        v.add(ModelFamilyContractViolation.of("fixture_model_type_unclaimed",
                                        "Fixture model_type '" + fixtureModelType
                                                        + "' is not declared in descriptor modelTypes "
                                                        + desc.modelTypes()));
                }

                // ── Architecture class check ──────────────────────────────────
                Set<String> allAdapterArchClasses = plugin.architectureAdapters().stream()
                                .flatMap(a -> a.supportedArchClassNames().stream())
                                .collect(Collectors.toSet());

                JsonNode archsNode = config.path("architectures");
                if (archsNode.isArray()) {
                        for (JsonNode archNode : archsNode) {
                                String arch = archNode.asText();
                                if (!allAdapterArchClasses.contains(arch)
                                                && !desc.architectureClassNames().contains(arch)) {
                                        v.add(ModelFamilyContractViolation.of("fixture_architecture_unclaimed",
                                                        "Fixture architecture '" + arch
                                                                        + "' is not declared in any adapter or descriptor architectures"));
                                }
                        }
                }

                // ── Tokenizer files check ─────────────────────────────────────
                boolean anyTokenizerMatched = plugin.tokenizerDescriptors().stream()
                                .anyMatch(td -> td.firstExistingFileGroup(fixtureDir).isPresent());
                if (!plugin.tokenizerDescriptors().isEmpty() && !anyTokenizerMatched) {
                        v.add(ModelFamilyContractViolation.of("fixture_tokenizer_files_unmatched",
                                        "None of the tokenizer descriptors found matching files in " + fixtureDir));
                }

                // ── Adapter match check ───────────────────────────────────────
                List<String> fixtureArchs = archsNode.isArray()
                                ? StreamSupport.stream(archsNode.spliterator(), false)
                                                .map(JsonNode::asText).toList()
                                : List.of();
                boolean anyAdapterMatched = plugin.architectureAdapters().stream()
                                .anyMatch(adapter -> {
                                        boolean typeMatch = fixtureModelType != null
                                                        && adapter.supportedModelTypes().contains(fixtureModelType);
                                        boolean archMatch = !fixtureArchs.isEmpty()
                                                        && adapter.supportedArchClassNames().stream()
                                                                        .anyMatch(fixtureArchs::contains);
                                        return typeMatch && archMatch;
                                });

                if (!plugin.architectureAdapters().isEmpty() && !anyAdapterMatched) {
                        v.add(ModelFamilyContractViolation.of("fixture_architecture_adapter_unmatched",
                                        "No architecture adapter matches fixture model_type='" + fixtureModelType
                                                        + "' and architectures=" + fixtureArchs));
                }

                return List.copyOf(v);
        }
}
