package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.api.GgufBackend;
import tech.kayys.aljabr.gguf.api.GgufBackendCapability;
import tech.kayys.aljabr.gguf.api.GgufBackendProvider;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFParser;
import tech.kayys.aljabr.gguf.loader.GGUFReader;
import tech.kayys.aljabr.gguf.runtime.GgufRuntimeProfile;

import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * ServiceLoader registration for the pure-Java GGUF engine.
 *
 * <p>{@link #isAvailable()} always returns {@code true} (no native dependency).
 * {@link #probe} inspects GGUF header metadata to determine whether the Java
 * engine can generate for a specific model architecture; as of the initial
 * migration the answer is always "not yet" because
 * {@link #ARCHITECTURES_READY_FOR_GENERATION} is empty. The architecture is
 * reported in the capability's {@code diagnostics} map so operators can see
 * the readiness state in {@code ModelHandle} metadata even before generation
 * is enabled.</p>
 *
 * <p>Once the decoder loop (tokenizer → transformer → sampler → KV cache) has
 * been verified against golden output from llama.cpp for an architecture,
 * add that architecture's lowercase string to
 * {@link #ARCHITECTURES_READY_FOR_GENERATION}. That is the only change needed
 * to flip AUTO selection to prefer Java for that architecture — the selection
 * loop in {@link tech.kayys.aljabr.gguf.api.GgufInferenceEngine} picks the
 * highest-priority provider that reports {@code generationReady}, so Java
 * (priority 10) automatically wins over llama.cpp (priority -10) the moment
 * it claims readiness.</p>
 */
public final class JavaGgufBackendProvider implements GgufBackendProvider {

    static final Set<String> ARCHITECTURES_READY_FOR_GENERATION = Set.of();

    @Override
    public String id() {
        return "java";
    }

    @Override
    public Set<String> aliases() {
        return Set.of("javanative", "java-native", "pure-java");
    }

    @Override
    public boolean isAvailable() {
        return true; // pure Java, no native dependency to check
    }

    @Override
    public GgufBackendCapability probe(Path modelPath) throws Exception {
        Arena arena = Arena.ofConfined();
        try (GGUFReader reader = new GGUFReader(modelPath, arena)) {
            GGUFModel model = new GGUFParser().parse(reader.segment(), arena);
            try {
                GgufRuntimeProfile profile = GgufRuntimeProfile.fromModel(model, Files.size(modelPath), 0L);

                Map<String, Object> diagnostics = new LinkedHashMap<>();
                diagnostics.put("knownTensorTypeRatio", profile.knownTensorTypeRatio());
                diagnostics.put("decoderTensorsReady", profile.decoderTensorSetComplete());
                diagnostics.put("rowDotReady", profile.rowDotPrimitivesReady());
                diagnostics.put("missingDecoderTensorCount", profile.missingDecoderTensorCount());

                String architecture = profile.architecture();
                boolean ready = architecture != null
                        && ARCHITECTURES_READY_FOR_GENERATION.contains(architecture.toLowerCase(java.util.Locale.ROOT))
                        && profile.decoderTensorSetComplete()
                        && profile.rowDotPrimitivesReady();

                return ready
                        ? GgufBackendCapability.ready(architecture, Set.of(), diagnostics)
                        : GgufBackendCapability.notReady(architecture, Set.of(), profile.javaStatus(), diagnostics);
            } finally {
                model.close();
            }
        }
    }

    @Override
    public GgufBackend create(Path modelPath, Map<String, Object> config) throws Exception {
        return new JavaGgufBackend(modelPath);
    }

    @Override
    public int priority() {
        // Outranks llama.cpp once ready for a given architecture: AUTO tries
        // higher-priority providers first, so pure Java wins whenever it can.
        return 10;
    }
}
