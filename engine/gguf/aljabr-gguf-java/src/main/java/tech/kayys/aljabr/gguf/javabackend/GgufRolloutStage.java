package tech.kayys.aljabr.gguf.javabackend;

/**
 * Names the three-stage rollout for the pure-Java GGUF engine, agreed as
 * the actual plan (not something this codebase invented on its own):
 *
 * <ol>
 *   <li><b>EXPERIMENTAL</b> — llama.cpp is the default runner.
 *       {@link JavaNativeGgufBackendProvider#ARCHITECTURES_READY_FOR_GENERATION}
 *       is empty for this architecture, so AUTO selection never picks Java;
 *       it's reachable only via an explicit {@code gguf.backend=java}
 *       request, for iterating and testing in an IDE against real models.</li>
 *   <li><b>GRADUATED</b> — once an architecture has been verified (real
 *       model, golden-output comparison against llama.cpp — see the
 *       production roadmap's Tier 1), add it to
 *       {@code ARCHITECTURES_READY_FOR_GENERATION}. No other code changes:
 *       {@link JavaNativeGgufBackendProvider#priority()} (10) already
 *       outranks {@code LlamaCppGgufBackendProvider} (-10), so AUTO now
 *       prefers Java for that architecture specifically, with llama.cpp
 *       remaining the automatic fallback for every architecture that
 *       hasn't graduated yet. This is a per-architecture transition, not a
 *       global switch — Llama can graduate while Gemma is still
 *       experimental, or not yet structurally supported at all.</li>
 *   <li><b>LLAMA_CPP_DETACHED</b> — not a runtime state; there's nothing to
 *       check at request time, because it's a build-time decision: exclude
 *       the {@code aljabr-gguf-llamacpp} jar from the
 *       assembly. {@code GgufRunnerPlugin} already handles zero, one, or
 *       two available providers without code changes — that's the entire
 *       point of the ServiceLoader split. There's deliberately no enum
 *       value for this stage in code; it isn't something a running process
 *       can observe about itself.</li>
 * </ol>
 */
enum GgufRolloutStage {
    EXPERIMENTAL,
    GRADUATED;

    static GgufRolloutStage forArchitecture(String architecture, java.util.Set<String> readyArchitectures) {
        if (architecture != null && readyArchitectures.contains(architecture.toLowerCase(java.util.Locale.ROOT))) {
            return GRADUATED;
        }
        return EXPERIMENTAL;
    }
}
