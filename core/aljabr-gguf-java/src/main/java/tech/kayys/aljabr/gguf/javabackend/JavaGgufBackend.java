package tech.kayys.aljabr.gguf.javabackend;

import tech.kayys.aljabr.gguf.api.GgufBackend;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFParser;
import tech.kayys.aljabr.gguf.loader.GGUFReader;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;
import tech.kayys.aljabr.gguf.loader.sampler.Sampler;
import tech.kayys.aljabr.gguf.runtime.GgufBudget;
import tech.kayys.aljabr.gguf.runtime.GgufRuntimeProbe;
import tech.kayys.aljabr.gguf.runtime.GgufRuntimeProfile;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps;
import tech.kayys.aljabr.gguf.tokenizer.GGUFTokenizer;
import tech.kayys.aljabr.spi.Message;
import tech.kayys.aljabr.spi.exception.InferenceException;
import tech.kayys.aljabr.spi.inference.InferenceRequest;
import tech.kayys.aljabr.spi.inference.InferenceResponse;
import tech.kayys.aljabr.spi.tokenizer.ChatTemplateFormatter;
import tech.kayys.aljabr.spi.tokenizer.DecodeOptions;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;

import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Pure-Java GGUF engine.
 *
 * <p><b>Status: real generation loop, functionally unverified.</b> Runs
 * tokenizer -> decoder -> sampler -> KV cache using {@link GgufDecodeStep}
 * and {@link GgufModelConfig}. It has NOT been run against a real model or
 * compared to llama.cpp output —
 * {@link JavaGgufBackendProvider#ARCHITECTURES_READY_FOR_GENERATION}
 * stays empty until that happens. An explicit {@code gguf.backend=java}
 * request still routes here for testing in the meantime.</p>
 *
 * <p><b>Response shape parity:</b> builds a real {@link InferenceResponse} —
 * {@code requestId}/{@code model}/{@code sessionId} from the request when
 * present, {@code finishReason} computed correctly (STOP when the loop
 * broke on an EOS token, LENGTH when it hit {@code maxTokens} or the
 * context limit first) — closing the response-shape gap against the
 * llama.cpp backend.</p>
 */
final class JavaGgufBackend implements GgufBackend {
    private static boolean isVerbose() {
        return Boolean.getBoolean("aljabr.metal.verbose")
                || Boolean.getBoolean("aqli.verbose")
                || System.getenv("ALJABR_METAL_VERBOSE") != null
                || System.getenv("ALJABR_DEBUG") != null
                || System.getenv("AQLI_DEBUG") != null;
    }

    private final GGUFModel model;
    private final GgufRuntimeProfile profile;
    private final GgufRuntimeProbe.PreparedMatrixCacheDecision preparedCacheDecision;
    private final GgufModelConfig cfg;
    private final GgufDecodeStep decodeStep;
    private final GGUFTokenizer tokenizer;
    private final KVCache sharedCache;

    private static final class PromptCacheEntry {
        final int[] tokens;
        final float[][] k;
        final float[][] v;
        final float[] logits;

        PromptCacheEntry(int[] tokens, float[][] k, float[][] v, float[] logits) {
            this.tokens = tokens;
            this.k = k;
            this.v = v;
            this.logits = logits;
        }
    }

    private final Object promptCacheLock = new Object();
    private PromptCacheEntry cachedPrompt = null;

    /**
     * Tracks whether the native Metal KV buffers already hold the correct K/V
     * data for the first {@link #metalKvPrefixTokenCount} tokens of the cached
     * prompt. When true, a cache-hit path can skip {@code syncMetalCache} and
     * save the 170–270 ms spent re-uploading 75 MB of KV data to Metal shared
     * memory.
     *
     * <p>Invariant: set to {@code true} only right after a full prefill (the
     * Metal buffers were just written by the prefill pass) or right after a
     * successful cache-hit decode (the prefix portion 0..T-1 was never
     * overwritten; decode only writes at positions ≥ T). Reset to {@code false}
     * whenever Metal state is uncertain (init failure, model reload, etc.).
     */
    private volatile boolean metalKvPrefixValid = false;
    private volatile int     metalKvPrefixTokenCount = 0;

    private KVCache createKVCache() {
        if (cfg.nKVHeadsPerLayer() != null && cfg.headDimPerLayer() != null) {
            return new KVCache(cfg.nLayers(), cfg.contextLength(), cfg.nKVHeadsPerLayer(), cfg.headDimPerLayer());
        } else {
            return new KVCache(cfg.nLayers(), cfg.contextLength(), cfg.nKVHeads(), cfg.headDim());
        }
    }

    JavaGgufBackend(Path modelPath) throws Exception {
        long startNanos = System.nanoTime();
        this.model = loadModel(modelPath);
        long loadMillis = Duration.ofNanos(System.nanoTime() - startNanos).toMillis();
        this.profile = GgufRuntimeProfile.fromModel(model, Files.size(modelPath), loadMillis);
        this.preparedCacheDecision = prepareDecoderMatrixCaches(model);
        this.cfg = GgufModelConfig.from(model);
        this.decodeStep = new GgufDecodeStep(model, cfg);
        this.tokenizer = new GGUFTokenizer(model);
        this.sharedCache = createKVCache();
    }

    @Override
    public String name() {
        return "java";
    }

    @Override
    public Map<String, Object> metadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("backend", name());
        metadata.put("javaStatus", profile.javaStatus());
        metadata.put("architecture", profile.architecture());
        metadata.put("ggufVersion", profile.ggufVersion());
        metadata.put("tensorCount", profile.tensorCount());
        metadata.put("decoderTensorRatio", profile.decoderTensorRatio());
        metadata.put("missingDecoderTensorCount", profile.missingDecoderTensorCount());
        metadata.put("malformedDecoderTensorCount", profile.malformedDecoderTensorCount());
        metadata.put("missingDecoderTensorExamples", profile.missingDecoderTensorExamples());
        metadata.put("malformedDecoderTensorExamples", profile.malformedDecoderTensorExamples());
        metadata.put("knownTensorTypeRatio", profile.knownTensorTypeRatio());
        metadata.put("preparedMatrixCachePlan", preparedCacheDecision.plan().compactSummary());
        metadata.put("preparedMatrixCache", preparedCacheDecision.compactSummary());
        metadata.put("preparedMatrixCacheMode", preparedCacheDecision.mode());
        metadata.put("preparedMatrixCacheEstimatedBytes", preparedCacheDecision.plan().estimatedPreparedBytes());
        metadata.put("preparedMatrixCacheBytes", preparedCacheDecision.stats().cacheBytes());
        metadata.put("preparedMatrixCacheEntries", preparedCacheDecision.stats().cacheEntries());
        metadata.put("loaderReady", profile.knownTensorTypeRatio() > 0.0d);
        metadata.put("decoderTensorsReady", profile.decoderTensorSetComplete());
        metadata.put("rowDotReady", profile.rowDotPrimitivesReady());
        metadata.put("generationReady", false);
        metadata.put("rolloutStage", GgufRolloutStage.forArchitecture(
                profile.architecture(), JavaGgufBackendProvider.ARCHITECTURES_READY_FOR_GENERATION).name());
        metadata.put("chatTemplateSupported", ChatTemplateFormatter.supportsModelType(cfg.architecture()));
        if (profile.modelConfig() != null) {
            metadata.put("modelType", profile.modelConfig().modelType());
            metadata.put("layers", profile.modelConfig().numHiddenLayers());
            metadata.put("hiddenSize", profile.modelConfig().hiddenSize());
            metadata.put("attentionHeads", profile.modelConfig().numAttentionHeads());
            metadata.put("kvHeads", profile.modelConfig().resolvedNumKvHeads());
            metadata.put("headDim", profile.modelConfig().resolvedHeadDim());
            metadata.put("contextLength", profile.modelConfig().maxPositionEmbeddings());
            metadata.put("vocabSize", profile.modelConfig().vocabSize());
        }
        return Map.copyOf(metadata);
    }

    @Override
    public InferenceResponse execute(InferenceRequest request) {
        try {
            String prompt = extractPrompt(request);
            if (prompt == null || prompt.isBlank()) {
                throw new InferenceException("No prompt or inference request messages provided");
            }

            SamplingParams sampling = SamplingParams.from(request);

            EncodeOptions encodeOptions = EncodeOptions.builder()
                    .addBos(tokenizer.shouldAddBos())
                    .addEos(tokenizer.shouldAddEos())
                    .build();
            long[] promptTokens = tokenizer.encode(prompt, encodeOptions);
            if (promptTokens.length == 0) {
                throw new InferenceException("Prompt tokenized to zero tokens.");
            }
            if (promptTokens.length >= cfg.contextLength()) {
                throw new InferenceException("Prompt (" + promptTokens.length
                        + " tokens) does not fit in context length " + cfg.contextLength());
            }

            long startNanos = System.nanoTime();
            KVCache cache;
            Sampler sampler = new Sampler(
                    sampling.temperature(), sampling.topK(), sampling.topP(), sampling.repeatPenalty());

            List<Integer> history = new ArrayList<>(); // prompt + generated, for repetition penalty
            List<Integer> generated = new ArrayList<>();

            int pos = 0;
            float[] logits = null;
            long t0 = System.currentTimeMillis();
            long prefillMs = 0;
            if (isVerbose()) {
                System.err.printf("[java-engine] Prefilling %d prompt tokens...%n", promptTokens.length);
            }
            if (promptTokens.length > 0) {
                int[] tokenIds = new int[promptTokens.length];
                for (int pIdx = 0; pIdx < promptTokens.length; pIdx++) {
                    tokenIds[pIdx] = (int) promptTokens[pIdx];
                    history.add((int) promptTokens[pIdx]);
                }

                PromptCacheEntry hit = null;
                synchronized (promptCacheLock) {
                    if (cachedPrompt != null && Arrays.equals(cachedPrompt.tokens, tokenIds)) {
                        hit = cachedPrompt;
                    }
                }

                if (hit != null) {
                    cache = sharedCache;
                    cache.resetFast();
                    for (int l = 0; l < cfg.nLayers(); l++) {
                        int count = tokenIds.length * cache.headStride(l);
                        System.arraycopy(hit.k[l], 0, cache.kLayer(l), 0, count);
                        System.arraycopy(hit.v[l], 0, cache.vLayer(l), 0, count);
                    }
                    cache.advance(tokenIds.length);
                    // If the Metal-side KV buffers already hold the correct prefix
                    // data for this exact prompt (positions 0..T-1 are valid from the
                    // previous prefill/decode and were never overwritten), skip the
                    // 170-270 ms syncMetalCache round-trip. The decode step only writes
                    // at positions >= seqLen, so the prefix is always preserved.
                    boolean skipSync = metalKvPrefixValid
                            && metalKvPrefixTokenCount == tokenIds.length;
                    if (!skipSync) {
                        decodeStep.syncMetalCache(tokenIds.length, cache);
                        metalKvPrefixValid = true;
                        metalKvPrefixTokenCount = tokenIds.length;
                    }
                    logits = hit.logits.clone();
                    pos = tokenIds.length;
                    prefillMs = Math.max(1, System.currentTimeMillis() - t0);
                    if (isVerbose()) {
                        System.err.printf("[java-engine] Prompt prefix cache HIT: %d tokens restored in %d ms (Metal sync: %s)%n",
                                tokenIds.length, prefillMs, skipSync ? "SKIPPED" : "done");
                    }
                } else {
                    metalKvPrefixValid = false; // Metal state unknown until prefill completes
                    cache = sharedCache;
                    cache.resetFast();
                    logits = decodeStep.prefill(tokenIds, cache);
                    pos = tokenIds.length;
                    prefillMs = Math.max(1, System.currentTimeMillis() - t0);
                    // Prefill just wrote positions 0..T-1 into Metal KV buffers.
                    metalKvPrefixValid = true;
                    metalKvPrefixTokenCount = tokenIds.length;

                    try {
                        int nLayers = cfg.nLayers();
                        float[][] snapK = new float[nLayers][];
                        float[][] snapV = new float[nLayers][];
                        for (int l = 0; l < nLayers; l++) {
                            int count = tokenIds.length * cache.headStride(l);
                            snapK[l] = new float[count];
                            snapV[l] = new float[count];
                            System.arraycopy(cache.kLayer(l), 0, snapK[l], 0, count);
                            System.arraycopy(cache.vLayer(l), 0, snapV[l], 0, count);
                        }
                        synchronized (promptCacheLock) {
                            cachedPrompt = new PromptCacheEntry(tokenIds.clone(), snapK, snapV, logits.clone());
                        }
                    } catch (Throwable ignored) {
                    }

                    if (isVerbose()) {
                        System.err.printf("[java-engine] Prompt prefill: %d tokens in %d ms (%.1f tok/s)%n",
                                promptTokens.length, prefillMs, (promptTokens.length * 1000.0) / prefillMs);
                        DecodeOptions dbgOpts = DecodeOptions.builder().build();
                        for (int pIdx = 0; pIdx < tokenIds.length; pIdx++) {
                            System.err.printf("  [PROMPT tok %d] %d -> '%s'%n", pIdx, tokenIds[pIdx], tokenizer.decode(new long[]{tokenIds[pIdx]}, dbgOpts));
                        }
                    }
                }
            } else {
                cache = sharedCache;
                cache.resetFast();
            }

            InferenceResponse.FinishReason finishReason = InferenceResponse.FinishReason.LENGTH;
            if (isVerbose()) {
                System.err.println("[java-engine] Generating tokens...");
            }
            long genStart = System.currentTimeMillis();
            boolean isGreedy = (sampling.temperature() <= 0.0f);
            int nextToken = -1;
            boolean hasNextToken = false;

            for (int i = 0; i < sampling.maxTokens() && pos < cfg.contextLength(); i++) {
                if (!hasNextToken) {
                    nextToken = sampler.sample(logits, history);
                } else {
                    hasNextToken = false;
                }
                if (isVerbose() && i < 10) {
                    System.err.printf("  [GEN step %d] token=%d ('%s')%n", i, nextToken, tokenizer.decode(new long[]{nextToken}, DecodeOptions.builder().build()));
                }
                if (tokenizer.isEosToken(nextToken)) {
                    finishReason = InferenceResponse.FinishReason.STOP;
                    break;
                }
                generated.add(nextToken);
                history.add(nextToken);

                if (i + 1 >= sampling.maxTokens() || pos + 1 >= cfg.contextLength()) {
                    pos++;
                    break;
                }

                // The native argmax path does not apply Sampler's repetition penalty.
                // Keep greedy requests with a configured penalty on the regular sampler path.
                if (canUseMetalGreedy(isGreedy, sampling.repeatPenalty(), decodeStep.canUseMetalGreedy())) {
                    int greedyNext = decodeStep.forwardGreedy(nextToken, pos, cache);
                    if (greedyNext >= 0) {
                        nextToken = greedyNext;
                        hasNextToken = true;
                        pos++;
                        continue;
                    }
                }

                logits = decodeStep.forward(nextToken, pos, cache, true);
                pos++;
            }
            long genElapsed = System.currentTimeMillis() - genStart;
            if (isVerbose() && !generated.isEmpty()) {
                System.err.printf("[java-engine] Generated %d tokens in %d ms (%.2f tok/s)%n",
                        generated.size(), genElapsed, (generated.size() * 1000.0) / Math.max(1, genElapsed));
            }

            long[] generatedTokens = generated.stream().mapToLong(Integer::longValue).toArray();
            DecodeOptions decodeOptions = DecodeOptions.builder().build();
            String text = tokenizer.decode(generatedTokens, decodeOptions);
            long durationMs = Duration.ofNanos(System.nanoTime() - startNanos).toMillis();

            Map<String, Object> respMetadata = new LinkedHashMap<>();
            respMetadata.put("backend", "java-native");
            respMetadata.put("promptTokens", promptTokens.length);
            respMetadata.put("prefillMs", prefillMs);
            respMetadata.put("generatedTokens", generated.size());
            respMetadata.put("decodedTokens", generated.size());
            respMetadata.put("decodeMs", genElapsed);
            respMetadata.put("nativeMetrics", String.format(Locale.ROOT,
                    "backend=java-native (Metal),promptTokens=%d,prefill=%.2f,decodedTokens=%d,decode=%.2f,generatedTokens=%d",
                    promptTokens.length, (double) prefillMs, generated.size(), (double) genElapsed, generated.size()));

            return new InferenceResponse(
                    request != null && request.getRequestId() != null ? request.getRequestId() : java.util.UUID.randomUUID().toString(),
                    text,
                    request != null ? request.getModel() : null,
                    0, // tokensUsed: 0 triggers the constructor's own inputTokens+outputTokens fallback
                    promptTokens.length,
                    generated.size(),
                    durationMs,
                    null, // timestamp: constructor defaults to Instant.now()
                    respMetadata,
                    null, // toolCalls: Java engine doesn't support tool calls; constructor defaults to empty list
                    finishReason,
                    request != null && request.getSessionId().isPresent() ? request.getSessionId().get() : null);
        } catch (InferenceException ie) {
            metalKvPrefixValid = false; // Metal state unknown after error
            throw ie;
        } catch (Exception e) {
            metalKvPrefixValid = false; // Metal state unknown after error
            e.printStackTrace();
            throw new InferenceException("Java GGUF generation failed: " + e.getClass().getSimpleName() + ": " + e.getMessage(), e);
        }
    }

    private String extractPrompt(InferenceRequest req) {
        if (req == null) {
            return "";
        }
        if (req.getParameters() != null && req.getParameters().containsKey("prompt")) {
            Object p = req.getParameters().get("prompt");
            if (p != null && !p.toString().isBlank()) {
                return p.toString();
            }
        }

        String prompt = req.getPrompt();
        if (prompt != null && !prompt.isBlank()) {
            return prompt;
        }

        List<Message> messages = req.getMessages();
        if (messages != null && !messages.isEmpty() && ChatTemplateFormatter.supportsModelType(cfg.architecture())) {
            return ChatTemplateFormatter.format(messages, cfg.architecture());
        }

        return "";
    }

    @Override
    public void close() {
        GgufTensorOps.clearPreparedMatrixCaches(model);
        model.close();
    }

    private static GgufRuntimeProbe.PreparedMatrixCacheDecision prepareDecoderMatrixCaches(GGUFModel model) {
        int explicitMinRows = Math.max(0, Integer.getInteger("aljabr.gguf.java.prepare_min_rows", 0));
        if (explicitMinRows > 0) {
            return GgufRuntimeProbe.prepareDecoderMatrixCaches(
                    model,
                    GgufRuntimeProbe.selectDecoderPreparedMatrixCache(model, explicitMinRows, false, 1, 0L));
        }

        int autoMinRows = Math.max(1, Integer.getInteger("aljabr.gguf.java.auto_prepare_min_rows", 32));
        long budgetBytes = GgufBudget.byteSizeProperty(
                "aljabr.gguf.java.auto_prepare_budget_bytes", GgufBudget.defaultAutoPrepareBytes());
        return GgufRuntimeProbe.prepareDecoderMatrixCaches(
                model,
                GgufRuntimeProbe.selectDecoderPreparedMatrixCache(
                        model,
                        0,
                        Boolean.parseBoolean(System.getProperty("aljabr.gguf.java.auto_prepare", "false")),
                        autoMinRows,
                        budgetBytes));
    }

    private static GGUFModel loadModel(Path modelPath) throws Exception {
        Arena arena = Arena.ofShared();
        try (GGUFReader reader = new GGUFReader(modelPath, arena)) {
            return new GGUFParser().parse(reader.segment(), arena);
        } catch (Exception exception) {
            arena.close();
            throw exception;
        } catch (Error error) {
            arena.close();
            throw error;
        }
    }

    static boolean canUseMetalGreedy(boolean isGreedy, float repeatPenalty, boolean metalDecodeAvailable) {
        return isGreedy && (repeatPenalty <= 1.0001f || Float.isNaN(repeatPenalty)) && metalDecodeAvailable;
    }

    private record SamplingParams(float temperature, int topK, float topP, float repeatPenalty, int maxTokens) {
        static SamplingParams from(InferenceRequest req) {
            if (req == null) {
                return new SamplingParams(0.2f, 40, 0.9f, 1.1f, 256);
            }
            float temp = (float) req.getTemperature();
            float rep = (float) req.getRepeatPenalty();
            if (temp <= 0.0f && req.getParameters() != null
                    && !req.getParameters().containsKey("repeat_penalty")
                    && !req.getParameters().containsKey("repetition_penalty")) {
                rep = 1.0f;
            }
            return new SamplingParams(
                    temp,
                    req.getTopK(),
                    (float) req.getTopP(),
                    rep,
                    req.getMaxTokens());
        }
    }
}
