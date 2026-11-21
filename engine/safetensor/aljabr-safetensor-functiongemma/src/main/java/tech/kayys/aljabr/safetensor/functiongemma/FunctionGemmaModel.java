package tech.kayys.aljabr.safetensor.functiongemma;

import java.util.Objects;
import tech.kayys.aljabr.models.functiongemma.FunctionGemmaTokenizer;
import tech.kayys.aljabr.models.functiongemma.FunctionGemmaToolCallParser;
import tech.kayys.aljabr.spi.Message;
import tech.kayys.aljabr.models.gemma3.Gemma3TextBackbone;
import tech.kayys.aljabr.spi.model.ModelConfig;
import tech.kayys.aljabr.spi.tokenizer.DecodeOptions;
import tech.kayys.aljabr.spi.tokenizer.EncodeOptions;
import tech.kayys.aljabr.spi.tokenizer.ChatTemplateFormatter;

import java.util.ArrayList;
import java.util.List;

/**
 * Executable FunctionGemma checkpoint backed by Aljabr's shared Gemma 3
 * text decoder reference implementation.
 *
 * <p>Weights are detached from the SafeTensors mapping during load, so this
 * model does not retain a file mapping and requires no close operation.
 * Runtime/tool dispatch is intentionally outside this model adapter.</p>
 */
public final class FunctionGemmaModel {
    private final ModelConfig config;
    private final FunctionGemmaTokenizer tokenizer;
    private final Gemma3TextBackbone backbone;

    FunctionGemmaModel(
            ModelConfig config,
            FunctionGemmaTokenizer tokenizer,
            Gemma3TextBackbone backbone) {
        this.config = Objects.requireNonNull(config, "config");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.backbone = Objects.requireNonNull(backbone, "backbone");
    }

    /** Returns logits for each token in one unpadded input sequence. */
    public float[][] forward(int[] inputIds) {
        return backbone.forward(inputIds);
    }

    /**
     * Accepts tokenizer-native token IDs and rejects values that cannot be
     * represented by the reference backbone's integer vocabulary indices.
     */
    public float[][] forward(long[] inputIds) {
        Objects.requireNonNull(inputIds, "inputIds");
        int[] converted = new int[inputIds.length];
        for (int index = 0; index < inputIds.length; index++) {
            converted[index] = Math.toIntExact(inputIds[index]);
        }
        return backbone.forward(converted);
    }


    /**
     * Formats structured conversation messages with the registered
     * FunctionGemma chat template, then performs greedy generation.
     */
    public GenerationResult generate(List<Message> messages, int maxNewTokens) {
        Objects.requireNonNull(messages, "messages");
        String prompt = ChatTemplateFormatter.format(messages, "functiongemma");
        return generate(prompt, maxNewTokens);
    }

    /**
     * Extracts native FunctionGemma calls from a generation result. Parsed
     * arguments remain untrusted data; execution and authorization belong to
     * the application/runtime.
     */
    public List<FunctionGemmaToolCallParser.FunctionCall> parseFunctionCalls(GenerationResult result) {
        Objects.requireNonNull(result, "result");
        return new FunctionGemmaToolCallParser().parse(result.text());
    }

    /**
     * Greedily generates from a caller-formatted prompt. The prompt is tokenized
     * by the checkpoint tokenizer; callers that need the FunctionGemma chat
     * template should format it before calling this method.
     *
     * <p>Generation uses the shared Gemma 3 incremental KV-cache decoder rather
     * than recomputing the complete prompt for every output token. The
     * FunctionGemma {@code <start_function_response>} marker is a stop
     * condition so the consuming runtime can execute the preceding function
     * call and feed its response back into a subsequent turn.</p>
     */
    public GenerationResult generate(String prompt, int maxNewTokens) {
        Objects.requireNonNull(prompt, "prompt");
        int[] promptIds = toIntArray(tokenizer.encode(prompt, EncodeOptions.defaultOptions()));
        return generate(promptIds, maxNewTokens);
    }

    /**
     * Greedy generation from token IDs. The returned token sequence includes
     * the stop token when generation ends on EOS or a FunctionGemma function
     * response marker.
     */
    public GenerationResult generate(int[] promptTokenIds, int maxNewTokens) {
        Objects.requireNonNull(promptTokenIds, "promptTokenIds");
        if (promptTokenIds.length == 0) {
            throw new IllegalArgumentException("promptTokenIds must not be empty");
        }
        if (maxNewTokens < 0) {
            throw new IllegalArgumentException("maxNewTokens must not be negative");
        }
        if ((long) promptTokenIds.length + maxNewTokens > config.maxPositionEmbeddings()) {
            throw new IllegalArgumentException("prompt and requested output exceed maxPositionEmbeddings");
        }

        List<Integer> generated = new ArrayList<>(maxNewTokens);
        GenerationStopReason stopReason = GenerationStopReason.MAX_NEW_TOKENS;
        int eosTokenId = tokenizer.eosTokenId();
        int functionResponseTokenId = tokenizer.tokenId(FunctionGemmaTokenizer.START_FUNCTION_RESPONSE);

        try (Gemma3TextBackbone.DecodeSession session = backbone.newDecodeSession()) {
            float[][] promptLogits = session.prefill(promptTokenIds);
            float[] nextLogits = promptLogits[promptLogits.length - 1];
            for (int index = 0; index < maxNewTokens; index++) {
                int nextToken = greedyToken(nextLogits);
                generated.add(nextToken);

                if (nextToken == functionResponseTokenId) {
                    stopReason = GenerationStopReason.FUNCTION_RESPONSE;
                    break;
                }
                if (eosTokenId >= 0 && nextToken == eosTokenId) {
                    stopReason = GenerationStopReason.END_OF_SEQUENCE;
                    break;
                }
                if (index + 1 < maxNewTokens) {
                    nextLogits = session.decode(nextToken);
                }
            }
        }

        int[] tokenIds = generated.stream().mapToInt(Integer::intValue).toArray();
        String text = tokenizer.decode(toLongArray(tokenIds),
                DecodeOptions.defaultOptions().skipSpecialTokens(false));
        return new GenerationResult(tokenIds, text, stopReason);
    }

    private static int greedyToken(float[] logits) {
        if (logits == null || logits.length == 0) {
            throw new IllegalArgumentException("decoder returned empty logits");
        }
        int bestIndex = -1;
        float bestValue = Float.NEGATIVE_INFINITY;
        for (int index = 0; index < logits.length; index++) {
            float value = logits[index];
            if (Float.isNaN(value)) {
                continue;
            }
            if (bestIndex < 0 || value > bestValue) {
                bestIndex = index;
                bestValue = value;
            }
        }
        if (bestIndex < 0) {
            throw new IllegalStateException("decoder returned no usable logits");
        }
        return bestIndex;
    }

    private static int[] toIntArray(long[] values) {
        int[] result = new int[values.length];
        for (int index = 0; index < values.length; index++) {
            result[index] = Math.toIntExact(values[index]);
        }
        return result;
    }

    private static long[] toLongArray(int[] values) {
        long[] result = new long[values.length];
        for (int index = 0; index < values.length; index++) {
            result[index] = values[index];
        }
        return result;
    }

    public enum GenerationStopReason {
        MAX_NEW_TOKENS,
        END_OF_SEQUENCE,
        FUNCTION_RESPONSE
    }

    /** Immutable result of one greedy generation pass. */
    public record GenerationResult(int[] tokenIds, String text, GenerationStopReason stopReason) {
        public GenerationResult {
            tokenIds = tokenIds.clone();
            Objects.requireNonNull(text, "text");
            Objects.requireNonNull(stopReason, "stopReason");
        }

        @Override
        public int[] tokenIds() {
            return tokenIds.clone();
        }
    }

    public ModelConfig config() {
        return config;
    }

    public FunctionGemmaTokenizer tokenizer() {
        return tokenizer;
    }

    public int vocabularySize() {
        return backbone.vocabularySize();
    }

    public int hiddenSize() {
        return backbone.hiddenSize();
    }

    public int layerCount() {
        return backbone.layerCount();
    }
}
