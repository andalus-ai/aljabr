package tech.kayys.aljabr.models.gemma3;

import java.util.Arrays;
import java.util.Objects;
import tech.kayys.aljabr.cache.GrowingHeapKVCacheStorage;
import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.cache.KVCacheSession;
import tech.kayys.aljabr.cache.KVCacheStorage;
import tech.kayys.aljabr.cache.KVCacheStorageFactory;
import tech.kayys.aljabr.spi.model.ModelConfig;

/**
 * Float32 CPU reference forward path for the Gemma 3 text decoder.
 *
 * <p>This class provides full-sequence causal forward and a float32 incremental
 * reference decoder. Incremental sessions use Aljabr's shared KV-cache storage
 * and committed-prefix contract; optimized backends can implement the same
 * storage abstraction without changing model semantics.</p>
 */
public final class Gemma3TextBackbone {
    private final ModelConfig config;
    private final Gemma3TextCheckpointMapper.Weights weights;
    private final int hidden;
    private final int queryHeads;
    private final int kvHeads;
    private final int headDim;
    private final double epsilon;
    private final KVCacheStorageFactory cacheStorageFactory;

    public Gemma3TextBackbone(ModelConfig config, Gemma3TextCheckpointMapper.Weights weights) {
        this(config, weights, GrowingHeapKVCacheStorage::new);
    }

    /** Creates a reference decoder with an explicitly selected KV-cache provider. */
    public Gemma3TextBackbone(
            ModelConfig config,
            Gemma3TextCheckpointMapper.Weights weights,
            KVCacheStorageFactory cacheStorageFactory) {
        this.config = Objects.requireNonNull(config, "config");
        this.cacheStorageFactory = Objects.requireNonNull(cacheStorageFactory, "cacheStorageFactory");
        this.weights = Objects.requireNonNull(weights, "weights");
        this.hidden = config.hiddenSize();
        this.queryHeads = config.numAttentionHeads();
        this.kvHeads = config.resolvedNumKvHeads();
        this.headDim = config.resolvedHeadDim();
        this.epsilon = config.rmsNormEps();
        validate();
    }

    /**
     * Computes vocabulary logits for every input position.
     *
     * @param inputIds token IDs in one unpadded sequence
     * @return logits with shape [sequence][vocabulary]
     */
    public float[][] forward(int[] inputIds) {
        Objects.requireNonNull(inputIds, "inputIds");
        if (inputIds.length == 0) throw new IllegalArgumentException("inputIds must not be empty");
        if (inputIds.length > config.maxPositionEmbeddings()) {
            throw new IllegalArgumentException("input sequence exceeds maxPositionEmbeddings");
        }
        float[][] embeddings = new float[inputIds.length][hidden];
        float scale = (float) Math.sqrt(hidden) * config.resolvedEmbeddingMultiplier();
        float[][] table = weights.tokenEmbeddings();
        for (int token = 0; token < inputIds.length; token++) {
            int id = inputIds[token];
            if (id < 0 || id >= config.vocabSize()) {
                throw new IllegalArgumentException("token ID out of vocabulary range at index " + token + ": " + id);
            }
            for (int d = 0; d < hidden; d++) embeddings[token][d] = table[id][d] * scale;
        }

        float[][] state = embeddings;
        for (int layer = 0; layer < weights.layers().size(); layer++) {
            state = forwardLayer(state, weights.layers().get(layer), layer);
        }

        float[][] normalized = rmsNorm(state, weights.finalNorm());
        float[][] logits = linear(normalized, weights.languageModelHead());
        float softcap = config.finalLogitSoftcapping() == null
                ? 0.0f : config.finalLogitSoftcapping().floatValue();
        float lmScale = config.resolvedLmHeadMultiplier();
        for (int token = 0; token < logits.length; token++) {
            for (int vocab = 0; vocab < logits[token].length; vocab++) {
                float value = logits[token][vocab] * lmScale;
                logits[token][vocab] = softcap > 0.0f
                        ? (float) (Math.tanh(value / softcap) * softcap) : value;
            }
        }
        return logits;
    }


    /**
     * Creates isolated incremental decode state for one sequence. Each session
     * owns its own per-layer key/value cache and may be used independently of
     * other sessions sharing this immutable backbone.
     */
    public DecodeSession newDecodeSession() {
        return new DecodeSession();
    }

    /**
     * Stateful single-sequence decoder. Call {@link #prefill(int[])} once, then
     * call {@link #decode(int)} for each newly generated token. Returned values
     * are vocabulary logits for the newly appended token.
     */
    public final class DecodeSession implements AutoCloseable {
        private final int cacheWidth = Math.multiplyExact(kvHeads, headDim);
        private final int maximumPositions = config.maxPositionEmbeddings();
        private final KVCacheStorage cacheStorage;
        private final KVCacheSession cacheSession;
        private boolean closed;

        private DecodeSession() {
            int[] headsPerLayer = new int[weights.layers().size()];
            int[] dimensionsPerLayer = new int[weights.layers().size()];
            Arrays.fill(headsPerLayer, kvHeads);
            Arrays.fill(dimensionsPerLayer, headDim);
            KVCacheLayout layout = new KVCacheLayout(
                    maximumPositions, headsPerLayer, dimensionsPerLayer);
            cacheStorage = Objects.requireNonNull(
                    cacheStorageFactory.create(layout), "cacheStorageFactory returned null");
            if (cacheStorage.layout() != layout) {
                cacheStorage.close();
                throw new IllegalArgumentException(
                        "KV cache storage factory must preserve the supplied layout instance");
            }
            cacheSession = new KVCacheSession(cacheStorage);
        }

        /** Processes a prompt on a fresh session and returns logits for every prompt token. */
        public float[][] prefill(int[] inputIds) {
            ensureOpen();
            Objects.requireNonNull(inputIds, "inputIds");
            if (position() != 0) {
                throw new IllegalStateException("prefill is only valid on a fresh decode session");
            }
            if (inputIds.length == 0) {
                throw new IllegalArgumentException("inputIds must not be empty");
            }
            float[][] result = new float[inputIds.length][];
            for (int index = 0; index < inputIds.length; index++) {
                result[index] = decode(inputIds[index]);
            }
            return result;
        }

        /** Appends one token and returns its vocabulary logits. */
        public float[] decode(int tokenId) {
            ensureOpen();
            if (tokenId < 0 || tokenId >= config.vocabSize()) {
                throw new IllegalArgumentException("token ID out of vocabulary range: " + tokenId);
            }
            if (position() >= maximumPositions) {
                throw new IllegalArgumentException("decode exceeds maxPositionEmbeddings");
            }

            float scale = (float) Math.sqrt(hidden) * config.resolvedEmbeddingMultiplier();
            float[][] embeddingTable = weights.tokenEmbeddings();
            float[][] state = new float[1][hidden];
            for (int dimension = 0; dimension < hidden; dimension++) {
                state[0][dimension] = embeddingTable[tokenId][dimension] * scale;
            }

            for (int layerIndex = 0; layerIndex < weights.layers().size(); layerIndex++) {
                state = forwardIncrementalLayer(
                        state, weights.layers().get(layerIndex), layerIndex, position());
            }

            float[][] normalized = rmsNorm(state, weights.finalNorm());
            float[][] logits = linear(normalized, weights.languageModelHead());
            float softcap = config.finalLogitSoftcapping() == null
                    ? 0.0f : config.finalLogitSoftcapping().floatValue();
            float lmScale = config.resolvedLmHeadMultiplier();
            float[] result = logits[0];
            for (int vocab = 0; vocab < result.length; vocab++) {
                float value = result[vocab] * lmScale;
                result[vocab] = softcap > 0.0f
                        ? (float) (Math.tanh(value / softcap) * softcap) : value;
            }
            cacheSession.commit(position() + 1);
            return result;
        }

        /** Number of tokens already processed by this session. */
        public int position() {
            ensureOpen();
            return cacheSession.committedLength();
        }

        @Override
        public void close() {
            if (!closed) {
                cacheSession.reset();
                cacheStorage.close();
                closed = true;
            }
        }

        private void ensureOpen() {
            if (closed) throw new IllegalStateException("decode session is closed");
        }

        private float[][] forwardIncrementalLayer(
                float[][] input,
                Gemma3TextCheckpointMapper.LayerWeights layer,
                int layerIndex,
                int absolutePosition) {
            float[][] normalized = rmsNorm(input, layer.attentionNorm());
            float[][] queryFlat = linear(normalized, layer.query());
            float[][] keyFlat = linear(normalized, layer.key());
            float[][] valueFlat = linear(normalized, layer.value());

            float[][][] query = reshape(queryFlat, queryHeads, headDim);
            float[][][] key = headRmsNorm(reshape(keyFlat, kvHeads, headDim), layer.keyNorm());
            query = headRmsNorm(query, layer.queryNorm());
            query = applyRope(query, new int[] {absolutePosition},
                    config.ropeThetaForLayer(layerIndex),
                    config.partialRotaryFactorForLayer(layerIndex));
            key = applyRope(key, new int[] {absolutePosition},
                    config.ropeThetaForLayer(layerIndex),
                    config.partialRotaryFactorForLayer(layerIndex));

            float[] currentKey = new float[cacheWidth];
            float[] currentValue = valueFlat[0].clone();
            for (int head = 0; head < kvHeads; head++) {
                System.arraycopy(key[0][head], 0, currentKey, head * headDim, headDim);
            }
            cacheSession.write(layerIndex, absolutePosition,
                    currentKey, 0, currentValue, 0);

            float[][][] attended = new float[1][queryHeads][headDim];
            int groups = queryHeads / kvHeads;
            int firstVisible = config.isSlidingAttentionLayer(layerIndex)
                    ? Math.max(0, absolutePosition - config.slidingWindowSize() + 1) : 0;
            double scoreScale = 1.0d / Math.sqrt(config.queryPreAttnScalar());
            Double configuredSoftcap = config.attnLogitSoftcapping();
            double softcap = configuredSoftcap == null ? 0.0d : configuredSoftcap;
            int visibleTokenCount = absolutePosition - firstVisible + 1;
            float[] visibleKeys = new float[Math.multiplyExact(visibleTokenCount, cacheWidth)];
            float[] visibleValues = new float[visibleKeys.length];
            cacheSession.readKeyRangeForAttention(
                    layerIndex, firstVisible, visibleTokenCount, visibleKeys, 0);
            cacheSession.readValueRangeForAttention(
                    layerIndex, firstVisible, visibleTokenCount, visibleValues, 0);
            for (int head = 0; head < queryHeads; head++) {
                int kvHead = head / groups;
                double maximum = Double.NEGATIVE_INFINITY;
                double[] scores = new double[visibleTokenCount];
                for (int source = firstVisible; source <= absolutePosition; source++) {
                    int sourceOffset = (source - firstVisible) * cacheWidth + kvHead * headDim;
                    double dot = 0.0d;
                    for (int dimension = 0; dimension < headDim; dimension++) {
                        dot += (double) query[0][head][dimension]
                                * visibleKeys[sourceOffset + dimension];
                    }
                    double score = dot * scoreScale;
                    if (softcap > 0.0d) score = Math.tanh(score / softcap) * softcap;
                    scores[source - firstVisible] = score;
                    maximum = Math.max(maximum, score);
                }
                double denominator = 0.0d;
                for (double score : scores) denominator += Math.exp(score - maximum);
                for (int source = firstVisible; source <= absolutePosition; source++) {
                    int sourceOffset = (source - firstVisible) * cacheWidth + kvHead * headDim;
                    double probability = Math.exp(scores[source - firstVisible] - maximum) / denominator;
                    for (int dimension = 0; dimension < headDim; dimension++) {
                        attended[0][head][dimension] +=
                                (float) (probability * visibleValues[sourceOffset + dimension]);
                    }
                }
            }

            float[][] attentionOutput = linear(flatten(attended), layer.attentionOutput());
            attentionOutput = rmsNorm(attentionOutput, layer.postAttentionNorm());
            float[][] afterAttention = add(input, attentionOutput);
            normalized = rmsNorm(afterAttention, layer.preFeedForwardNorm());
            float[][] gate = linear(normalized, layer.feedForwardGate());
            float[][] up = linear(normalized, layer.feedForwardUp());
            for (int dimension = 0; dimension < gate[0].length; dimension++) {
                gate[0][dimension] = geluTanh(gate[0][dimension]) * up[0][dimension];
            }
            float[][] feedForward = linear(gate, layer.feedForwardDown());
            feedForward = rmsNorm(feedForward, layer.postFeedForwardNorm());
            return add(afterAttention, feedForward);
        }
    }

    public int hiddenSize() { return hidden; }
    public int layerCount() { return weights.layers().size(); }
    public int vocabularySize() { return config.vocabSize(); }

    private float[][] forwardLayer(float[][] input, Gemma3TextCheckpointMapper.LayerWeights layer, int index) {
        float[][] normalized = rmsNorm(input, layer.attentionNorm());
        float[][] queryFlat = linear(normalized, layer.query());
        float[][] keyFlat = linear(normalized, layer.key());
        float[][] valueFlat = linear(normalized, layer.value());

        float[][][] query = reshape(queryFlat, queryHeads, headDim);
        float[][][] key = reshape(keyFlat, kvHeads, headDim);
        float[][][] value = reshape(valueFlat, kvHeads, headDim);
        query = headRmsNorm(query, layer.queryNorm());
        key = headRmsNorm(key, layer.keyNorm());

        int[] positions = new int[input.length];
        for (int i = 0; i < positions.length; i++) positions[i] = i;
        double theta = config.ropeThetaForLayer(index);
        double rotaryFactor = config.partialRotaryFactorForLayer(index);
        query = applyRope(query, positions, theta, rotaryFactor);
        key = applyRope(key, positions, theta, rotaryFactor);

        float[][][] attended = causalAttention(query, key, value, index);
        float[][] attentionFlat = flatten(attended);
        float[][] attentionOutput = linear(attentionFlat, layer.attentionOutput());
        attentionOutput = rmsNorm(attentionOutput, layer.postAttentionNorm());
        float[][] afterAttention = add(input, attentionOutput);

        normalized = rmsNorm(afterAttention, layer.preFeedForwardNorm());
        float[][] gate = linear(normalized, layer.feedForwardGate());
        float[][] up = linear(normalized, layer.feedForwardUp());
        for (int token = 0; token < gate.length; token++) {
            for (int d = 0; d < gate[token].length; d++) {
                gate[token][d] = geluTanh(gate[token][d]) * up[token][d];
            }
        }
        float[][] feedForward = linear(gate, layer.feedForwardDown());
        feedForward = rmsNorm(feedForward, layer.postFeedForwardNorm());
        return add(afterAttention, feedForward);
    }

    private float[][][] causalAttention(float[][][] query, float[][][] key, float[][][] value, int layerIndex) {
        int sequence = query.length;
        int groups = queryHeads / kvHeads;
        float[][][] output = new float[sequence][queryHeads][headDim];
        double scoreScale = 1.0d / Math.sqrt(config.queryPreAttnScalar());
        Double softcapValue = config.attnLogitSoftcapping();
        double softcap = softcapValue == null ? 0.0d : softcapValue;
        boolean sliding = config.isSlidingAttentionLayer(layerIndex);
        int window = config.slidingWindowSize();

        for (int head = 0; head < queryHeads; head++) {
            int kvHead = head / groups;
            for (int token = 0; token < sequence; token++) {
                int firstVisible = sliding ? Math.max(0, token - window + 1) : 0;
                double max = Double.NEGATIVE_INFINITY;
                double[] scores = new double[token - firstVisible + 1];
                for (int source = firstVisible; source <= token; source++) {
                    double dot = 0.0d;
                    for (int d = 0; d < headDim; d++) {
                        dot += (double) query[token][head][d] * key[source][kvHead][d];
                    }
                    double score = dot * scoreScale;
                    if (softcap > 0.0d) score = Math.tanh(score / softcap) * softcap;
                    scores[source - firstVisible] = score;
                    max = Math.max(max, score);
                }
                double denominator = 0.0d;
                for (double score : scores) denominator += Math.exp(score - max);
                for (int source = firstVisible; source <= token; source++) {
                    double probability = Math.exp(scores[source - firstVisible] - max) / denominator;
                    for (int d = 0; d < headDim; d++) {
                        output[token][head][d] += (float) (probability * value[source][kvHead][d]);
                    }
                }
            }
        }
        return output;
    }

    private float[][] rmsNorm(float[][] input, float[] scale) {
        float[][] output = new float[input.length][hidden];
        for (int token = 0; token < input.length; token++) {
            double squareSum = 0.0d;
            for (float value : input[token]) squareSum += (double) value * value;
            double inverse = 1.0d / Math.sqrt(squareSum / hidden + epsilon);
            for (int d = 0; d < hidden; d++) {
                output[token][d] = (float) (input[token][d] * inverse * (scale[d] + 1.0f));
            }
        }
        return output;
    }

    private float[][][] headRmsNorm(float[][][] input, float[] scale) {
        float[][][] output = new float[input.length][input[0].length][headDim];
        for (int token = 0; token < input.length; token++) {
            for (int head = 0; head < input[token].length; head++) {
                double squareSum = 0.0d;
                for (float value : input[token][head]) squareSum += (double) value * value;
                double inverse = 1.0d / Math.sqrt(squareSum / headDim + epsilon);
                for (int d = 0; d < headDim; d++) {
                    output[token][head][d] = (float) (input[token][head][d] * inverse * (scale[d] + 1.0f));
                }
            }
        }
        return output;
    }

    private static float[][][] applyRope(float[][][] input, int[] positions, double theta, double factor) {
        int sequence = input.length;
        int heads = input[0].length;
        int dimension = input[0][0].length;
        int rotaryDimension = ((int) Math.floor(dimension * factor)) & ~1;
        if (rotaryDimension <= 0) throw new IllegalArgumentException("Gemma 3 rotary dimension must be positive");
        int half = rotaryDimension / 2;
        float[][][] output = new float[sequence][heads][dimension];
        for (int token = 0; token < sequence; token++) {
            for (int head = 0; head < heads; head++) {
                System.arraycopy(input[token][head], 0, output[token][head], 0, dimension);
                for (int d = 0; d < half; d++) {
                    double inverseFrequency = 1.0d / Math.pow(theta, (2.0d * d) / rotaryDimension);
                    double angle = positions[token] * inverseFrequency;
                    double cosine = Math.cos(angle);
                    double sine = Math.sin(angle);
                    double first = input[token][head][d];
                    double second = input[token][head][d + half];
                    output[token][head][d] = (float) (first * cosine - second * sine);
                    output[token][head][d + half] = (float) (second * cosine + first * sine);
                }
            }
        }
        return output;
    }

    private static float[][][] reshape(float[][] flat, int heads, int dimension) {
        int expected = Math.multiplyExact(heads, dimension);
        float[][][] result = new float[flat.length][heads][dimension];
        for (int token = 0; token < flat.length; token++) {
            if (flat[token].length != expected) {
                throw new IllegalArgumentException("projection width does not match attention head geometry");
            }
            for (int head = 0; head < heads; head++) {
                System.arraycopy(flat[token], head * dimension, result[token][head], 0, dimension);
            }
        }
        return result;
    }

    private static float[][] flatten(float[][][] input) {
        int sequence = input.length;
        int heads = input[0].length;
        int dimension = input[0][0].length;
        float[][] output = new float[sequence][heads * dimension];
        for (int token = 0; token < sequence; token++) {
            for (int head = 0; head < heads; head++) {
                System.arraycopy(input[token][head], 0, output[token], head * dimension, dimension);
            }
        }
        return output;
    }

    private static float[][] linear(float[][] input, float[][] weight) {
        int outputSize = weight.length;
        int inputSize = weight[0].length;
        float[][] output = new float[input.length][outputSize];
        for (int token = 0; token < input.length; token++) {
            for (int out = 0; out < outputSize; out++) {
                double sum = 0.0d;
                for (int in = 0; in < inputSize; in++) sum += (double) weight[out][in] * input[token][in];
                output[token][out] = (float) sum;
            }
        }
        return output;
    }

    private static float[][] add(float[][] left, float[][] right) {
        float[][] output = new float[left.length][left[0].length];
        for (int token = 0; token < left.length; token++) {
            for (int d = 0; d < left[token].length; d++) output[token][d] = left[token][d] + right[token][d];
        }
        return output;
    }

    private static float geluTanh(float value) {
        double inner = Math.sqrt(2.0d / Math.PI) * (value + 0.044715d * value * value * value);
        return (float) (0.5d * value * (1.0d + Math.tanh(inner)));
    }

    private void validate() {
        if (hidden <= 0 || queryHeads <= 0 || kvHeads <= 0 || headDim <= 0
                || queryHeads % kvHeads != 0 || hidden != queryHeads * headDim
                || config.numHiddenLayers() != weights.layers().size()) {
            throw new IllegalArgumentException("Gemma 3 config dimensions do not match the mapped checkpoint");
        }
        if (weights.tokenEmbeddings().length != config.vocabSize()
                || weights.finalNorm().length != hidden
                || weights.languageModelHead().length != config.vocabSize()) {
            throw new IllegalArgumentException("Gemma 3 embedding, norm, or LM-head dimensions are inconsistent");
        }
        for (int layerIndex = 0; layerIndex < weights.layers().size(); layerIndex++) {
            Gemma3TextCheckpointMapper.LayerWeights layer = weights.layers().get(layerIndex);
            requireMatrix(layer.query(), queryHeads * headDim, hidden, "query", layerIndex);
            requireMatrix(layer.key(), kvHeads * headDim, hidden, "key", layerIndex);
            requireMatrix(layer.value(), kvHeads * headDim, hidden, "value", layerIndex);
            requireMatrix(layer.attentionOutput(), hidden, queryHeads * headDim, "attentionOutput", layerIndex);
            requireMatrix(layer.feedForwardGate(), config.intermediateSize(), hidden, "feedForwardGate", layerIndex);
            requireMatrix(layer.feedForwardUp(), config.intermediateSize(), hidden, "feedForwardUp", layerIndex);
            requireMatrix(layer.feedForwardDown(), hidden, config.intermediateSize(), "feedForwardDown", layerIndex);
            requireVector(layer.attentionNorm(), hidden, "attentionNorm", layerIndex);
            requireVector(layer.postAttentionNorm(), hidden, "postAttentionNorm", layerIndex);
            requireVector(layer.preFeedForwardNorm(), hidden, "preFeedForwardNorm", layerIndex);
            requireVector(layer.postFeedForwardNorm(), hidden, "postFeedForwardNorm", layerIndex);
            requireVector(layer.queryNorm(), headDim, "queryNorm", layerIndex);
            requireVector(layer.keyNorm(), headDim, "keyNorm", layerIndex);
        }
    }

    private static void requireMatrix(float[][] matrix, int rows, int columns, String name, int layer) {
        if (matrix == null || matrix.length != rows) {
            throw new IllegalArgumentException(name + " weights have invalid output dimension at layer " + layer);
        }
        for (float[] row : matrix) {
            if (row == null || row.length != columns) {
                throw new IllegalArgumentException(name + " weights have invalid input dimension at layer " + layer);
            }
        }
    }

    private static void requireVector(float[] vector, int length, String name, int layer) {
        if (vector == null || vector.length != length) {
            throw new IllegalArgumentException(name + " weights have invalid dimension at layer " + layer);
        }
    }
}
