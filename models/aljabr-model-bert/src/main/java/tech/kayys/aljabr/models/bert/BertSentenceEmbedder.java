package tech.kayys.aljabr.models.bert;

import java.util.Objects;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.models.embedding.common.L2Normalizer;
import tech.kayys.aljabr.models.embedding.common.MaskedMeanPooler;

/**
 * Sentence-transformer output pipeline for BERT-family encoders.
 *
 * <p>Runs the encoder, performs attention-mask-aware mean pooling over the
 * final token states, then L2-normalizes each sentence vector. The returned
 * vectors are host-side float arrays; encoder execution remains on its
 * configured compute backend.</p>
 *
 * <p>As in the Sentence Transformers pooling module, every position with
 * attention-mask value {@code 1} contributes to the mean, including special
 * tokens such as CLS and SEP.</p>
 */
public final class BertSentenceEmbedder {
    private final BertEncoder encoder;

    public BertSentenceEmbedder(BertEncoder encoder) {
        this.encoder = Objects.requireNonNull(encoder, "encoder must not be null");
    }

    /** Returns one normalized vector per input row, at the encoder hidden size. */
    public float[][] embed(Tensor inputIds, Tensor attentionMask) {
        return embed(inputIds, attentionMask, null, encoder.config().hiddenSize());
    }

    /** Returns normalized embeddings, optionally truncated before normalization. */
    public float[][] embed(Tensor inputIds, Tensor attentionMask, Tensor tokenTypeIds,
            int outputDimension) {
        Objects.requireNonNull(inputIds, "inputIds must not be null");
        if (outputDimension <= 0 || outputDimension > encoder.config().hiddenSize()) {
            throw new IllegalArgumentException("outputDimension must be between 1 and encoder hidden size "
                    + encoder.config().hiddenSize());
        }
        long[] dimensions = inputIds.shape().dims();
        if (dimensions.length != 2 || dimensions[0] <= 0 || dimensions[1] <= 0) {
            throw new IllegalArgumentException("inputIds must have non-empty [batch, sequence] shape");
        }
        int batch = Math.toIntExact(dimensions[0]);
        int sequence = Math.toIntExact(dimensions[1]);
        boolean[][] included = readAttentionMask(attentionMask, batch, sequence);
        Tensor hidden = null;
        try {
            hidden = encoder.forward(inputIds, attentionMask, tokenTypeIds);
            float[] values = hidden.toFloatArray();
            int hiddenSize = encoder.config().hiddenSize();
            int expected = Math.toIntExact(Math.multiplyExact(
                    Math.multiplyExact((long) batch, sequence), hiddenSize));
            if (values.length != expected) {
                throw new IllegalStateException("BERT encoder returned " + values.length
                        + " values; expected " + expected);
            }
            float[][] embeddings = new float[batch][];
            for (int b = 0; b < batch; b++) {
                float[][] tokenStates = new float[sequence][hiddenSize];
                int batchOffset = b * sequence * hiddenSize;
                for (int token = 0; token < sequence; token++) {
                    System.arraycopy(values, batchOffset + token * hiddenSize,
                            tokenStates[token], 0, hiddenSize);
                }
                embeddings[b] = L2Normalizer.normalizePrefix(
                        MaskedMeanPooler.meanPool(tokenStates, included[b]), outputDimension);
            }
            return embeddings;
        } finally {
            if (hidden != null) hidden.release();
        }
    }

    private static boolean[][] readAttentionMask(Tensor attentionMask, int batch, int sequence) {
        boolean[][] included = new boolean[batch][sequence];
        if (attentionMask == null) {
            for (boolean[] row : included) java.util.Arrays.fill(row, true);
            return included;
        }
        if (attentionMask.dtype() != DType.I32) {
            throw new IllegalArgumentException("attentionMask must have dtype I32");
        }
        if (attentionMask.shape().rank() != 2
                || attentionMask.shape().dim(0) != batch
                || attentionMask.shape().dim(1) != sequence) {
            throw new IllegalArgumentException("attentionMask must match inputIds [batch, sequence]");
        }
        int[] values = attentionMask.toIntArray();
        for (int i = 0; i < values.length; i++) {
            if (values[i] != 0 && values[i] != 1) {
                throw new IllegalArgumentException("attentionMask values must be binary (0 or 1)");
            }
            included[i / sequence][i % sequence] = values[i] == 1;
        }
        return included;
    }
}
