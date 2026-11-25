package tech.kayys.aljabr.core.nn;

import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Post-norm transformer encoder layer with batch-first sequence layout.
 *
 * <p>The layer composes reusable attention, linear, and layer-normalization
 * modules. An optional additive attention mask uses {@code [sequence, sequence]}
 * layout, matching {@link MultiheadAttention}. Dropout is currently not
 * applied; callers should use this implementation for deterministic inference
 * until a backend-neutral dropout operation is available.</p>
 */
public class TransformerEncoderLayer extends NNModule {
    private final int dModel;
    private final int nHead;
    private final int dimFeedforward;
    private final float dropout;
    private final MultiheadAttention selfAttention;
    private final Linear feedForwardIn;
    private final Linear feedForwardOut;
    private final LayerNorm attentionNorm;
    private final LayerNorm feedForwardNorm;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public TransformerEncoderLayer(int dModel, int nHead, int dimFeedforward, float dropout) {
        this(null, dModel, nHead, dimFeedforward, dropout);
    }

    public TransformerEncoderLayer(ComputeBackend backend, int dModel, int nHead,
            int dimFeedforward, float dropout) {
        this(backend, dModel, nHead, dimFeedforward, dropout, 1.0e-5f);
    }

    /** Creates a post-norm encoder layer with the model's configured LayerNorm epsilon. */
    public TransformerEncoderLayer(ComputeBackend backend, int dModel, int nHead,
            int dimFeedforward, float dropout, float layerNormEpsilon) {
        super(backend);
        if (dModel <= 0 || nHead <= 0 || dModel % nHead != 0 || dimFeedforward <= 0) {
            throw new IllegalArgumentException(
                    "dModel and dimFeedforward must be positive and dModel divisible by nHead");
        }
        if (!Float.isFinite(dropout) || dropout < 0.0f || dropout >= 1.0f) {
            throw new IllegalArgumentException("dropout must be finite and in [0, 1)");
        }
        if (!Float.isFinite(layerNormEpsilon) || layerNormEpsilon <= 0.0f) {
            throw new IllegalArgumentException("layerNormEpsilon must be finite and positive");
        }
        this.dModel = dModel;
        this.nHead = nHead;
        this.dimFeedforward = dimFeedforward;
        this.dropout = dropout;

        selfAttention = registerModule("self_attn", new MultiheadAttention(backend, dModel, nHead));
        feedForwardIn = registerModule("linear1", new Linear(backend, dModel, dimFeedforward));
        feedForwardOut = registerModule("linear2", new Linear(backend, dimFeedforward, dModel));
        attentionNorm = registerModule("norm1",
                new LayerNorm(backend, new long[] {dModel}, layerNormEpsilon, true));
        feedForwardNorm = registerModule("norm2",
                new LayerNorm(backend, new long[] {dModel}, layerNormEpsilon, true));
    }

    @Override
    public Tensor forward(Tensor input) {
        return forward(input, null);
    }

    /**
     * Runs a bidirectional encoder layer.
     *
     * @param input batch-first hidden states {@code [batch, sequence, dModel]}
     * @param additiveAttentionMask optional additive mask {@code [sequence, sequence]}
     * @return newly owned hidden states with the same shape as {@code input}
     */
    public Tensor forward(Tensor input, Tensor additiveAttentionMask) {
        Objects.requireNonNull(input, "input must not be null");
        validateInputBackend(input);
        if (input.shape().rank() != 3 || input.shape().dim(2) != dModel) {
            throw new IllegalArgumentException(
                    "input must have shape [batch, sequence, " + dModel + "]");
        }
        if (additiveAttentionMask != null) {
            validateInputBackend(additiveAttentionMask);
        }

        Tensor attentionOutput = null;
        Tensor attentionResidual = null;
        Tensor normalizedAttention = null;
        Tensor projectedIn = null;
        Tensor activated = null;
        Tensor projectedOut = null;
        Tensor feedForwardResidual = null;
        try {
            attentionOutput = selfAttention.forward(
                    input, input, input, additiveAttentionMask, false);
            attentionResidual = input.add(attentionOutput);
            normalizedAttention = attentionNorm.forward(attentionResidual);

            long batch = input.shape().dim(0);
            long sequence = input.shape().dim(1);
            Tensor flattened = normalizedAttention.reshape(Math.multiplyExact(batch, sequence), dModel);
            try {
                projectedIn = feedForwardIn.forward(flattened);
            } finally {
                flattened.release();
            }
            activated = projectedIn.gelu();
            Tensor activatedFlat = activated.reshape(Math.multiplyExact(batch, sequence), dimFeedforward);
            try {
                projectedOut = feedForwardOut.forward(activatedFlat);
            } finally {
                activatedFlat.release();
            }
            Tensor projectedSequence = projectedOut.reshape(batch, sequence, dModel);
            try {
                feedForwardResidual = normalizedAttention.add(projectedSequence);
            } finally {
                projectedSequence.release();
            }
            return feedForwardNorm.forward(feedForwardResidual);
        } finally {
            if (feedForwardResidual != null) feedForwardResidual.release();
            if (projectedOut != null) projectedOut.release();
            if (activated != null) activated.release();
            if (projectedIn != null) projectedIn.release();
            if (normalizedAttention != null) normalizedAttention.release();
            if (attentionResidual != null) attentionResidual.release();
            if (attentionOutput != null) attentionOutput.release();
        }
    }

    public int dModel() { return dModel; }
    public int nHead() { return nHead; }
    public int dimFeedforward() { return dimFeedforward; }
    public float dropout() { return dropout; }
}
