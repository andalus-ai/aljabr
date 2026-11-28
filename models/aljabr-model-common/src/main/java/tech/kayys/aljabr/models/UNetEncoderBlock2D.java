package tech.kayys.aljabr.models;

import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Reusable encoder stage: residual feature processing followed by optional
 * spatial self/cross-attention.
 */
public final class UNetEncoderBlock2D extends NNModule {

    private final ResnetBlock2D resnet;
    private final SpatialTransformer attention;

    public UNetEncoderBlock2D(
            ComputeBackend backend,
            int inChannels,
            int outChannels,
            int timeEmbeddingDim,
            Integer attentionHeads,
            int attentionHeadDim,
            int contextDim) {
        super(backend);
        this.resnet = register("resnet", new ResnetBlock2D(
                backend, inChannels, outChannels, timeEmbeddingDim));
        this.attention = attentionHeads == null ? null : register(
                "attention", new SpatialTransformer(
                        backend, outChannels, attentionHeads, attentionHeadDim, contextDim));
        if (attentionHeads != null && attentionHeads <= 0) {
            throw new IllegalArgumentException("attentionHeads must be positive when attention is enabled");
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        return forward(input, null, null);
    }

    public Tensor forward(Tensor input, Tensor timeEmbedding, Tensor context) {
        Objects.requireNonNull(input, "input must not be null");
        Tensor hidden = resnet.forward(input, timeEmbedding);
        return attention == null ? hidden : attention.forward(hidden, context);
    }

    public boolean hasAttention() {
        return attention != null;
    }
}
