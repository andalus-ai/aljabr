package tech.kayys.aljabr.models;

import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Reusable UNet bottleneck stage with residual processing and optional
 * spatial attention.
 */
public final class UNetMidBlock2D extends NNModule {

    private final ResnetBlock2D resnet;
    private final SpatialTransformer attention;

    public UNetMidBlock2D(
            ComputeBackend backend,
            int channels,
            int timeEmbeddingDim,
            Integer attentionHeads,
            int attentionHeadDim,
            int contextDim) {
        super(backend);
        this.resnet = register("resnet", new ResnetBlock2D(
                backend, channels, channels, timeEmbeddingDim));
        this.attention = attentionHeads == null ? null : register(
                "attention", new SpatialTransformer(
                        backend, channels, attentionHeads, attentionHeadDim, contextDim));
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
