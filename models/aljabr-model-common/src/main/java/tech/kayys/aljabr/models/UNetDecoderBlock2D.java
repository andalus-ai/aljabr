package tech.kayys.aljabr.models;

import java.util.List;
import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.Upsample2d;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorOps;

/**
 * Reusable decoder stage: resize to the encoder feature map, concatenate the
 * skip, then apply residual/optional-attention/residual processing.
 */
public final class UNetDecoderBlock2D extends NNModule {

    private final Upsample2d upsample;
    private final ResnetBlock2D resnetIn;
    private final SpatialTransformer attention;
    private final ResnetBlock2D resnetOut;

    public UNetDecoderBlock2D(
            ComputeBackend backend,
            int hiddenChannels,
            int skipChannels,
            int outChannels,
            int timeEmbeddingDim,
            Integer attentionHeads,
            int attentionHeadDim,
            int contextDim) {
        super(backend);
        this.upsample = register("upsample", new Upsample2d(backend, 2.0f));
        this.resnetIn = register("resnet_in", new ResnetBlock2D(
                backend, Math.addExact(hiddenChannels, skipChannels), outChannels, timeEmbeddingDim));
        this.attention = attentionHeads == null ? null : register(
                "attention", new SpatialTransformer(
                        backend, outChannels, attentionHeads, attentionHeadDim, contextDim));
        this.resnetOut = register("resnet_out", new ResnetBlock2D(
                backend, outChannels, outChannels, timeEmbeddingDim));
        if (attentionHeads != null && attentionHeads <= 0) {
            throw new IllegalArgumentException("attentionHeads must be positive when attention is enabled");
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException(
                "UNetDecoderBlock2D requires an encoder skip tensor and optional conditioning");
    }

    public Tensor forward(Tensor input, Tensor skip, Tensor timeEmbedding, Tensor context) {
        Objects.requireNonNull(input, "input must not be null");
        Objects.requireNonNull(skip, "skip must not be null");
        if (input.shape().rank() != 4 || skip.shape().rank() != 4
                || input.shape().dim(0) != skip.shape().dim(0)) {
            throw new IllegalArgumentException("decoder input and skip must be rank-4 tensors with matching batches");
        }
        Tensor resized = upsample.forwardTo(input, skip);
        Tensor merged = TensorOps.cat(List.of(resized, skip), 1);
        Tensor hidden = resnetIn.forward(merged, timeEmbedding);
        if (attention != null) {
            hidden = attention.forward(hidden, context);
        }
        return resnetOut.forward(hidden, timeEmbedding);
    }

    public boolean hasAttention() {
        return attention != null;
    }
}
