package tech.kayys.aljabr.models;

import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.GroupNorm;
import tech.kayys.aljabr.core.nn.LayerNorm;
import tech.kayys.aljabr.core.nn.Conv2d;
import tech.kayys.aljabr.core.nn.MultiheadAttention;
import tech.kayys.aljabr.core.nn.Linear;

/**
 * Spatial transformer block for diffusion models.
 *
 * <p>This class owns its parameters and child modules on one explicit compute
 * backend. Context features are projected to the transformer inner dimension
 * before cross-attention.</p>
 */
public class SpatialTransformer extends NNModule {

    private static final int NORM_GROUPS = 32;

    private final int inChannels;
    private final int innerDim;
    private final int contextDim;

    private final GroupNorm norm;
    private final Conv2d projIn;
    private final LayerNorm norm1;
    private final MultiheadAttention selfAttn;
    private final LayerNorm norm2;
    private final MultiheadAttention crossAttn;
    private final Linear contextProjection;
    private final LayerNorm norm3;
    private final Linear ff1;
    private final Linear ff2;
    private final Conv2d projOut;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public SpatialTransformer(int inChannels, int numHeads, int headDim, int contextDim) {
        this(null, inChannels, numHeads, headDim, contextDim);
    }

    public SpatialTransformer(
            ComputeBackend backend,
            int inChannels,
            int numHeads,
            int headDim,
            int contextDim) {
        super(backend);
        if (inChannels <= 0 || numHeads <= 0 || headDim <= 0 || contextDim <= 0) {
            throw new IllegalArgumentException("channel and attention dimensions must be positive");
        }
        if (inChannels % NORM_GROUPS != 0) {
            throw new IllegalArgumentException(
                    "inChannels must be divisible by " + NORM_GROUPS + " for GroupNorm");
        }

        this.inChannels = inChannels;
        this.innerDim = Math.multiplyExact(numHeads, headDim);
        this.contextDim = contextDim;

        this.norm = register("norm", new GroupNorm(backend, NORM_GROUPS, inChannels));
        this.projIn = register("proj_in", new Conv2d(backend, inChannels, innerDim, 1, 1, 0));

        this.norm1 = register("norm1", new LayerNorm(backend, innerDim));
        this.selfAttn = register("attn1", new MultiheadAttention(backend, innerDim, numHeads));

        this.norm2 = register("norm2", new LayerNorm(backend, innerDim));
        this.crossAttn = register("attn2", new MultiheadAttention(backend, innerDim, numHeads));
        this.contextProjection = register("context_projection", new Linear(backend, contextDim, innerDim));

        this.norm3 = register("norm3", new LayerNorm(backend, innerDim));
        this.ff1 = register("ff_net_primary", new Linear(backend, innerDim, innerDim * 4L));
        this.ff2 = register("ff_net_secondary", new Linear(backend, innerDim * 4L, innerDim));

        this.projOut = register("proj_out", new Conv2d(backend, innerDim, inChannels, 1, 1, 0));
    }

    @Override
    public Tensor forward(Tensor input) {
        return forward(input, null);
    }

    /**
     * Forward pass with optional context.
     *
     * @param input spatial features with shape [N, inChannels, H, W]
     * @param context optional context with shape [N, sequenceLength, contextDim]
     * @return spatial features with shape [N, inChannels, H, W]
     */
    public Tensor forward(Tensor input, Tensor context) {
        Objects.requireNonNull(input, "input must not be null");
        validateInputBackend(input);
        if (input.shape().rank() != 4 || input.shape().dim(1) != inChannels) {
            throw new IllegalArgumentException(
                    "SpatialTransformer expected [N, " + inChannels + ", H, W] input but got " + input.shape());
        }
        if (context != null) {
            validateInputBackend(context);
            if (context.shape().rank() != 3
                    || context.shape().dim(0) != input.shape().dim(0)
                    || context.shape().dim(2) != contextDim) {
                throw new IllegalArgumentException(
                        "context must have shape [N, sequenceLength, " + contextDim + "]");
            }
        }

        long[] shape = input.shape().dims();
        int batch = Math.toIntExact(shape[0]);
        int height = Math.toIntExact(shape[2]);
        int width = Math.toIntExact(shape[3]);

        Tensor x = norm.forward(input);
        x = projIn.forward(x);

        // [N, innerDim, H, W] -> [N, H*W, innerDim]
        x = x.reshape(batch, innerDim, height * width).transpose();

        Tensor residual = x;
        x = norm1.forward(x.reshape(batch * height * width, innerDim))
                .reshape(batch, height * width, innerDim);
        x = selfAttn.forward(x).add(residual);

        if (context != null) {
            residual = x;
            x = norm2.forward(x.reshape(batch * height * width, innerDim))
                    .reshape(batch, height * width, innerDim);
            Tensor projectedContext = context.reshape(batch * context.shape().dim(1), contextDim);
            Tensor contextFeatures = null;
            try {
                contextFeatures = contextProjection.forward(projectedContext)
                        .reshape(batch, context.shape().dim(1), innerDim);
                x = crossAttn.forward(x, contextFeatures, contextFeatures, null).add(residual);
            } finally {
                projectedContext.release();
                if (contextFeatures != null) contextFeatures.release();
            }
        }

        residual = x;
        x = norm3.forward(x.reshape(batch * height * width, innerDim))
                .reshape(batch, height * width, innerDim);
        x = ff1.forward(x).silu();
        x = ff2.forward(x).add(residual);

        x = x.transpose().reshape(batch, innerDim, height, width);
        x = projOut.forward(x);
        return x.add(input);
    }

    public int inChannels() {
        return inChannels;
    }

    public int innerDim() {
        return innerDim;
    }

    public int contextDim() {
        return contextDim;
    }
}
