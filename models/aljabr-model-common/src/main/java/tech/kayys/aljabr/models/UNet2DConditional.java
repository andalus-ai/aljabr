package tech.kayys.aljabr.models;

import java.util.List;
import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorOps;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.GroupNorm;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.Sequential;
import tech.kayys.aljabr.core.nn.Conv2d;
import tech.kayys.aljabr.core.nn.Downsample2d;
import tech.kayys.aljabr.core.nn.Upsample2d;
import tech.kayys.aljabr.core.nn.SiLU;

/**
 * Simplified conditional 2D UNet scaffold for diffusion denoising.
 *
 * <p>The model and all child modules are owned by one explicit compute backend.
 * This class is not yet a complete Stable Diffusion UNet: its spatial
 * downsampling/upsampling and several attention operations remain simplified
 * or placeholders.</p>
 */
public class UNet2DConditional extends NNModule {

    private static final int BASE_CHANNELS = 320;
    private static final int EXPANDED_CHANNELS = 640;

    private final int inChannels;
    private final int outChannels;
    private final int tembDim;
    private final int contextDim;

    private final Conv2d convIn;
    private final Sequential timeEmbedding;

    private final ResnetBlock2D down1;
    private final SpatialTransformer downAttn1;
    private final Downsample2d downsample;
    private final ResnetBlock2D down2;
    private final SpatialTransformer downAttn2;

    private final ResnetBlock2D midRes;
    private final SpatialTransformer midAttn;

    private final Upsample2d upsample1;
    private final ResnetBlock2D up1;
    private final SpatialTransformer upAttn1;
    private final Upsample2d upsample2;
    private final ResnetBlock2D up2;

    private final GroupNorm convNormOut;
    private final Conv2d convOut;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public UNet2DConditional(int inChannels, int outChannels, int tembDim, int contextDim) {
        this(null, inChannels, outChannels, tembDim, contextDim);
    }

    public UNet2DConditional(
            ComputeBackend backend,
            int inChannels,
            int outChannels,
            int tembDim,
            int contextDim) {
        super(backend);
        if (inChannels <= 0 || outChannels <= 0 || tembDim <= 0 || contextDim <= 0) {
            throw new IllegalArgumentException("UNet dimensions must be positive");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.tembDim = tembDim;
        this.contextDim = contextDim;

        this.convIn = register("conv_in",
                new Conv2d(backend, inChannels, BASE_CHANNELS, 3, 1, 1));
        this.timeEmbedding = register("time_embedding", new Sequential(
                backend,
                new Linear(backend, tembDim, tembDim * 4L),
                new SiLU(backend),
                new Linear(backend, tembDim * 4L, tembDim * 4L)));

        this.down1 = register("down1_res",
                new ResnetBlock2D(backend, BASE_CHANNELS, BASE_CHANNELS, tembDim * 4));
        this.downAttn1 = register("down1_attn",
                new SpatialTransformer(backend, BASE_CHANNELS, 8, 40, contextDim));

        this.downsample = register("downsample",
                new Downsample2d(backend, BASE_CHANNELS, EXPANDED_CHANNELS));
        this.down2 = register("down2_res",
                new ResnetBlock2D(backend, EXPANDED_CHANNELS, EXPANDED_CHANNELS, tembDim * 4));
        this.downAttn2 = register("down2_attn",
                new SpatialTransformer(backend, EXPANDED_CHANNELS, 8, 80, contextDim));

        this.midRes = register("mid_res",
                new ResnetBlock2D(backend, EXPANDED_CHANNELS, EXPANDED_CHANNELS, tembDim * 4));
        this.midAttn = register("mid_attn",
                new SpatialTransformer(backend, EXPANDED_CHANNELS, 8, 80, contextDim));

        this.upsample1 = register("up1_upsample", new Upsample2d(backend, 2.0f));
        // 640 decoder channels concatenate with the 320-channel encoder skip.
        this.up1 = register("up1_res",
                new ResnetBlock2D(backend, EXPANDED_CHANNELS + BASE_CHANNELS, BASE_CHANNELS, tembDim * 4));
        this.upAttn1 = register("up1_attn",
                new SpatialTransformer(backend, BASE_CHANNELS, 8, 40, contextDim));

        this.up2 = register("up2_res",
                new ResnetBlock2D(backend, BASE_CHANNELS, BASE_CHANNELS, tembDim * 4));

        this.convNormOut = register("conv_norm_out",
                new GroupNorm(backend, 32, BASE_CHANNELS));
        this.convOut = register("conv_out",
                new Conv2d(backend, BASE_CHANNELS, outChannels, 3, 1, 1));
    }

    /**
     * Denoising forward pass.
     *
     * @param latents noisy latent image with shape [N, inChannels, H, W]
     * @param timestep timestep embedding input with shape [N, tembDim]
     * @param context optional conditioning with shape [N, sequenceLength, contextDim]
     * @return predicted noise with shape [N, outChannels, H, W]
     */
    public Tensor forward(Tensor latents, Tensor timestep, Tensor context) {
        Objects.requireNonNull(latents, "latents must not be null");
        Objects.requireNonNull(timestep, "timestep must not be null");
        validateInputBackend(latents);
        validateInputBackend(timestep);

        if (latents.shape().rank() != 4
                || latents.shape().dim(1) != inChannels
                || latents.shape().dim(2) < 2
                || latents.shape().dim(3) < 2) {
            throw new IllegalArgumentException(
                    "latents must have shape [N, " + inChannels + ", H, W] with H and W >= 2");
        }
        if (timestep.shape().rank() != 2
                || timestep.shape().dim(0) != latents.shape().dim(0)
                || timestep.shape().dim(1) != tembDim) {
            throw new IllegalArgumentException(
                    "timestep must have shape [N, " + tembDim + "]");
        }
        if (context != null) {
            validateInputBackend(context);
            if (context.shape().rank() != 3
                    || context.shape().dim(0) != latents.shape().dim(0)
                    || context.shape().dim(2) != contextDim) {
                throw new IllegalArgumentException(
                        "context must have shape [N, sequenceLength, " + contextDim + "]");
            }
        }

        Tensor temb = timeEmbedding.forward(timestep);

        Tensor h = convIn.forward(latents);
        Tensor s1 = down1.forward(h, temb);
        s1 = downAttn1.forward(s1, context);

        Tensor h = downsample.forward(s1);
        Tensor s2 = down2.forward(h, temb);
        s2 = downAttn2.forward(s2, context);

        h = midRes.forward(s2, temb);
        h = midAttn.forward(h, context);

        h = upsample1.forwardTo(h, s1);
        h = TensorOps.cat(List.of(h, s1), 1);
        h = up1.forward(h, temb);
        h = upAttn1.forward(h, context);
        h = up2.forward(h, temb);

        h = convNormOut.forward(h).silu();
        return convOut.forward(h);
    }

    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException(
                "UNet2DConditional requires latents, timestep embeddings, and optional context");
    }

    public int inChannels() { return inChannels; }
    public int outChannels() { return outChannels; }
    public int tembDim() { return tembDim; }
    public int contextDim() { return contextDim; }
}
