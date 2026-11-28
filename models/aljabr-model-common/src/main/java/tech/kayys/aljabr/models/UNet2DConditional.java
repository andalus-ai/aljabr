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
 * Configurable two-resolution conditional 2D UNet scaffold for diffusion denoising.
 *
 * <p>Channel widths and attention head count are supplied by
 * {@link UNet2DConditionalConfig}; model-family adapters remain responsible for
 * checkpoint-specific timestep and conditioning mapping. This is still a
 * deliberately limited scaffold, not a claim of Stable Diffusion checkpoint
 * compatibility or a complete arbitrary-depth UNet implementation.</p>
 */
public class UNet2DConditional extends NNModule {

    private final UNet2DConditionalConfig config;

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
    private final ResnetBlock2D up2;

    private final GroupNorm convNormOut;
    private final Conv2d convOut;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public UNet2DConditional(int inChannels, int outChannels, int tembDim, int contextDim) {
        this(null, UNet2DConditionalConfig.standard(inChannels, outChannels, tembDim, contextDim));
    }

    /** Preserves the original standard scaffold dimensions. */
    public UNet2DConditional(
            ComputeBackend backend, int inChannels, int outChannels, int tembDim, int contextDim) {
        this(backend, UNet2DConditionalConfig.standard(inChannels, outChannels, tembDim, contextDim));
    }

    public UNet2DConditional(ComputeBackend backend, UNet2DConditionalConfig config) {
        super(backend);
        this.config = Objects.requireNonNull(config, "config must not be null");

        int inChannels = config.inChannels();
        int outChannels = config.outChannels();
        int base = config.baseChannels();
        int expanded = config.expandedChannels();
        int timeDim = config.timeEmbeddingDim();
        int heads = config.attentionHeads();

        this.convIn = register("conv_in", new Conv2d(backend, inChannels, base, 3, 1, 1));
        this.timeEmbedding = register("time_embedding", new Sequential(
                backend,
                new Linear(backend, config.tembDim(), timeDim),
                new SiLU(backend),
                new Linear(backend, timeDim, timeDim)));

        this.down1 = register("down1_res", new ResnetBlock2D(backend, base, base, timeDim));
        this.downAttn1 = register("down1_attn",
                new SpatialTransformer(backend, base, heads, config.baseHeadDim(), config.contextDim()));

        this.downsample = register("downsample", new Downsample2d(backend, base, expanded));
        this.down2 = register("down2_res", new ResnetBlock2D(backend, expanded, expanded, timeDim));
        this.downAttn2 = register("down2_attn",
                new SpatialTransformer(backend, expanded, heads, config.expandedHeadDim(), config.contextDim()));

        this.midRes = register("mid_res", new ResnetBlock2D(backend, expanded, expanded, timeDim));
        this.midAttn = register("mid_attn",
                new SpatialTransformer(backend, expanded, heads, config.expandedHeadDim(), config.contextDim()));

        this.upsample1 = register("up1_upsample", new Upsample2d(backend, 2.0f));
        // Decoder features concatenate with the first-resolution encoder skip.
        this.up1 = register("up1_res", new ResnetBlock2D(backend, expanded + base, base, timeDim));
        this.upAttn1 = register("up1_attn",
                new SpatialTransformer(backend, base, heads, config.baseHeadDim(), config.contextDim()));
        this.up2 = register("up2_res", new ResnetBlock2D(backend, base, base, timeDim));

        this.convNormOut = register("conv_norm_out",
                new GroupNorm(backend, UNet2DConditionalConfig.NORMALIZATION_GROUPS, base));
        this.convOut = register("conv_out", new Conv2d(backend, base, outChannels, 3, 1, 1));
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
                || latents.shape().dim(1) != config.inChannels()
                || latents.shape().dim(2) < 2
                || latents.shape().dim(3) < 2) {
            throw new IllegalArgumentException(
                    "latents must have shape [N, " + config.inChannels() + ", H, W] with H and W >= 2");
        }
        if (timestep.shape().rank() != 2
                || timestep.shape().dim(0) != latents.shape().dim(0)
                || timestep.shape().dim(1) != config.tembDim()) {
            throw new IllegalArgumentException("timestep must have shape [N, " + config.tembDim() + "]");
        }
        if (context != null) {
            validateInputBackend(context);
            if (context.shape().rank() != 3
                    || context.shape().dim(0) != latents.shape().dim(0)
                    || context.shape().dim(2) != config.contextDim()) {
                throw new IllegalArgumentException(
                        "context must have shape [N, sequenceLength, " + config.contextDim() + "]");
            }
        }

        Tensor temb = timeEmbedding.forward(timestep);
        Tensor h = convIn.forward(latents);
        Tensor s1 = down1.forward(h, temb);
        s1 = downAttn1.forward(s1, context);

        h = downsample.forward(s1);
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

    public UNet2DConditionalConfig config() { return config; }
    public int inChannels() { return config.inChannels(); }
    public int outChannels() { return config.outChannels(); }
    public int tembDim() { return config.tembDim(); }
    public int contextDim() { return config.contextDim(); }
}
