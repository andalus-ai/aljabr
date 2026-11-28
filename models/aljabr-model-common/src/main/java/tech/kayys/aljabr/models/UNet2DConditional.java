package tech.kayys.aljabr.models;

import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.GroupNorm;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.Sequential;
import tech.kayys.aljabr.core.nn.Conv2d;
import tech.kayys.aljabr.core.nn.Downsample2d;
import tech.kayys.aljabr.core.nn.SiLU;

/**
 * Configurable two-resolution conditional 2D UNet scaffold for diffusion denoising.
 *
 * <p>Encoder, bottleneck, and decoder computation is delegated to reusable
 * stage blocks. This remains a two-resolution scaffold, not an arbitrary-depth
 * UNet or a claim of Stable Diffusion checkpoint compatibility.</p>
 */
public class UNet2DConditional extends NNModule {

    private final UNet2DConditionalConfig config;

    private final Conv2d convIn;
    private final Sequential timeEmbedding;
    private final UNetEncoderBlock2D down1;
    private final Downsample2d downsample;
    private final UNetEncoderBlock2D down2;
    private final UNetMidBlock2D mid;
    private final UNetDecoderBlock2D up1;
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

        int base = config.baseChannels();
        int expanded = config.expandedChannels();
        int timeDim = config.timeEmbeddingDim();
        int heads = config.attentionHeads();

        this.convIn = register("conv_in",
                new Conv2d(backend, config.inChannels(), base, 3, 1, 1));
        this.timeEmbedding = register("time_embedding", new Sequential(
                backend,
                new Linear(backend, config.tembDim(), timeDim),
                new SiLU(backend),
                new Linear(backend, timeDim, timeDim)));

        this.down1 = register("down1", new UNetEncoderBlock2D(
                backend, base, base, timeDim, heads, config.baseHeadDim(), config.contextDim()));
        this.downsample = register("downsample", new Downsample2d(backend, base, expanded));
        this.down2 = register("down2", new UNetEncoderBlock2D(
                backend, expanded, expanded, timeDim, heads,
                config.expandedHeadDim(), config.contextDim()));

        this.mid = register("mid", new UNetMidBlock2D(
                backend, expanded, timeDim, heads,
                config.expandedHeadDim(), config.contextDim()));

        this.up1 = register("up1", new UNetDecoderBlock2D(
                backend, expanded, base, base, timeDim, heads,
                config.baseHeadDim(), config.contextDim()));

        this.convNormOut = register("conv_norm_out",
                new GroupNorm(backend, UNet2DConditionalConfig.NORMALIZATION_GROUPS, base));
        this.convOut = register("conv_out",
                new Conv2d(backend, base, config.outChannels(), 3, 1, 1));
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
        Tensor hidden = convIn.forward(latents);

        // Each forward pass owns its stack; feature tensors remain owned by the
        // tensor execution path and are borrowed by this bookkeeping object.
        UNetSkipConnections skips = new UNetSkipConnections();
        Tensor firstResolution = down1.forward(hidden, temb, context);
        skips.push(firstResolution);

        hidden = downsample.forward(firstResolution);
        hidden = down2.forward(hidden, temb, context);
        hidden = mid.forward(hidden, temb, context);

        Tensor decoderSkip = skips.pop();
        skips.requireEmpty();
        hidden = up1.forward(hidden, decoderSkip, temb, context);

        hidden = convNormOut.forward(hidden).silu();
        return convOut.forward(hidden);
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
