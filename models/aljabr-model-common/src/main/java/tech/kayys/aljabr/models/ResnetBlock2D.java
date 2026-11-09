package tech.kayys.aljabr.models;

import java.util.Optional;
import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.GroupNorm;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.SiLU;
import tech.kayys.aljabr.core.nn.Conv2d;

/**
 * ResNet block used by diffusion UNets.
 *
 * <p>Performs normalization, activation, convolution, timestep embedding
 * injection, and a residual connection. The block and all child modules share
 * one explicitly selected compute backend.</p>
 */
public class ResnetBlock2D extends NNModule {

    private static final int NORM_GROUPS = 32;

    private final int inChannels;
    private final int outChannels;
    private final GroupNorm norm1;
    private final SiLU act1;
    private final Conv2d conv1;
    private final SiLU timeActivation;
    private final Linear timeEmbProj;
    private final GroupNorm norm2;
    private final SiLU act2;
    private final Conv2d conv2;
    private final Optional<Conv2d> residualProj;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public ResnetBlock2D(int inChannels, int outChannels, int tembChannels) {
        this(null, inChannels, outChannels, tembChannels);
    }

    public ResnetBlock2D(ComputeBackend backend, int inChannels, int outChannels, int tembChannels) {
        super(backend);
        if (inChannels <= 0 || outChannels <= 0 || tembChannels <= 0) {
            throw new IllegalArgumentException("channel dimensions must be positive");
        }
        if (inChannels % NORM_GROUPS != 0 || outChannels % NORM_GROUPS != 0) {
            throw new IllegalArgumentException("input and output channels must be divisible by " + NORM_GROUPS);
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;

        this.norm1 = register("norm1", new GroupNorm(backend, NORM_GROUPS, inChannels));
        this.act1 = register("act1", new SiLU(backend));
        this.conv1 = register("conv1", new Conv2d(backend, inChannels, outChannels, 3, 1, 1));

        this.timeActivation = register("time_activation", new SiLU(backend));
        this.timeEmbProj = register("time_emb_proj", new Linear(backend, tembChannels, outChannels));

        this.norm2 = register("norm2", new GroupNorm(backend, NORM_GROUPS, outChannels));
        this.act2 = register("act2", new SiLU(backend));
        this.conv2 = register("conv2", new Conv2d(backend, outChannels, outChannels, 3, 1, 1));

        this.residualProj = inChannels != outChannels
                ? Optional.of(register("conv_shortcut", new Conv2d(backend, inChannels, outChannels, 1, 1, 0)))
                : Optional.empty();
    }

    @Override
    public Tensor forward(Tensor input) {
        return forward(input, null);
    }

    /**
     * Forward pass with optional timestep embedding injection.
     *
     * @param input [N, inChannels, H, W]
     * @param temb [N, tembChannels], or null when no embedding is supplied
     * @return [N, outChannels, H, W]
     */
    public Tensor forward(Tensor input, Tensor temb) {
        Objects.requireNonNull(input, "input must not be null");
        validateInputBackend(input);
        if (input.shape().rank() != 4 || input.shape().dim(1) != inChannels) {
            throw new IllegalArgumentException(
                    "ResnetBlock2D expected [N, " + inChannels + ", H, W] input but got " + input.shape());
        }
        if (temb != null) {
            validateInputBackend(temb);
            if (temb.shape().rank() != 2 || temb.shape().dim(0) != input.shape().dim(0)) {
                throw new IllegalArgumentException("timestep embedding must have shape [N, tembChannels]");
            }
        }

        Tensor h = act1.forward(norm1.forward(input));
        h = conv1.forward(h);

        if (temb != null) {
            Tensor activatedTemb = timeActivation.forward(temb);
            Tensor projectedTemb = timeEmbProj.forward(activatedTemb);
            Tensor broadcastTemb = projectedTemb.unsqueeze(-1).unsqueeze(-1);
            h = h.add(broadcastTemb);
        }

        h = act2.forward(norm2.forward(h));
        h = conv2.forward(h);

        Tensor shortcut = residualProj.map(projection -> projection.forward(input)).orElse(input);
        return h.add(shortcut);
    }

    public int inChannels() { return inChannels; }
    public int outChannels() { return outChannels; }

    @Override
    public String toString() {
        return "ResnetBlock2D(inChannels=" + inChannels + ", outChannels=" + outChannels + ")";
    }
}
