package tech.kayys.aljabr.models;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.GroupNorm;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.SiLU;
import tech.kayys.aljabr.core.nn.Conv2d;

import java.util.Optional;

/**
 * Resnet block used in Stable Diffusion UNet.
 * <p>
 * Performs residual connection with time embedding injection.
 * Architecture:
 * 1. Norm1 -> SiLU -> Conv1
 * 2. Time Embedding -> SiLU -> Linear (projection to channels)
 * 3. Add Time Embedding to Conv1 output
 * 4. Norm2 -> SiLU -> Conv2
 * 5. Add residual (with optional projection if channels differ)
 */
public class ResnetBlock2D extends NNModule {

    private final GroupNorm norm1;
    private final Conv2d conv1;
    
    private final Linear timeEmbProj;
    
    private final GroupNorm norm2;
    private final Conv2d conv2;
    
    private final Optional<Conv2d> residualProj;

    public ResnetBlock2D(int inChannels, int outChannels, int tembChannels) {
        this.norm1 = register("norm1", new GroupNorm(32, inChannels));
        this.conv1 = register("conv1", new Conv2d(inChannels, outChannels, 3, 1, 1));
        
        this.timeEmbProj = register("time_emb_proj", new Linear(tembChannels, outChannels));
        
        this.norm2 = register("norm2", new GroupNorm(32, outChannels));
        this.conv2 = register("conv2", new Conv2d(outChannels, outChannels, 3, 1, 1));
        
        if (inChannels != outChannels) {
            this.residualProj = Optional.of(register("conv_shortcut", new Conv2d(inChannels, outChannels, 1, 1, 0)));
        } else {
            this.residualProj = Optional.empty();
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        // Forward through time embedding path is handled by the caller or injected here
        return forward(input, null);
    }

    /**
     * Forward pass with time-step embedding injection.
     * 
     * @param input    [N, C, H, W]
     * @param temb     [N, tembChannels]
     * @return         [N, outChannels, H, W]
     */
    public Tensor forward(Tensor input, Tensor temb) {
        Tensor h = norm1.forward(input).silu();
        h = conv1.forward(h);

        if (temb != null) {
            // Project and reshape temb: [N, outChannels] -> [N, outChannels, 1, 1]
            Tensor t = timeEmbProj.forward(temb.silu()).unsqueeze(-1).unsqueeze(-1);
            h = h.add(t);
        }

        h = norm2.forward(h).silu();
        h = conv2.forward(h);

        Tensor shortcut = residualProj.map(p -> p.forward(input)).orElse(input);
        
        return h.add(shortcut);
    }

    @Override
    public String toString() {
        return "ResnetBlock2D(in=" + norm1 + ", out=" + conv2 + ")";
    }
}
