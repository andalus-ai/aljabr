package tech.kayys.aljabr.models;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.GroupNorm;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.Sequential;
import tech.kayys.aljabr.core.nn.Conv2d;
import tech.kayys.aljabr.core.nn.Upsample2d;
import tech.kayys.aljabr.core.nn.SiLU;

import java.util.ArrayList;
import java.util.List;

/**
 * Conditional 2D UNet for Stable Diffusion denoising.
 * <p>
 * This architecture performs iterative denoising of latents in a 
 * compressed spatial domain (typically 64x64 for 512x512 images).
 */
public class UNet2DConditional extends NNModule {

    private final Conv2d convIn;
    private final Sequential timeEmbedding;
    
    // Simplifed Down-Blocks for the demo architecture
    private final ResnetBlock2D down1;
    private final SpatialTransformer downAttn1;
    
    private final ResnetBlock2D down2;
    private final SpatialTransformer downAttn2;

    // Mid-Block
    private final ResnetBlock2D midRes;
    private final SpatialTransformer midAttn;
    
    // Up-Blocks
    private final Upsample2d upsample1;
    private final ResnetBlock2D up1;
    private final SpatialTransformer upAttn1;
    
    private final Upsample2d upsample2;
    private final ResnetBlock2D up2;
    
    // Out
    private final GroupNorm convNormOut;
    private final Conv2d convOut;

    public UNet2DConditional(int inChannels, int outChannels, int tembDim, int contextDim) {
        this.convIn = register("conv_in", new Conv2d(inChannels, 320, 3, 1, 1));
        
        this.timeEmbedding = register("time_embedding", new Sequential(
            new Linear(tembDim, tembDim * 4),
            new SiLU(),
            new Linear(tembDim * 4, tembDim * 4)
        ));

        // Down path
        this.down1 = register("down1_res", new ResnetBlock2D(320, 320, tembDim * 4));
        this.downAttn1 = register("down1_attn", new SpatialTransformer(320, 8, 40, contextDim));
        
        this.down2 = register("down2_res", new ResnetBlock2D(320, 640, tembDim * 4));
        this.downAttn2 = register("down2_attn", new SpatialTransformer(640, 8, 80, contextDim));

        // Mid path
        this.midRes = register("mid_res", new ResnetBlock2D(640, 640, tembDim * 4));
        this.midAttn = register("mid_attn", new SpatialTransformer(640, 8, 80, contextDim));

        // Up path
        this.upsample1 = register("up1_upsample", new Upsample2d(2.0f));
        this.up1 = register("up1_res", new ResnetBlock2D(640, 320, tembDim * 4));
        this.upAttn1 = register("up1_attn", new SpatialTransformer(320, 8, 40, contextDim));
        
        this.upsample2 = register("up2_upsample", new Upsample2d(2.0f));
        this.up2 = register("up2_res", new ResnetBlock2D(320, 320, tembDim * 4));

        this.convNormOut = register("conv_norm_out", new GroupNorm(32, 320));
        this.convOut = register("conv_out", new Conv2d(320, outChannels, 3, 1, 1));
    }

    /**
     * Denoising forward pass.
     * 
     * @param latents    [N, 4, H, W] noisy latent image
     * @param timestep   [N] indices of time steps
     * @param context    [N, 77, 768] text embeddings from CLIP
     * @return           [N, 4, H, W] predicted noise
     */
    public Tensor forward(Tensor latents, Tensor timestep, Tensor context) {
        // 1. Time Embedding
        Tensor temb = timeEmbedding.forward(timestep);

        // 2. Input convolution
        Tensor h = convIn.forward(latents);
        
        // 3. Down blocks (storing residuals for skip-connections)
        Tensor s1 = down1.forward(h, temb);
        s1 = downAttn1.forward(s1, context);
        
        Tensor s2 = down2.forward(s1, temb);
        s2 = downAttn2.forward(s2, context);

        // 4. Mid block
        h = midRes.forward(s2, temb);
        h = midAttn.forward(h, context);

        // 5. Up blocks with skip-connections (simplified concat)
        h = upsample1.forward(h);
        h = h.add(s2); // Residual join
        h = up1.forward(h, temb);
        h = upAttn1.forward(h, context);
        
        h = upsample2.forward(h);
        h = h.add(s1);
        h = up2.forward(h, temb);

        // 6. Output
        h = convNormOut.forward(h).silu();
        return convOut.forward(h);
    }

    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException("UNet requires timestep and context tensors");
    }
}
