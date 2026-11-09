package tech.kayys.aljabr.models;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.nn.GroupNorm;
import tech.kayys.aljabr.core.nn.LayerNorm;
import tech.kayys.aljabr.core.nn.Conv2d;
import tech.kayys.aljabr.core.nn.MultiheadAttention;
import tech.kayys.aljabr.core.nn.Linear;

/**
 * Spatial Transformer block for Stable Diffusion.
 * <p>
 * Bridges the gap between spatial CNN features and semantic text embeddings
 * through Multi-Head Cross Attention.
 */
public class SpatialTransformer extends NNModule {

    private final GroupNorm norm;
    private final Conv2d projIn;
    
    // Transformer block components
    private final LayerNorm norm1;
    private final MultiheadAttention selfAttn;
    
    private final LayerNorm norm2;
    private final MultiheadAttention crossAttn;
    
    private final LayerNorm norm3;
    private final Linear ff1;
    private final Linear ff2;
    
    private final Conv2d projOut;

    public SpatialTransformer(int inChannels, int numHeads, int headDim, int contextDim) {
        int innerDim = numHeads * headDim;
        
        this.norm = register("norm", new GroupNorm(32, inChannels));
        this.projIn = register("proj_in", new Conv2d(inChannels, innerDim, 1, 1, 0));
        
        this.norm1 = register("norm1", new LayerNorm(innerDim));
        this.selfAttn = register("attn1", new MultiheadAttention(innerDim, numHeads));
        
        this.norm2 = register("norm2", new LayerNorm(innerDim));
        this.crossAttn = register("attn2", new MultiheadAttention(innerDim, numHeads));
        
        this.norm3 = register("norm3", new LayerNorm(innerDim));
        this.ff1 = register("ff_net_primary", new Linear(innerDim, innerDim * 4));
        this.ff2 = register("ff_net_secondary", new Linear(innerDim * 4, innerDim));
        
        this.projOut = register("proj_out", new Conv2d(innerDim, inChannels, 1, 1, 0));
    }

    @Override
    public Tensor forward(Tensor input) {
        return forward(input, null);
    }

    /**
     * Differentiable forward pass with optional cross-attention context.
     * 
     * @param input    [N, C, H, W] spatial features
     * @param context  [N, seqLen, contextDim] text embeddings from CLIP
     * @return         [N, C, H, W] conditioned spatial features
     */
    public Tensor forward(Tensor input, Tensor context) {
        long[] shape = input.shape().dims();
        int N = (int) shape[0];
        int C = (int) shape[1];
        int H = (int) shape[2];
        int W = (int) shape[3];

        // 1. Initial projection to inner transformer space
        Tensor x = norm.forward(input);
        x = projIn.forward(x);
        
        // Reshape to sequence: [N, innerDim, H, W] -> [N, H*W, innerDim]
        int innerDim = (int) x.shape().dim(1);
        x = x.reshape(N, innerDim, H * W).transpose(); // [N, H*W, innerDim]

        // 2. Self-Attention Block
        Tensor residual = x;
        x = norm1.forward(x.reshape(N * H * W, innerDim)).reshape(N, H * W, innerDim);
        x = selfAttn.forward(x).add(residual);

        // 3. Cross-Attention Block (Text Conditioning)
        if (context != null) {
            residual = x;
            x = norm2.forward(x.reshape(N * H * W, innerDim)).reshape(N, H * W, innerDim);
            x = crossAttn.forward(x, context, context, null).add(residual);
        }

        // 4. Feed-Forward Block
        residual = x;
        x = norm3.forward(x.reshape(N * H * W, innerDim)).reshape(N, H * W, innerDim);
        x = ff1.forward(x).silu();
        x = ff2.forward(x).add(residual);

        // 5. Project back to spatial dimensions
        x = x.transpose().reshape(N, innerDim, H, W);
        x = projOut.forward(x);

        return x.add(input);
    }
}
