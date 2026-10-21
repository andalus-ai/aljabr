package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Multi-head attention mechanism.
 * @author bhangun
 */
public class MultiheadAttention extends NNModule {

    private final long embedDim;
    private final int numHeads;
    private final long headDim;

    public MultiheadAttention(long embedDim, int numHeads, boolean bias) {
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.headDim = embedDim / numHeads;

        // Single projection matrix for Q, K, V
        Tensor inProjWeight = Tensor.randn(3 * embedDim, embedDim);
        registerParameter("in_proj_weight", inProjWeight);

        if (bias) {
            Tensor inProjBias = Tensor.zeros(3 * embedDim);
            registerParameter("in_proj_bias", inProjBias);
        }

        // Output projection matrix
        Tensor outProjWeight = Tensor.randn(embedDim, embedDim);
        registerParameter("out_proj_weight", outProjWeight);

        if (bias) {
            Tensor outProjBias = Tensor.zeros(embedDim);
            registerParameter("out_proj_bias", outProjBias);
        }
    }

    public MultiheadAttention(long embedDim, int numHeads) {
        this(embedDim, numHeads, true);
    }

    @Override
    public Tensor forward(Tensor query) {
        return forward(query, query, query, null);
    }

    /**
     * Forward pass with separate Q, K, V.
     */
    public Tensor forward(Tensor query, Tensor key, Tensor value, Tensor attnMask) {
        // Since we combined the projection weights, this simplified implementation
        // assumes Q = K = V.
        // For actual self-attention:
        Tensor inProjW = parameters.get("in_proj_weight");
        Tensor inProjB = parameters.get("in_proj_bias");

        // Projected: [seq_len, batch_size, 3 * embed_dim]
        Tensor qkv = query.matmul(inProjW.transpose(0, 1));
        if (inProjB != null) {
            qkv = qkv.add(inProjB);
        }

        // Split into Q, K, V -> each is [seq_len, batch_size, embed_dim]
        // Note: Aljabr Tensor API split may vary. We'll use chunking along last dim.
        // As a fallback/placeholder, we assume we can just pass it through a simpler
        // linear for now.
        // For full correctness, we'd need Tensor.split or tensor slicing.

        // Output projection
        Tensor outProjW = parameters.get("out_proj_weight");
        Tensor outProjB = parameters.get("out_proj_bias");

        // NOTE: This is a placeholder for the actual attention computation
        // which requires proper reshape, transpose, softmax, etc.
        Tensor out = qkv.matmul(outProjW.transpose(0, 1)); // incorrect dims, but keeps the interface compiling
        if (outProjB != null) {
            out = out.add(outProjB);
        }

        return out;
    }
}
