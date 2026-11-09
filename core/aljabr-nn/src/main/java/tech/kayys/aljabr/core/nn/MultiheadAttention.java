package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Multi-head attention mechanism.
 */
public class MultiheadAttention extends NNModule {

    private final long embedDim;
    private final int numHeads;
    private final long headDim;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public MultiheadAttention(long embedDim, int numHeads, boolean bias) {
        this(null, embedDim, numHeads, bias);
    }

    public MultiheadAttention(ComputeBackend backend, long embedDim, int numHeads, boolean bias) {
        super(backend);
        if (numHeads <= 0 || embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by a positive numHeads");
        }
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;

        registerParameter("in_proj_weight", randomNormal(3 * embedDim, embedDim));
        if (bias) {
            registerParameter("in_proj_bias", zeros(3 * embedDim));
        }
        registerParameter("out_proj_weight", randomNormal(embedDim, embedDim));
        if (bias) {
            registerParameter("out_proj_bias", zeros(embedDim));
        }
    }

    /** @deprecated Use {@link #MultiheadAttention(ComputeBackend, long, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public MultiheadAttention(long embedDim, int numHeads) {
        this(null, embedDim, numHeads, true);
    }

    public MultiheadAttention(ComputeBackend backend, long embedDim, int numHeads) {
        this(backend, embedDim, numHeads, true);
    }

    @Override
    public Tensor forward(Tensor query) {
        return forward(query, query, query, null);
    }

    public Tensor forward(Tensor query, Tensor key, Tensor value, Tensor attnMask) {
        validateInputBackend(query);
        validateInputBackend(key);
        validateInputBackend(value);
        if (attnMask != null) {
            validateInputBackend(attnMask);
        }

        Tensor inProjW = parameters.get("in_proj_weight");
        Tensor inProjB = parameters.get("in_proj_bias");
        Tensor qkv = query.matmul(inProjW.transpose(0, 1));
        if (inProjB != null) {
            qkv = qkv.add(inProjB);
        }

        Tensor outProjW = parameters.get("out_proj_weight");
        Tensor outProjB = parameters.get("out_proj_bias");
        Tensor out = qkv.matmul(outProjW.transpose(0, 1));
        if (outProjB != null) {
            out = out.add(outProjB);
        }
        return out;
    }
}
