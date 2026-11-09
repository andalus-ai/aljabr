package tech.kayys.aljabr.core.nn;

import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Multi-head scaled dot-product attention for batch-first sequence tensors.
 *
 * <p>Inputs use {@code [batch, sequence, embedding]} layout. Query, key, and
 * value projections are independent learned linear layers; attention is
 * evaluated in {@code [batch, heads, sequence, head_dim]} layout.</p>
 */
public class MultiheadAttention extends NNModule {

    private final long embedDim;
    private final int numHeads;
    private final long headDim;
    private final Linear queryProjection;
    private final Linear keyProjection;
    private final Linear valueProjection;
    private final Linear outputProjection;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public MultiheadAttention(long embedDim, int numHeads, boolean bias) {
        this(null, embedDim, numHeads, bias);
    }

    public MultiheadAttention(ComputeBackend backend, long embedDim, int numHeads, boolean bias) {
        super(backend);
        if (embedDim <= 0 || numHeads <= 0 || embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be positive and divisible by a positive numHeads");
        }
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.queryProjection = registerModule("query", new Linear(backend, embedDim, embedDim, bias));
        this.keyProjection = registerModule("key", new Linear(backend, embedDim, embedDim, bias));
        this.valueProjection = registerModule("value", new Linear(backend, embedDim, embedDim, bias));
        this.outputProjection = registerModule("output", new Linear(backend, embedDim, embedDim, bias));
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

    /** Computes batch-first attention without causal restriction. */
    public Tensor forward(Tensor query, Tensor key, Tensor value, Tensor attnMask) {
        return forward(query,key,value,attnMask,false);
    }

    /** Additive mask shape is [querySequence,keySequence]; negative infinity suppresses a position. */
    public Tensor forward(Tensor query, Tensor key, Tensor value, Tensor attnMask, boolean causal) {
        Objects.requireNonNull(query,"query must not be null");
        Objects.requireNonNull(key,"key must not be null");
        Objects.requireNonNull(value,"value must not be null");
        validateInputBackend(query); validateInputBackend(key); validateInputBackend(value);
        if(attnMask!=null) validateInputBackend(attnMask);
        requireSequenceShape("query",query); requireSequenceShape("key",key); requireSequenceShape("value",value);
        if(query.shape().dim(0)!=key.shape().dim(0) || key.shape().dim(0)!=value.shape().dim(0)
                || key.shape().dim(1)!=value.shape().dim(1))
            throw new IllegalArgumentException("query/key/value batch sizes and key/value sequence lengths must match");
        long batch=query.shape().dim(0), queryLength=query.shape().dim(1), keyLength=key.shape().dim(1);
        if(attnMask!=null && (attnMask.shape().rank()!=2 || attnMask.shape().dim(0)!=queryLength || attnMask.shape().dim(1)!=keyLength))
            throw new IllegalArgumentException("attention mask must have shape [querySequence, keySequence]");
        Tensor q=projectAndSplit(queryProjection,query,batch,queryLength);
        Tensor k=projectAndSplit(keyProjection,key,batch,keyLength);
        Tensor v=projectAndSplit(valueProjection,value,batch,keyLength);
        Tensor attended=null, merged=null, flattened=null;
        try {
            attended=query.backend().attention(q,k,v,attnMask,causal);
            merged=attended.transpose(1,2).reshape(batch,queryLength,embedDim);
            flattened=merged.reshape(Math.multiplyExact(batch,queryLength),embedDim);
            Tensor projected=outputProjection.forward(flattened);
            try { return projected.reshape(batch,queryLength,embedDim); }
            finally { projected.release(); }
        } finally {
            if(flattened!=null) flattened.release();
            if(merged!=null) merged.release();
            if(attended!=null) attended.release();
            q.release(); k.release(); v.release();
        }
    }

    private Tensor projectAndSplit(Linear projection, Tensor input, long batch, long sequenceLength) {
        Tensor flattened = input.reshape(Math.multiplyExact(batch, sequenceLength), embedDim);
        Tensor projected = null;
        Tensor shaped = null;
        try {
            projected = projection.forward(flattened);
            shaped = projected.reshape(batch, sequenceLength, numHeads, headDim);
            return shaped.transpose(1, 2);
        } finally {
            if (shaped != null) shaped.release();
            if (projected != null) projected.release();
            flattened.release();
        }
    }

    private void requireSequenceShape(String name, Tensor tensor) {
        if (tensor.shape().rank() != 3 || tensor.shape().dim(2) != embedDim) {
            throw new IllegalArgumentException(name + " must have shape [batch, sequence, " + embedDim + "]");
        }
    }

    public long embedDim() {
        return embedDim;
    }

    public int numHeads() {
        return numHeads;
    }

    public long headDim() {
        return headDim;
    }
}
