package tech.kayys.aljabr.core.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Backend-neutral grouped-query attention (GQA).
 *
 * <p>Queries may have more heads than keys/values. Each KV head is expanded to
 * the corresponding number of query heads before dispatching to the backend's
 * ordinary attention primitive. This is intentionally a correctness-first
 * reference implementation: backends can later add a native grouped-attention
 * kernel without changing model code.</p>
 *
 * <p>Inputs use {@code [batch, heads, sequence, headDim]}. The output has the
 * same shape as the query tensor.</p>
 */
public final class GroupedQueryAttention extends NNModule {

    private final int queryHeads;
    private final int keyValueHeads;
    private final int headDim;

    public GroupedQueryAttention(ComputeBackend backend, int queryHeads,
            int keyValueHeads, int headDim) {
        super(Objects.requireNonNull(backend, "backend must not be null"));
        if (queryHeads <= 0 || keyValueHeads <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("head counts and headDim must be positive");
        }
        if (queryHeads % keyValueHeads != 0) {
            throw new IllegalArgumentException(
                    "queryHeads must be divisible by keyValueHeads");
        }
        this.queryHeads = queryHeads;
        this.keyValueHeads = keyValueHeads;
        this.headDim = headDim;
    }

    /**
     * Computes non-causal scaled dot-product GQA.
     */
    @Override
    public Tensor forward(Tensor query) {
        throw new UnsupportedOperationException(
                "Use forward(query, key, value, mask, causal) for attention inputs");
    }

    public Tensor forward(Tensor query, Tensor key, Tensor value,
            Tensor additiveMask, boolean causal) {
        validateInputBackend(query);
        validateInputBackend(key);
        validateInputBackend(value);
        if (additiveMask != null) validateInputBackend(additiveMask);
        validateHeads("query", query, queryHeads);
        validateHeads("key", key, keyValueHeads);
        validateHeads("value", value, keyValueHeads);

        if (key.shape().dim(0) != query.shape().dim(0)
                || value.shape().dim(0) != query.shape().dim(0)
                || key.shape().dim(2) != value.shape().dim(2)
                || key.shape().dim(3) != headDim
                || value.shape().dim(3) != headDim
                || query.shape().dim(3) != headDim) {
            throw new IllegalArgumentException(
                    "query/key/value must have matching batch and headDim with compatible sequence lengths");
        }
        if (additiveMask != null
                && (additiveMask.shape().rank() != 2
                || additiveMask.shape().dim(0) != query.shape().dim(2)
                || additiveMask.shape().dim(1) != key.shape().dim(2))) {
            throw new IllegalArgumentException(
                    "attention mask must have shape [querySequence, keySequence]");
        }

        Tensor expandedKey = null;
        Tensor expandedValue = null;
        try {
            expandedKey = repeatHeads(key);
            expandedValue = repeatHeads(value);
            return query.backend().attention(query, expandedKey, expandedValue, additiveMask, causal);
        } finally {
            if (expandedValue != null) expandedValue.release();
            if (expandedKey != null) expandedKey.release();
        }
    }

    private Tensor repeatHeads(Tensor input) {
        int copiesPerHead = queryHeads / keyValueHeads;
        List<Tensor> heads = input.split(1, keyValueHeads);
        List<Tensor> repeated = new ArrayList<>(queryHeads);
        try {
            for (Tensor head : heads) {
                for (int copy = 0; copy < copiesPerHead; copy++) {
                    repeated.add(head);
                }
            }
            return Tensor.cat(repeated, 1);
        } finally {
            heads.forEach(Tensor::release);
        }
    }

    private void validateHeads(String name, Tensor tensor, int expectedHeads) {
        if (tensor.shape().rank() != 4
                || tensor.shape().dim(1) != expectedHeads
                || tensor.shape().dim(3) != headDim) {
            throw new IllegalArgumentException(name + " must have shape [batch, "
                    + expectedHeads + ", sequence, " + headDim + "]");
        }
    }

    public int queryHeads() {
        return queryHeads;
    }

    public int keyValueHeads() {
        return keyValueHeads;
    }

    public int headDim() {
        return headDim;
    }

    public int queriesPerKeyValueHead() {
        return queryHeads / keyValueHeads;
    }
}
