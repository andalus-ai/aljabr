package tech.kayys.aljabr.backend.hat;

import java.lang.foreign.Arena;
import java.util.Objects;

import optkl.util.carriers.ArenaAndLookupCarrier;
import hat.Accelerator.Compute;
import jdk.incubator.code.Reflect;
import tech.kayys.aljabr.cache.KVCacheReadView;

/**
 * Optional staged HAT consumer for a borrowed logical KV read view.
 *
 * <p>K/V are copied from the format-neutral read view into temporary HAT
 * {@code F32Array} buffers for each invocation. The lease remains open through
 * dispatch and copy-out. This deliberately does not consume Metal/native
 * device handles and does not claim zero-copy.</p>
 *
 * <p>The accelerator is reused and dispatch is serialized because the current
 * Babylon HAT Java backends cache mutable compute contexts. The upstream API
 * does not expose an accelerator close contract.</p>
 */
public final class HatKVCacheAttentionConsumer {
    private final Object executionLock = new Object();
    private final hat.Accelerator accelerator;

    public HatKVCacheAttentionConsumer() {
        this(new hat.Accelerator(java.lang.invoke.MethodHandles.lookup()));
    }

    HatKVCacheAttentionConsumer(hat.Accelerator accelerator) {
        this.accelerator = Objects.requireNonNull(accelerator, "accelerator");
    }

    /**
     * Runs one-query GQA decode attention against the visible prefix.
     *
     * @param lease borrowed logical cache view; remains caller-owned
     * @param layer cache layer
     * @param query row-major {@code [queryHeads, headDimension]}
     * @param queryHeads number of query heads
     * @param queryPosition absolute query position, used only for causal mode
     * @param causal whether future cache positions must be excluded
     * @param scale attention score scale (typically {@code 1/sqrt(headDimension)})
     * @return newly allocated row-major {@code [queryHeads, headDimension]} output
     */
    public float[] decode(
            KVCacheReadView lease,
            int layer,
            float[] query,
            int queryHeads,
            int queryPosition,
            boolean causal,
            float scale) {
        Objects.requireNonNull(lease, "lease");
        Objects.requireNonNull(query, "query");
        if (queryHeads <= 0 || !Float.isFinite(scale) || scale <= 0.0f) {
            throw new IllegalArgumentException("queryHeads and scale must be positive and finite");
        }

        int sequenceLength = lease.visibleSequenceLength();
        int kvHeads = lease.layout().kvHeads(layer);
        int headDimension = lease.layout().headDimension(layer);
        if (sequenceLength <= 0) {
            throw new IllegalArgumentException("KV cache view must expose at least one position");
        }
        if (queryHeads % kvHeads != 0) {
            throw new IllegalArgumentException("queryHeads must be divisible by kvHeads for GQA");
        }
        if (query.length != Math.multiplyExact(queryHeads, headDimension)) {
            throw new IllegalArgumentException("query length does not match [queryHeads, headDimension]");
        }
        if (causal && (queryPosition < 0 || queryPosition >= sequenceLength)) {
            throw new IllegalArgumentException("causal queryPosition must be inside the visible prefix");
        }

        int lastPosition = causal ? queryPosition : sequenceLength - 1;
        int cacheElements = Math.multiplyExact(sequenceLength,
                Math.multiplyExact(kvHeads, headDimension));
        float[] keyValues = new float[cacheElements];
        float[] valueValues = new float[cacheElements];
        lease.readKeyRange(layer, 0, sequenceLength, keyValues, 0);
        lease.readValueRange(layer, 0, sequenceLength, valueValues, 0);

        float[] output = new float[Math.multiplyExact(queryHeads, headDimension)];
        synchronized (executionLock) {
            try (Arena arena = Arena.ofShared()) {
                ArenaAndLookupCarrier carrier =
                        ArenaAndLookupCarrier.of(accelerator.lookup(), arena);
                hat.buffer.F32Array hatQuery = hat.buffer.F32Array.create(carrier, query.length);
                hat.buffer.F32Array hatKeys = hat.buffer.F32Array.create(carrier, cacheElements);
                hat.buffer.F32Array hatValues = hat.buffer.F32Array.create(carrier, cacheElements);
                hat.buffer.F32Array hatOutput = hat.buffer.F32Array.create(carrier, output.length);

                hatQuery.copyFrom(query);
                hatKeys.copyFrom(keyValues);
                hatValues.copyFrom(valueValues);
                accelerator.compute((@Reflect Compute) cc ->
                        HatKVCacheAttentionKernel.dispatch(cc, hatQuery, hatKeys,
                                hatValues, hatOutput, queryHeads, kvHeads,
                                headDimension, sequenceLength, lastPosition, scale));
                hatOutput.copyTo(output);
            }
        }
        return output;
    }
}
