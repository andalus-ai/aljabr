package tech.kayys.aljabr.backend.cpu;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import java.util.function.LongFunction;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * CPU attention execution, including standard multi-head attention and GQA.
 *
 * <p>This component owns attention-specific validation, masking, work partitioning
 * and online-softmax execution. The backend facade supplies backend identity and
 * output-storage/tensor construction so allocation lifecycle remains backend-owned.</p>
 */
final class CpuAttentionExecution {
    @FunctionalInterface
    interface OutputFactory {
        Tensor create(Shape shape, HostAccessibleStorage storage);
    }

    private final ComputeBackend backend;
    private final LongFunction<HostAccessibleStorage> outputAllocator;
    private final OutputFactory outputFactory;

    CpuAttentionExecution(
            ComputeBackend backend,
            LongFunction<HostAccessibleStorage> outputAllocator,
            OutputFactory outputFactory) {
        this.backend = Objects.requireNonNull(backend, "backend");
        this.outputAllocator = Objects.requireNonNull(outputAllocator, "outputAllocator");
        this.outputFactory = Objects.requireNonNull(outputFactory, "outputFactory");
    }

        public Tensor attention(Tensor Q, Tensor K, Tensor V) {
        return attention(Q, K, V, null, false);
    }

        public Tensor attention(Tensor Q, Tensor K, Tensor V, Tensor additiveMask, boolean causal) {
        if (Q == null || K == null || V == null) throw new IllegalArgumentException("attention Q, K, and V must not be null");
        if (Q.backend() != backend || K.backend() != backend || V.backend() != backend)
            throw new IllegalArgumentException("attention tensors must belong to this CPU backend instance");
        if (Q.dtype() != DType.F32 || K.dtype() != DType.F32 || V.dtype() != DType.F32)
            throw new UnsupportedOperationException("CPU attention currently supports F32 tensors only");
        int rank = Q.shape().rank();
        if (rank == 2 && additiveMask == null && !causal) {
            if (K.shape().rank() != 2 || V.shape().rank() != 2
                    || Q.shape().dim(0) != K.shape().dim(0) || K.shape().dim(0) != V.shape().dim(0)
                    || Q.shape().dim(1) != K.shape().dim(1) || K.shape().dim(1) != V.shape().dim(1))
                throw new IllegalArgumentException("rank-2 attention expects matching [sequence, headDim] Q/K/V");
            return FlashAttentionCpu.forward(Q, K, V, Runtime.getRuntime().availableProcessors());
        }
        if (rank != 4 || K.shape().rank() != 4 || V.shape().rank() != 4)
            throw new IllegalArgumentException("masked attention expects rank-4 [B,H,S,D] Q/K/V tensors");
        long batch=Q.shape().dim(0), heads=Q.shape().dim(1), queryLength=Q.shape().dim(2), headDim=Q.shape().dim(3), keyLength=K.shape().dim(2);
        if (batch<=0 || heads<=0 || queryLength<=0 || keyLength<=0 || headDim<=0
                || K.shape().dim(0)!=batch || K.shape().dim(1)!=heads || K.shape().dim(3)!=headDim
                || V.shape().dim(0)!=batch || V.shape().dim(1)!=heads || V.shape().dim(2)!=keyLength || V.shape().dim(3)!=headDim)
            throw new IllegalArgumentException("rank-4 attention expects Q[B,H,Sq,D], K/V[B,H,Sk,D] with matching B/H/D");
        if (additiveMask != null && (additiveMask.backend()!=backend || additiveMask.dtype()!=DType.F32
                || additiveMask.shape().rank()!=2 || additiveMask.shape().dim(0)!=queryLength
                || additiveMask.shape().dim(1)!=keyLength))
            throw new IllegalArgumentException("additive attention mask must be F32 [queryLength, keyLength] on this backend");
        MemorySegment mask = additiveMask == null ? null : CpuTensorLayoutAccess.contiguousF32(additiveMask);
        if (mask != null) {
            validateAttentionMask(mask, queryLength, keyLength);
        }
        Shape outputShape=new Shape(batch,heads,queryLength,headDim);
        HostAccessibleStorage output=outputAllocator.apply(Math.multiplyExact(outputShape.numel(),Float.BYTES));
        MemorySegment q=CpuTensorLayoutAccess.contiguousF32(Q), k=CpuTensorLayoutAccess.contiguousF32(K),
                v=CpuTensorLayoutAccess.contiguousF32(V), out=output.segment();
        float scale=(float)(1.0/Math.sqrt(headDim));
        int groupCount = Math.toIntExact(Math.multiplyExact(batch, heads));
        try {
            if (shouldParallelizeAttention(groupCount, queryLength, keyLength, headDim)) {
                // The common ForkJoinPool bounds CPU attention workers globally
                // across backend instances and concurrent inference requests.
                java.util.stream.IntStream.range(0, groupCount).parallel().forEach(group ->
                        computeAttentionGroup(q, k, v, mask, out, group, group,
                                queryLength, keyLength, headDim, scale, causal));
            } else {
                for (int group = 0; group < groupCount; group++) {
                    computeAttentionGroup(q, k, v, mask, out, group, group,
                            queryLength, keyLength, headDim, scale, causal);
                }
            }
            return outputFactory.create(outputShape, output);
        } catch (RuntimeException | Error failure) {
            output.release();
            throw failure;
        }
    }


        public Tensor attentionGqa(Tensor Q, Tensor K, Tensor V, Tensor additiveMask, boolean causal) {
        if (Q == null || K == null || V == null) {
            throw new IllegalArgumentException("attention Q, K, and V must not be null");
        }
        if (Q.backend() != backend || K.backend() != backend || V.backend() != backend) {
            throw new IllegalArgumentException("attention tensors must belong to this CPU backend instance");
        }
        if (Q.dtype() != DType.F32 || K.dtype() != DType.F32 || V.dtype() != DType.F32) {
            throw new UnsupportedOperationException("CPU attention currently supports F32 tensors only");
        }
        if (Q.shape().rank() != 4 || K.shape().rank() != 4 || V.shape().rank() != 4) {
            throw new IllegalArgumentException("GQA expects rank-4 [B,H,S,D] Q/K/V tensors");
        }

        long batch = Q.shape().dim(0);
        long queryHeads = Q.shape().dim(1);
        long queryLength = Q.shape().dim(2);
        long headDim = Q.shape().dim(3);
        long keyValueHeads = K.shape().dim(1);
        long keyLength = K.shape().dim(2);

        if (batch <= 0 || queryHeads <= 0 || keyValueHeads <= 0
                || queryLength <= 0 || keyLength <= 0 || headDim <= 0
                || queryHeads % keyValueHeads != 0
                || K.shape().dim(0) != batch || V.shape().dim(0) != batch
                || K.shape().dim(3) != headDim || V.shape().dim(1) != keyValueHeads
                || V.shape().dim(2) != keyLength || V.shape().dim(3) != headDim) {
            throw new IllegalArgumentException(
                    "GQA expects Q[B,Hq,Sq,D], K/V[B,Hkv,Sk,D] with Hq divisible by Hkv");
        }

        if (additiveMask != null && (additiveMask.backend() != backend
                || additiveMask.dtype() != DType.F32
                || additiveMask.shape().rank() != 2
                || additiveMask.shape().dim(0) != queryLength
                || additiveMask.shape().dim(1) != keyLength)) {
            throw new IllegalArgumentException(
                    "GQA additive attention mask must be F32 [queryLength, keyLength] on this backend");
        }

        MemorySegment mask = additiveMask == null ? null : CpuTensorLayoutAccess.contiguousF32(additiveMask);
        if (mask != null) validateAttentionMask(mask, queryLength, keyLength);

        Shape outputShape = new Shape(batch, queryHeads, queryLength, headDim);
        HostAccessibleStorage output =
                outputAllocator.apply(Math.multiplyExact(outputShape.numel(), Float.BYTES));
        MemorySegment q = CpuTensorLayoutAccess.contiguousF32(Q),
                k = CpuTensorLayoutAccess.contiguousF32(K),
                v = CpuTensorLayoutAccess.contiguousF32(V), out = output.segment();
        float scale = (float) (1.0 / Math.sqrt(headDim));
        int groupCount = Math.toIntExact(Math.multiplyExact(batch, queryHeads));
        int queriesPerKeyValueHead = Math.toIntExact(queryHeads / keyValueHeads);
        try {
            if (shouldParallelizeAttention(groupCount, queryLength, keyLength, headDim)) {
                java.util.stream.IntStream.range(0, groupCount).parallel().forEach(group -> {
                    int queryHead = group % Math.toIntExact(queryHeads);
                    int keyValueHead = queryHead / queriesPerKeyValueHead;
                    computeAttentionGroup(q, k, v, mask, out, group,
                            (group / Math.toIntExact(queryHeads))
                                    * Math.toIntExact(keyValueHeads) + keyValueHead,
                            queryLength, keyLength, headDim, scale, causal);
                });
            } else {
                for (int group = 0; group < groupCount; group++) {
                    int queryHead = group % Math.toIntExact(queryHeads);
                    int keyValueHead = queryHead / queriesPerKeyValueHead;
                    computeAttentionGroup(q, k, v, mask, out, group,
                            (group / Math.toIntExact(queryHeads))
                                    * Math.toIntExact(keyValueHeads) + keyValueHead,
                            queryLength, keyLength, headDim, scale, causal);
                }
            }
            return outputFactory.create(outputShape, output);
        } catch (RuntimeException | Error failure) {
            output.release();
            throw failure;
        }
    }

    /**
     * Validates a shared additive mask once before attention work is scheduled.
     * The same mask is reused by every batch/head group, so validation must not
     * be repeated in each group's query/key hot loop.
     */
    private static void validateAttentionMask(
            MemorySegment mask, long queryLength, long keyLength) {
        long elements = Math.multiplyExact(queryLength, keyLength);
        for (long index = 0; index < elements; index++) {
            float value = mask.get(ValueLayout.JAVA_FLOAT, index * Float.BYTES);
            if (Float.isNaN(value) || value == Float.POSITIVE_INFINITY) {
                throw new IllegalArgumentException(
                        "attention mask values must not be NaN or positive infinity");
            }
        }
    }

    /**
     * Avoid parallel scheduling for small attention workloads where task
     * coordination costs more than processing independent batch/head groups.
     */
    private static boolean shouldParallelizeAttention(
            int groupCount, long queryLength, long keyLength, long headDim) {
        if (groupCount < 2) {
            return false;
        }
        long work = groupCount;
        if (work > Long.MAX_VALUE / queryLength) {
            return true;
        }
        work *= queryLength;
        if (work > Long.MAX_VALUE / keyLength) {
            return true;
        }
        work *= keyLength;
        if (work > Long.MAX_VALUE / headDim) {
            return true;
        }
        work *= headDim;
        return work >= (1L << 18);
    }

    /** Computes one independent [batch, head] attention group. */
    private static void computeAttentionGroup(
            MemorySegment q,
            MemorySegment k,
            MemorySegment v,
            MemorySegment mask,
            MemorySegment out,
            int group,
            int keyGroup,
            long queryLength,
            long keyLength,
            long headDim,
            float scale,
            boolean causal) {
        int dimension = Math.toIntExact(headDim);
        long qBase = (long) group * queryLength * dimension;
        long kBase = (long) keyGroup * keyLength * dimension;
        double[] weightedValues = new double[dimension];

        for (long qi = 0; qi < queryLength; qi++) {
            java.util.Arrays.fill(weightedValues, 0d);
            float runningMax = Float.NEGATIVE_INFINITY;
            double denominator = 0d;
            long causalLimit = keyLength - queryLength + qi;
            long maskRowOffset = qi * keyLength * Float.BYTES;

            // Online softmax: keep only the running normalization and weighted
            // value sum; no query-sized score/probability array is allocated.
            // Bound the loop directly so causally excluded keys incur neither
            // an inner-loop branch nor an additive-mask memory read.
            long keyLimit = causal ? Math.min(keyLength, causalLimit + 1) : keyLength;
            for (long ki = 0; ki < keyLimit; ki++) {
                float additive = mask == null ? 0f
                        : mask.get(ValueLayout.JAVA_FLOAT, maskRowOffset + ki * Float.BYTES);
                if (additive == Float.NEGATIVE_INFINITY) {
                    continue;
                }

                float score = CpuAttentionOps.dot(
                        q, (qBase + qi * dimension) * Float.BYTES,
                        k, (kBase + ki * dimension) * Float.BYTES,
                        dimension) * scale + additive;

                float nextMax;
                double previousScale;
                double currentScale;
                if (Float.isFinite(score)) {
                    if (score > runningMax) {
                        previousScale = Math.exp(runningMax - score);
                        currentScale = 1d;
                        nextMax = score;
                    } else {
                        previousScale = 1d;
                        currentScale = Math.exp(score - runningMax);
                        nextMax = runningMax;
                    }
                } else {
                    // Preserve the existing exceptional-value behavior.
                    nextMax = Math.max(runningMax, score);
                    previousScale = runningMax == Float.NEGATIVE_INFINITY
                            ? 0d : Math.exp(runningMax - nextMax);
                    currentScale = Math.exp(score - nextMax);
                }
                denominator = denominator * previousScale + currentScale;

                for (int d = 0; d < dimension; d++) {
                    float valueAtKey = v.get(ValueLayout.JAVA_FLOAT,
                            (kBase + ki * dimension + d) * Float.BYTES);
                    weightedValues[d] = weightedValues[d] * previousScale
                            + currentScale * valueAtKey;
                }
                runningMax = nextMax;
            }

            for (int d = 0; d < dimension; d++) {
                float result = denominator == 0d
                        ? 0f : (float) (weightedValues[d] / denominator);
                out.set(ValueLayout.JAVA_FLOAT,
                        (qBase + qi * dimension + d) * Float.BYTES, result);
            }
        }
    }
}
