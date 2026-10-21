package tech.kayys.aljabr.backend.cpu.ops;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * SIMD-accelerated building blocks for the CPU backend using the Java Vector
 * API.
 *
 * <p>
 * Every public method follows the pattern:
 * 
 * <pre>
 *   1. SIMD loop   — processes {@code
 * SPECIES.length()
 * } floats per iteration
 *   2. Scalar tail — handles the remainder ({@code
 * n % SPECIES.length()
 * } elements)
 * </pre>
 *
 * <p>
 * On ARM (Apple Silicon) {@code SPECIES_PREFERRED} maps to 128-bit NEON (4
 * floats/cycle).
 * On x86-64 with AVX-512 it maps to 512-bit lanes (16 floats/cycle).
 */
public final class CpuOps {

    public static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final ByteOrder NATIVE_ORDER = ByteOrder.nativeOrder();

    // ── Minimax polynomial coefficients for exp(x) approximation ─────────────
    // Based on the 5th-degree Horner form by Schraudolph, good for x ∈ [-87, 88]
    private static final float EXP_C0 = 1.0f;
    private static final float EXP_C1 = 0.9999999403f;
    private static final float EXP_C2 = 0.4999999702f;
    private static final float EXP_C3 = 0.1666666716f;
    private static final float EXP_C4 = 0.0416664764f;
    private static final float EXP_C5 = 0.0083337840f;

    private CpuOps() {
    }

    // ── Scalar helpers ────────────────────────────────────────────────────────

    public static float getF(MemorySegment seg, long idx) {
        return seg.get(ValueLayout.JAVA_FLOAT, idx * 4L);
    }

    public static void setF(MemorySegment seg, long idx, float v) {
        seg.set(ValueLayout.JAVA_FLOAT, idx * 4L, v);
    }

    // ── Pointwise SIMD ops ────────────────────────────────────────────────────

    public static void add(MemorySegment a, MemorySegment b, MemorySegment dst, long n) {
        long bound = SPECIES.loopBound(n), i = 0;
        for (; i < bound; i += SPECIES.length())
            FloatVector.fromMemorySegment(SPECIES, a, i * 4L, NATIVE_ORDER)
                    .add(FloatVector.fromMemorySegment(SPECIES, b, i * 4L, NATIVE_ORDER))
                    .intoMemorySegment(dst, i * 4L, NATIVE_ORDER);
        for (; i < n; i++)
            setF(dst, i, getF(a, i) + getF(b, i));
    }

    public static void sub(MemorySegment a, MemorySegment b, MemorySegment dst, long n) {
        long bound = SPECIES.loopBound(n), i = 0;
        for (; i < bound; i += SPECIES.length())
            FloatVector.fromMemorySegment(SPECIES, a, i * 4L, NATIVE_ORDER)
                    .sub(FloatVector.fromMemorySegment(SPECIES, b, i * 4L, NATIVE_ORDER))
                    .intoMemorySegment(dst, i * 4L, NATIVE_ORDER);
        for (; i < n; i++)
            setF(dst, i, getF(a, i) - getF(b, i));
    }

    public static void mul(MemorySegment a, MemorySegment b, MemorySegment dst, long n) {
        long bound = SPECIES.loopBound(n), i = 0;
        for (; i < bound; i += SPECIES.length())
            FloatVector.fromMemorySegment(SPECIES, a, i * 4L, NATIVE_ORDER)
                    .mul(FloatVector.fromMemorySegment(SPECIES, b, i * 4L, NATIVE_ORDER))
                    .intoMemorySegment(dst, i * 4L, NATIVE_ORDER);
        for (; i < n; i++)
            setF(dst, i, getF(a, i) * getF(b, i));
    }

    public static void div(MemorySegment a, MemorySegment b, MemorySegment dst, long n) {
        long bound = SPECIES.loopBound(n), i = 0;
        for (; i < bound; i += SPECIES.length())
            FloatVector.fromMemorySegment(SPECIES, a, i * 4L, NATIVE_ORDER)
                    .div(FloatVector.fromMemorySegment(SPECIES, b, i * 4L, NATIVE_ORDER))
                    .intoMemorySegment(dst, i * 4L, NATIVE_ORDER);
        for (; i < n; i++)
            setF(dst, i, getF(a, i) / getF(b, i));
    }

    public static void mulScalar(MemorySegment a, float s, MemorySegment dst, long n) {
        long bound = SPECIES.loopBound(n), i = 0;
        for (; i < bound; i += SPECIES.length())
            FloatVector.fromMemorySegment(SPECIES, a, i * 4L, NATIVE_ORDER)
                    .mul(s).intoMemorySegment(dst, i * 4L, NATIVE_ORDER);
        for (; i < n; i++)
            setF(dst, i, getF(a, i) * s);
    }

    public static void addScalar(MemorySegment a, float s, MemorySegment dst, long n) {
        long bound = SPECIES.loopBound(n), i = 0;
        for (; i < bound; i += SPECIES.length())
            FloatVector.fromMemorySegment(SPECIES, a, i * 4L, NATIVE_ORDER)
                    .add(s).intoMemorySegment(dst, i * 4L, NATIVE_ORDER);
        for (; i < n; i++)
            setF(dst, i, getF(a, i) + s);
    }

    // ── Activations ───────────────────────────────────────────────────────────

    /** ReLU: max(0, x) — SIMD via vector max. */
    public static void relu(MemorySegment src, MemorySegment dst, long n) {
        long bound = SPECIES.loopBound(n), i = 0;
        for (; i < bound; i += SPECIES.length())
            FloatVector.fromMemorySegment(SPECIES, src, i * 4L, NATIVE_ORDER)
                    .max(0.0f).intoMemorySegment(dst, i * 4L, NATIVE_ORDER);
        for (; i < n; i++)
            setF(dst, i, Math.max(0f, getF(src, i)));
    }

    /**
     * Sigmoid: 1/(1+exp(-x)).
     * Uses scalar exp per element (JVM JIT typically vectorises Math.exp calls).
     */
    public static void sigmoid(MemorySegment src, MemorySegment dst, long n) {
        for (long i = 0; i < n; i++)
            setF(dst, i, (float) (1.0 / (1.0 + Math.exp(-getF(src, i)))));
    }

    /** tanh via java.lang.Math (JIT may auto-vectorise on AVX platforms). */
    public static void tanh(MemorySegment src, MemorySegment dst, long n) {
        for (long i = 0; i < n; i++)
            setF(dst, i, (float) Math.tanh(getF(src, i)));
    }

    /** Element-wise natural log. */
    public static void log(MemorySegment src, MemorySegment dst, long n) {
        long bound = SPECIES.loopBound(n), i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector v = FloatVector.fromMemorySegment(SPECIES, src, i * 4L, NATIVE_ORDER);
            // Vector API log is available via VectorOperators.LOG on JDK 22+
            v.lanewise(VectorOperators.LOG).intoMemorySegment(dst, i * 4L, NATIVE_ORDER);
        }
        for (; i < n; i++)
            setF(dst, i, (float) Math.log(getF(src, i)));
    }

    /** Element-wise exp. */
    public static void exp(MemorySegment src, MemorySegment dst, long n) {
        long bound = SPECIES.loopBound(n), i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector v = FloatVector.fromMemorySegment(SPECIES, src, i * 4L, NATIVE_ORDER);
            v.lanewise(VectorOperators.EXP).intoMemorySegment(dst, i * 4L, NATIVE_ORDER);
        }
        for (; i < n; i++)
            setF(dst, i, (float) Math.exp(getF(src, i)));
    }

    /** SiLU: x * sigmoid(x) = x / (1 + exp(-x)) — computed element-wise. */
    public static void silu(MemorySegment src, MemorySegment dst, long n) {
        for (long i = 0; i < n; i++) {
            float x = getF(src, i);
            setF(dst, i, x * (float) (1.0 / (1.0 + Math.exp(-x))));
        }
    }

    /**
     * GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))).
     * This is the standard PyTorch "tanh" GELU variant.
     */
    public static void gelu(MemorySegment src, MemorySegment dst, long n) {
        final float SQRT_2_OVER_PI = 0.7978845608f; // sqrt(2/π)
        final float COEFF = 0.044715f;
        for (long i = 0; i < n; i++) {
            float x = getF(src, i);
            float inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
            setF(dst, i, 0.5f * x * (1.0f + (float) Math.tanh(inner)));
        }
    }

    // ── Reductions ────────────────────────────────────────────────────────────

    /**
     * Online numerically-stable softmax over a contiguous row.
     * Two-pass: first find max, then compute sum(exp(x - max)), then normalize.
     */
    public static void softmax(MemorySegment src, MemorySegment dst, long rowStart, int rowLen) {
        // Pass 1: find row max (SIMD reduce)
        float maxVal = Float.NEGATIVE_INFINITY;
        long bound = SPECIES.loopBound(rowLen);
        long i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector v = FloatVector.fromMemorySegment(SPECIES, src, (rowStart + i) * 4L, NATIVE_ORDER);
            maxVal = Math.max(maxVal, v.reduceLanes(VectorOperators.MAX));
        }
        for (; i < rowLen; i++)
            maxVal = Math.max(maxVal, getF(src, rowStart + i));

        // Pass 2: exp(x - max) and accumulate sum
        float sum = 0f;
        for (long j = 0; j < rowLen; j++) {
            float e = (float) Math.exp(getF(src, rowStart + j) - maxVal);
            setF(dst, rowStart + j, e);
            sum += e;
        }

        // Pass 3: normalize
        float invSum = 1f / sum;
        mulScalar(dst.asSlice(rowStart * 4L, rowLen * 4L),
                invSum,
                dst.asSlice(rowStart * 4L, rowLen * 4L),
                rowLen);
    }

    /** Horizontal sum of all elements via SIMD partial sums. */
    public static float reduceSum(MemorySegment src, long n) {
        long bound = SPECIES.loopBound(n);
        long i = 0;
        FloatVector acc = FloatVector.zero(SPECIES);
        for (; i < bound; i += SPECIES.length())
            acc = acc.add(FloatVector.fromMemorySegment(SPECIES, src, i * 4L, NATIVE_ORDER));
        float total = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++)
            total += getF(src, i);
        return total;
    }

    /** Horizontal max of all elements. */
    public static float reduceMax(MemorySegment src, long n) {
        long bound = SPECIES.loopBound(n);
        long i = 0;
        FloatVector acc = FloatVector.broadcast(SPECIES, Float.NEGATIVE_INFINITY);
        for (; i < bound; i += SPECIES.length())
            acc = acc.max(FloatVector.fromMemorySegment(SPECIES, src, i * 4L, NATIVE_ORDER));
        float best = acc.reduceLanes(VectorOperators.MAX);
        for (; i < n; i++)
            best = Math.max(best, getF(src, i));
        return best;
    }

    /**
     * Dropout: zeroes each element with probability {@code p} during training.
     * Uses a cheap Xorshift64* PRNG — not cryptographic but fast.
     */
    public static void dropout(MemorySegment src, MemorySegment dst, long n, float p, long seed) {
        long state = seed ^ (seed << 13) | 1L;
        float scale = 1.0f / (1.0f - p);
        for (long i = 0; i < n; i++) {
            state ^= state >>> 12;
            state ^= state << 25;
            state ^= state >>> 27;
            float rand = (float) ((state * 0x2545F4914F6CDD1DL) >>> 32) / 0xFFFFFFFFL;
            float val = (rand < p) ? 0f : getF(src, i) * scale;
            setF(dst, i, val);
        }
    }
}
