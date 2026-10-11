package tech.kayys.alkhawarizm.gguf.loader.tensor;

import jdk.incubator.vector.*;
import tech.kayys.alkhawarizm.gguf.core.GgmlType;
import tech.kayys.alkhawarizm.gguf.core.GGUFTensorInfo;
import tech.kayys.alkhawarizm.gguf.loader.quant.Dequantizer;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * High-performance tensor operations implemented with the <b>Java Vector API</b>.
 * @author bhangun
 */
public final class GGUFVectorOps {

    private GGUFVectorOps() {}

    /** Preferred SIMD species for the current platform. */
    private static final VectorSpecies<Float> S = FloatVector.SPECIES_PREFERRED;

    // ── Dot product ───────────────────────────────────────────────────────

    /**
     * SIMD dot product of {@code a[0..len)} and {@code b[0..len)}.
     */
    public static float dot(float[] a, float[] b, int len) {
        var acc = FloatVector.zero(S);
        int i = 0;
        int bound = S.loopBound(len);
        for (; i < bound; i += S.length()) {
            var va = FloatVector.fromArray(S, a, i);
            var vb = FloatVector.fromArray(S, b, i);
            acc = va.fma(vb, acc);
        }
        float result = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) result += a[i] * b[i];
        return result;
    }

    /** Dot product of sub-ranges: {@code a[aOff..aOff+len)} · {@code b[bOff..bOff+len)}. */
    public static float dot(float[] a, int aOff, float[] b, int bOff, int len) {
        var acc = FloatVector.zero(S);
        int i = 0;
        int bound = S.loopBound(len);
        for (; i < bound; i += S.length()) {
            var va = FloatVector.fromArray(S, a, aOff + i);
            var vb = FloatVector.fromArray(S, b, bOff + i);
            acc = va.fma(vb, acc);
        }
        float result = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) result += a[aOff + i] * b[bOff + i];
        return result;
    }

    // ── Matrix × vector (f32 × f32) ───────────────────────────────────────

    /**
     * Dense f32 matrix-vector multiply: {@code out = rows × vec}.
     */
    public static void matVecMulF32(float[] rows, float[] vec, float[] out,
                                     int nRows, int cols) {
        for (int r = 0; r < nRows; r++) {
            out[r] = dot(rows, r * cols, vec, 0, cols);
        }
    }

    // ── Matrix × vector (quantized × f32) ────────────────────────────────

    /**
     * Quantization-aware matrix-vector multiply.
     */
    public static void matVecMulQuant(GGUFTensorInfo t, float[] vec, float[] out,
                                       float[] rowBuf) {
        int nRows = (int) t.rows();
        int cols  = (int) t.cols();

        if (t.type() == GgmlType.F32) {
            // Fast path: read f32 rows directly
            matVecMulF32Direct(t.data(), vec, out, nRows, cols);
            return;
        }

        for (int r = 0; r < nRows; r++) {
            Dequantizer.dequantizeRow(t, r, rowBuf);
            out[r] = dot(rowBuf, 0, vec, 0, cols);
        }
    }

    /** Read f32 rows directly from a memory-mapped segment without copying. */
    private static void matVecMulF32Direct(MemorySegment seg, float[] vec, float[] out,
                                            int nRows, int cols) {
        var leFloat = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
        float[] row = new float[cols];
        for (int r = 0; r < nRows; r++) {
            long base = (long) r * cols * 4;
            for (int c = 0; c < cols; c++) {
                row[c] = seg.get(leFloat, base + (long) c * 4);
            }
            out[r] = dot(row, 0, vec, 0, cols);
        }
    }

    // ── Element-wise ops ──────────────────────────────────────────────────

    /** {@code a[i] += b[i]} in-place. */
    public static void addInPlace(float[] a, float[] b, int len) {
        int i = 0, bound = S.loopBound(len);
        for (; i < bound; i += S.length()) {
            FloatVector.fromArray(S, a, i)
                       .add(FloatVector.fromArray(S, b, i))
                       .intoArray(a, i);
        }
        for (; i < len; i++) a[i] += b[i];
    }

    /** {@code dst[i] = a[i] * b[i]}. */
    public static void mulInto(float[] a, float[] b, float[] dst, int len) {
        int i = 0, bound = S.loopBound(len);
        for (; i < bound; i += S.length()) {
            FloatVector.fromArray(S, a, i)
                       .mul(FloatVector.fromArray(S, b, i))
                       .intoArray(dst, i);
        }
        for (; i < len; i++) dst[i] = a[i] * b[i];
    }

    /** {@code dst[i] = a[i] + scale * b[i]}. */
    public static void scaledAdd(float[] dst, float[] a, float scale, float[] b, int len) {
        var vScale = FloatVector.broadcast(S, scale);
        int i = 0, bound = S.loopBound(len);
        for (; i < bound; i += S.length()) {
            FloatVector.fromArray(S, a, i)
                       .add(FloatVector.fromArray(S, b, i).mul(vScale))
                       .intoArray(dst, i);
        }
        for (; i < len; i++) dst[i] = a[i] + scale * b[i];
    }

    /** {@code dst[i] = a[i] + scale * b[bOff + i]}. */
    public static void scaledAdd(float[] dst, float[] a, float scale, float[] b, int bOff, int len) {
        var vScale = FloatVector.broadcast(S, scale);
        int i = 0, bound = S.loopBound(len);
        for (; i < bound; i += S.length()) {
            FloatVector.fromArray(S, a, i)
                       .add(FloatVector.fromArray(S, b, bOff + i).mul(vScale))
                       .intoArray(dst, i);
        }
        for (; i < len; i++) dst[i] = a[i] + scale * b[bOff + i];
    }

    /** Scale all elements: {@code a[i] *= s}. */
    public static void scale(float[] a, float s, int len) {
        var vs = FloatVector.broadcast(S, s);
        int i = 0, bound = S.loopBound(len);
        for (; i < bound; i += S.length()) {
            FloatVector.fromArray(S, a, i).mul(vs).intoArray(a, i);
        }
        for (; i < len; i++) a[i] *= s;
    }

    /** Copy {@code src} into {@code dst}. */
    public static void copy(float[] src, float[] dst, int len) {
        System.arraycopy(src, 0, dst, 0, len);
    }

    /** Zero out {@code a[0..len)}. */
    public static void zero(float[] a, int len) {
        java.util.Arrays.fill(a, 0, len, 0f);
    }

    // ── RMS Norm ──────────────────────────────────────────────────────────

    /**
     * Root-mean-square normalisation (LLaMA-style).
     */
    public static void rmsNorm(float[] x, float[] weight, float[] out, int len, float eps) {
        // sum of squares
        var acc = FloatVector.zero(S);
        int i = 0, bound = S.loopBound(len);
        for (; i < bound; i += S.length()) {
            var v = FloatVector.fromArray(S, x, i);
            acc = v.fma(v, acc);
        }
        float ss = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) ss += x[i] * x[i];

        float scale = 1.0f / (float) Math.sqrt(ss / len + eps);

        // Normalise and apply weight
        var vscale = FloatVector.broadcast(S, scale);
        i = 0;
        for (; i < bound; i += S.length()) {
            FloatVector.fromArray(S, x, i)
                       .mul(vscale)
                       .mul(FloatVector.fromArray(S, weight, i))
                       .intoArray(out, i);
        }
        for (; i < len; i++) out[i] = x[i] * scale * weight[i];
    }

    /**
     * Root-mean-square normalisation on sub-ranges: {@code out[outOff..outOff+len) = norm(x[xOff..xOff+len)) * weight[0..len)}.
     */
    public static void rmsNorm(float[] x, int xOff, float[] weight, float[] out, int outOff, int len, float eps) {
        var acc = FloatVector.zero(S);
        int i = 0, bound = S.loopBound(len);
        for (; i < bound; i += S.length()) {
            var v = FloatVector.fromArray(S, x, xOff + i);
            acc = v.fma(v, acc);
        }
        float ss = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            float val = x[xOff + i];
            ss += val * val;
        }

        float scale = 1.0f / (float) Math.sqrt(ss / len + eps);

        var vscale = FloatVector.broadcast(S, scale);
        i = 0;
        for (; i < bound; i += S.length()) {
            FloatVector.fromArray(S, x, xOff + i)
                       .mul(vscale)
                       .mul(FloatVector.fromArray(S, weight, i))
                       .intoArray(out, outOff + i);
        }
        for (; i < len; i++) {
            out[outOff + i] = x[xOff + i] * scale * weight[i];
        }
    }

    /**
     * Root-mean-square normalisation on sub-ranges with unit weights (no weight tensor).
     */
    public static void rmsNormUnit(float[] x, int xOff, float[] out, int outOff, int len, float eps) {
        var acc = FloatVector.zero(S);
        int i = 0, bound = S.loopBound(len);
        for (; i < bound; i += S.length()) {
            var v = FloatVector.fromArray(S, x, xOff + i);
            acc = v.fma(v, acc);
        }
        float ss = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            float val = x[xOff + i];
            ss += val * val;
        }

        float scale = 1.0f / (float) Math.sqrt(ss / len + eps);

        var vscale = FloatVector.broadcast(S, scale);
        i = 0;
        for (; i < bound; i += S.length()) {
            FloatVector.fromArray(S, x, xOff + i)
                       .mul(vscale)
                       .intoArray(out, outOff + i);
        }
        for (; i < len; i++) {
            out[outOff + i] = x[xOff + i] * scale;
        }
    }

    // ── Softmax ───────────────────────────────────────────────────────────

    /**
     * In-place softmax over {@code a[0..len)}.
     */
    public static void softmax(float[] a, int len) {
        // find max
        float max = a[0];
        for (int i = 1; i < len; i++) if (a[i] > max) max = a[i];

        // exp(x - max) and sum
        float sum = 0f;
        for (int i = 0; i < len; i++) {
            a[i] = (float) Math.exp(a[i] - max);
            sum += a[i];
        }

        // divide by sum
        float invSum = 1f / sum;
        scale(a, invSum, len);
    }

    // ── SiLU ─────────────────────────────────────────────────────────────

    /**
     * SiLU activation: {@code x * sigmoid(x) = x / (1 + exp(-x))}.
     */
    public static void siluInPlace(float[] a, int len) {
        for (int i = 0; i < len; i++) {
            a[i] = a[i] / (1f + (float) Math.exp(-a[i]));
        }
    }

    /**
     * SwiGLU: {@code dst[i] = silu(gate[i]) * up[i]}.
     */
    public static void swiGLU(float[] gate, float[] up, float[] dst, int len) {
        for (int i = 0; i < len; i++) {
            float g = gate[i] / (1f + (float) Math.exp(-gate[i]));
            dst[i] = g * up[i];
        }
    }

    /**
     * GeGLU (approximate 'tanh'): {@code dst[i] = gelu(gate[i]) * up[i]}.
     */
    public static void geGLU(float[] gate, float[] up, float[] dst, int len) {
        float k = 0.7978845608f; // sqrt(2/pi)
        for (int i = 0; i < len; i++) {
            float g = gate[i];
            float gelu = 0.5f * g * (1.0f + (float) Math.tanh(k * (g + 0.044715f * g * g * g)));
            dst[i] = gelu * up[i];
        }
    }

    // ── Rope ─────────────────────────────────────────────────────────────

    /**
     * Apply Rotary Position Embedding (RoPE) in-place.
     */
    public static void rope(float[] x, int nHeads, int headDim, int pos, float freqBase) {
        int half = headDim / 2;
        for (int h = 0; h < nHeads; h++) {
            int base = h * headDim;
            for (int i = 0; i < half; i++) {
                float theta = pos / (float) Math.pow(freqBase, 2.0 * i / headDim);
                float cos   = (float) Math.cos(theta);
                float sin   = (float) Math.sin(theta);
                float x0    = x[base + i];
                float x1    = x[base + i + half];
                x[base + i]        = x0 * cos - x1 * sin;
                x[base + i + half] = x0 * sin + x1 * cos;
            }
        }
    }

    // ── Embedding lookup ──────────────────────────────────────────────────

    /**
     * Copy embedding row {@code tokenId} from the embedding table into {@code out}.
     */
    public static void embedLookup(float[] embTable, int tokenId, int dim, float[] out) {
        System.arraycopy(embTable, tokenId * dim, out, 0, dim);
    }

    // ── Argmax / sampling helpers ─────────────────────────────────────────

    /** Returns the index of the maximum value in {@code a[0..len)}. */
    public static int argmax(float[] a, int len) {
        int best = 0;
        for (int i = 1; i < len; i++) if (a[i] > a[best]) best = i;
        return best;
    }

    /** Returns the sum of {@code a[0..len)}. */
    public static float sum(float[] a, int len) {
        var acc = FloatVector.zero(S);
        int i = 0, bound = S.loopBound(len);
        for (; i < bound; i += S.length()) acc = acc.add(FloatVector.fromArray(S, a, i));
        float s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) s += a[i];
        return s;
    }
}
