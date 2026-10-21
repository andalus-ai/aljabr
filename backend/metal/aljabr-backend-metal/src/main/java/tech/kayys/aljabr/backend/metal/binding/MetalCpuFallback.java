package tech.kayys.aljabr.metal.binding;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Pure-Java CPU fallback for MetalBinding.
 *
 * <p>
 * Activated automatically when {@code libaljabr_metal.dylib} is absent —
 * identical to the pattern used by {@code FlashAttention3CpuFallback} and
 * {@code PagedAttentionCpuFallback} already in Aljabr. Correct but not
 * performance-optimised; intended for development and CI on non-Apple hardware.
 */
final class MetalCpuFallback {

    private MetalCpuFallback() {
    }

    // ── Matrix multiplication ─────────────────────────────────────────────────

    static int matmul(MemorySegment C, MemorySegment A, MemorySegment B,
            int M, int K, int N, float alpha, float beta) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float acc = 0f;
                for (int k = 0; k < K; k++) {
                    acc += get(A, (long) m * K + k) * get(B, (long) k * N + n);
                }
                float prev = beta != 0f ? get(C, (long) m * N + n) : 0f;
                set(C, (long) m * N + n, alpha * acc + prev);
            }
        }
        return 0;
    }

    static int matmulTransposedRight(MemorySegment C, MemorySegment A, MemorySegment B,
            int M, int K, int N, float alpha, float beta) {
        for (int m = 0; m < M; m++) {
            long aBase = (long) m * K;
            long cBase = (long) m * N;
            for (int n = 0; n < N; n++) {
                long bBase = (long) n * K;
                float acc = 0f;
                for (int k = 0; k < K; k++) {
                    acc += get(A, aBase + k) * get(B, bBase + k);
                }
                float prev = beta != 0f ? get(C, cBase + n) : 0f;
                set(C, cBase + n, alpha * acc + prev);
            }
        }
        return 0;
    }

    // ── Softmax attention ─────────────────────────────────────────────────────

    static int attention(MemorySegment out, MemorySegment Q,
            MemorySegment K_cache, MemorySegment V_cache,
            MemorySegment blockTable, MemorySegment contextLens,
            int B, int T, int H, int D,
            int blockSize, int maxBlocks,
            float scale, int isCausal) {
        return attentionWindowed(out, Q, K_cache, V_cache, blockTable, contextLens,
                B, T, H, H, D, blockSize, maxBlocks, scale, isCausal, 0, 0, 0.0f);
    }

    static int attentionGqa(MemorySegment out, MemorySegment Q,
            MemorySegment K_cache, MemorySegment V_cache,
            MemorySegment blockTable, MemorySegment contextLens,
            int B, int T, int H, int Hkv, int D,
            int blockSize, int maxBlocks,
            float scale, int isCausal, float softCap) {
        return attentionWindowed(out, Q, K_cache, V_cache, blockTable, contextLens,
                B, T, H, Hkv, D, blockSize, maxBlocks, scale, isCausal, 0, 0, softCap);
    }

    static int attentionWindowed(MemorySegment out, MemorySegment Q,
            MemorySegment K_cache, MemorySegment V_cache,
            MemorySegment blockTable, MemorySegment contextLens,
            int B, int T, int H, int Hkv, int D,
            int blockSize, int maxBlocks,
            float scale, int isCausal, int queryStartPos, int slidingWindow, float softCap) {
        if (Hkv <= 0 || H % Hkv != 0) {
            return -1;
        }
        int groupSize = H / Hkv;
        for (int b = 0; b < B; b++) {
            int ctxLen = contextLens.getAtIndex(ValueLayout.JAVA_INT, b);
            int numBlocks = (ctxLen + blockSize - 1) / blockSize;

            for (int h = 0; h < H; h++) {
                int kvHead = h / groupSize;
                for (int t = 0; t < T; t++) {
                    int absQueryPos = queryStartPos + t;
                    int maxPos = isCausal == 1 ? Math.min(absQueryPos, ctxLen - 1) : (ctxLen - 1);
                    int minPos = slidingWindow > 0 ? Math.max(0, absQueryPos - slidingWindow + 1) : 0;
                    if (maxPos < minPos) {
                        for (int d = 0; d < D; d++) {
                            long outOff = ((long) (b * T * H + t * H + h)) * D + d;
                            set(out, outOff, 0.0f);
                        }
                        continue;
                    }

                    int activeTokens = maxPos - minPos + 1;
                    float[] scores = new float[activeTokens];
                    float maxS = Float.NEGATIVE_INFINITY;

                    for (int absPos = minPos; absPos <= maxPos; absPos++) {
                        int blk = absPos / blockSize;
                        int tok = absPos % blockSize;
                        int phys = blockTable.getAtIndex(ValueLayout.JAVA_INT, (long) b * maxBlocks + blk);

                        float dot = 0f;
                        for (int d = 0; d < D; d++) {
                            long qOff = ((long) (b * T * H + t * H + h)) * D + d;
                            long kOff = (((long) phys * Hkv + kvHead) * blockSize + tok) * D + d;
                            dot += get(Q, qOff) * get(K_cache, kOff);
                        }
                        dot *= scale;
                        if (softCap > 0.0f) {
                            dot = (float) (softCap * Math.tanh(dot / softCap));
                        }
                        scores[absPos - minPos] = dot;
                        if (dot > maxS) {
                            maxS = dot;
                        }
                    }

                    // Softmax
                    float sum = 0f;
                    for (int p = 0; p < activeTokens; p++) {
                        scores[p] = (float) Math.exp(scores[p] - maxS);
                        sum += scores[p];
                    }
                    for (int p = 0; p < activeTokens; p++) {
                        scores[p] /= sum;
                    }

                    // Weighted sum of V
                    for (int d = 0; d < D; d++) {
                        float acc = 0f;
                        for (int p = 0; p < activeTokens; p++) {
                            int absPos = minPos + p;
                            int blk = absPos / blockSize;
                            int tok = absPos % blockSize;
                            int phys = blockTable.getAtIndex(ValueLayout.JAVA_INT, (long) b * maxBlocks + blk);
                            long vOff = (((long) phys * Hkv + kvHead) * blockSize + tok) * D + d;
                            acc += scores[p] * get(V_cache, vOff);
                        }
                        long outOff = ((long) (b * T * H + t * H + h)) * D + d;
                        set(out, outOff, acc);
                    }
                }
            }
        }
        return 0;
    }

    // ── RMS Norm ──────────────────────────────────────────────────────────────

    static int rmsNorm(MemorySegment out, MemorySegment x,
            MemorySegment weight, int N, float eps, boolean addOne) {
        float ss = 0f;
        for (int i = 0; i < N; i++) {
            float v = get(x, i);
            ss += v * v;
        }
        float inv = 1f / (float) Math.sqrt(ss / N + eps);
        for (int i = 0; i < N; i++) {
            float w = get(weight, i);
            if (addOne)
                w += 1.0f;
            set(out, i, get(x, i) * inv * w);
        }
        return 0;
    }

    static int layerNorm(MemorySegment out, MemorySegment x,
            MemorySegment weight, MemorySegment bias, int N, float eps) {
        float sum = 0f;
        float sumSq = 0f;
        for (int i = 0; i < N; i++) {
            float v = get(x, i);
            sum += v;
            sumSq += v * v;
        }
        float mean = sum / N;
        float variance = (sumSq / N) - (mean * mean);
        float inv = 1f / (float) Math.sqrt(variance + eps);
        for (int i = 0; i < N; i++) {
            float w = weight != null ? get(weight, i) : 1.0f;
            float b = bias != null ? get(bias, i) : 0.0f;
            set(out, i, (get(x, i) - mean) * inv * w + b);
        }
        return 0;
    }

    // ── SiLU FFN ──────────────────────────────────────────────────────────────

    static int siluFfn(MemorySegment out, MemorySegment gate,
            MemorySegment up, int N) {
        for (int i = 0; i < N; i++) {
            float g = get(gate, i);
            set(out, i, (g / (1f + (float) Math.exp(-g))) * get(up, i));
        }
        return 0;
    }

    static int geluFfn(MemorySegment out, MemorySegment gate,
            MemorySegment up, int N) {
        for (int i = 0; i < N; i++) {
            float g = get(gate, i);
            float inner = 0.79788456f * (g + 0.044715f * g * g * g);
            float gelu = 0.5f * g * (1f + (float) Math.tanh(inner));
            set(out, i, gelu * get(up, i));
        }
        return 0;
    }

    static int silu(MemorySegment out, MemorySegment x, int N) {
        for (int i = 0; i < N; i++) {
            float g = get(x, i);
            set(out, i, (g / (1f + (float) Math.exp(-g))));
        }
        return 0;
    }

    static int gelu(MemorySegment out, MemorySegment x, int N) {
        for (int i = 0; i < N; i++) {
            float g = get(x, i);
            float inner = 0.79788456f * (g + 0.044715f * g * g * g);
            float gelu = 0.5f * g * (1f + (float) Math.tanh(inner));
            set(out, i, gelu);
        }
        return 0;
    }

    // ── Basic Math ───────────────────────────────────────────────────────────

    static int add(MemorySegment out, MemorySegment a, MemorySegment b, int n) {
        for (int i = 0; i < n; i++) {
            set(out, i, get(a, i) + get(b, i));
        }
        return 0;
    }

    static int sub(MemorySegment out, MemorySegment a, MemorySegment b, int n) {
        for (int i = 0; i < n; i++) {
            set(out, i, get(a, i) - get(b, i));
        }
        return 0;
    }

    static int mul(MemorySegment out, MemorySegment a, MemorySegment b, int n) {
        for (int i = 0; i < n; i++) {
            set(out, i, get(a, i) * get(b, i));
        }
        return 0;
    }

    static int div(MemorySegment out, MemorySegment a, MemorySegment b, int n) {
        for (int i = 0; i < n; i++) {
            set(out, i, get(a, i) / get(b, i));
        }
        return 0;
    }

    static int relu(MemorySegment out, MemorySegment a, int n) {
        for (int i = 0; i < n; i++) {
            set(out, i, Math.max(0.0f, get(a, i)));
        }
        return 0;
    }

    static int sigmoid(MemorySegment out, MemorySegment a, int n) {
        for (int i = 0; i < n; i++) {
            float x = get(a, i);
            set(out, i, 1.0f / (1.0f + (float) Math.exp(-x)));
        }
        return 0;
    }

    static int tanh(MemorySegment out, MemorySegment a, int n) {
        for (int i = 0; i < n; i++) {
            set(out, i, (float) Math.tanh(get(a, i)));
        }
        return 0;
    }

    static int exp(MemorySegment out, MemorySegment a, int n) {
        for (int i = 0; i < n; i++) {
            set(out, i, (float) Math.exp(get(a, i)));
        }
        return 0;
    }

    static int log(MemorySegment out, MemorySegment a, int n) {
        for (int i = 0; i < n; i++) {
            set(out, i, (float) Math.log(get(a, i)));
        }
        return 0;
    }

    static int sum(MemorySegment out, MemorySegment a, int n) {
        float acc = 0.0f;
        for (int i = 0; i < n; i++) {
            acc += get(a, i);
        }
        set(out, 0, acc);
        return 0;
    }

    static int mean(MemorySegment out, MemorySegment a, int n) {
        float acc = 0.0f;
        for (int i = 0; i < n; i++) {
            acc += get(a, i);
        }
        set(out, 0, acc / n);
        return 0;
    }

    static int pow(MemorySegment out, MemorySegment a, int n, float p) {
        for (int i = 0; i < n; i++) {
            set(out, i, (float) Math.pow(get(a, i), p));
        }
        return 0;
    }

    static int transpose2d(MemorySegment out, MemorySegment a, int rows, int cols) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                set(out, (long) c * rows + r, get(a, (long) r * cols + c));
            }
        }
        return 0;
    }

    // ── Rotary Positional Embedding ───────────────────────────────────────────

    /**
     * Model-agnostic RoPE CPU fallback.
     *
     * <p>
     * Two dimension-pairing conventions are supported:
     * <ul>
     * <li><b>NeoX / HuggingFace</b> ({@code isNeox=true}): pairs element {@code i}
     * with
     * element {@code i + headDim/2} within the same head.</li>
     * <li><b>LLaMA / Gemma interleaved</b> ({@code isNeox=false}): pairs element
     * {@code 2i} with element {@code 2i+1} within the same head.</li>
     * </ul>
     * Both conventions compute the same cos/sin frequencies; only the memory layout
     * differs.
     */
    static int rope(MemorySegment out, MemorySegment x,
            int n, int headDim, int posOffset, float freqBase, boolean isNeox) {
        int numTokenHeads = n / headDim; // total (seqLen * numHeads)
        int halfDim = headDim / 2;

        for (int th = 0; th < numTokenHeads; th++) {
            // Absolute position of this token. For multi-head, all heads at the same
            // sequence position share the same position index.
            int pos = posOffset + th; // simple stride for single-head batches;
            // NOTE: callers that know their seqLen/numHeads layout should compute
            // posOffset per-token before calling this. Here we keep it generic.

            for (int i = 0; i < halfDim; i++) {
                // Frequency: freq_i = 1 / (freqBase ^ (2i / headDim))
                float freq = (float) (1.0 / Math.pow(freqBase, 2.0 * i / headDim));
                float theta = pos * freq;
                float cosTheta = (float) Math.cos(theta);
                float sinTheta = (float) Math.sin(theta);

                long base = (long) th * headDim;
                long i0, i1; // indices of the two elements to rotate together
                if (isNeox) {
                    // GPT-NeoX / HuggingFace style
                    i0 = base + i;
                    i1 = base + i + halfDim;
                } else {
                    // LLaMA / Gemma interleaved style
                    i0 = base + 2L * i;
                    i1 = base + 2L * i + 1;
                }

                float x0 = get(x, i0);
                float x1 = get(x, i1);
                set(out, i0, x0 * cosTheta - x1 * sinTheta);
                set(out, i1, x0 * sinTheta + x1 * cosTheta);
            }
        }
        return 0;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static float get(MemorySegment seg, long idx) {
        return seg.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
    }

    private static void set(MemorySegment seg, long idx, float val) {
        seg.setAtIndex(ValueLayout.JAVA_FLOAT, idx, val);
    }
}
