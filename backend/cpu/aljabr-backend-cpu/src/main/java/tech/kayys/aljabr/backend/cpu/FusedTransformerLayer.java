package tech.kayys.aljabr.backend.cpu;

import tech.kayys.aljabr.core.tensor.*;
import tech.kayys.aljabr.core.memory.*;
import java.lang.foreign.*;
import java.nio.ByteOrder;
import jdk.incubator.vector.*;

public final class FusedTransformerLayer {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private FusedTransformerLayer() {
    }

    // =========================================================
    // PUBLIC API
    // =========================================================

    public static Tensor forward(
            Tensor x,
            Tensor wqkv, // [D, 3D]
            Tensor wout, // [D, D]
            Tensor w1, // [D, 4D]
            Tensor w2, // [4D, D]
            int numHeads) {
        int seq = (int) x.shape().dim(0);
        int dim = (int) x.shape().dim(1);

        // ---- buffers ----
        CpuBuffer norm1 = new CpuBuffer((long) seq * dim * 4);
        CpuBuffer qkv = new CpuBuffer((long) seq * dim * 3 * 4);
        CpuBuffer attn = new CpuBuffer((long) seq * dim * 4);
        CpuBuffer proj = new CpuBuffer((long) seq * dim * 4);
        CpuBuffer norm2 = new CpuBuffer((long) seq * dim * 4);
        CpuBuffer ffn1 = new CpuBuffer((long) seq * dim * 4 * 4);
        CpuBuffer ffn2 = new CpuBuffer((long) seq * dim * 4);

        // 1. RMSNorm
        rmsnorm(((DefaultTensor) x).buffer().segment(), norm1.segment(), seq, dim);

        // 2. QKV fused projection
        matmul(norm1.segment(), ((DefaultTensor) wqkv).buffer().segment(), qkv.segment(), seq, dim, dim * 3);

        // 3. Split Q K V
        MemorySegment Q = slice(qkv.segment(), 0, (long) seq * dim);
        MemorySegment K = slice(qkv.segment(), (long) seq * dim, (long) seq * dim);
        MemorySegment V = slice(qkv.segment(), (long) seq * dim * 2, (long) seq * dim);

        // 4. Flash Attention (stub; FlashAttentionCpu not yet bound)
        try {
            FlashAttentionV2.forward(Q, K, V, attn.segment(), seq, dim,
                    Runtime.getRuntime().availableProcessors());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Flash attention interrupted", e);
        }

        // 5. Output projection
        matmul(attn.segment(), ((DefaultTensor) wout).buffer().segment(), proj.segment(), seq, dim, dim);

        // Residual add
        add(proj.segment(), ((DefaultTensor) x).buffer().segment(), proj.segment(), seq * dim);

        // 6. RMSNorm 2
        rmsnorm(proj.segment(), norm2.segment(), seq, dim);

        // 7. FFN (fused SwiGLU-style)
        matmul(norm2.segment(), ((DefaultTensor) w1).buffer().segment(), ffn1.segment(), seq, dim, dim * 4);
        silu(ffn1.segment(), seq * dim * 4);
        matmul(ffn1.segment(), ((DefaultTensor) w2).buffer().segment(), ffn2.segment(), seq, dim * 4, dim);

        // Residual
        add(ffn2.segment(), proj.segment(), ffn2.segment(), seq * dim);

        return new DefaultTensor(new Shape(seq, dim), x.dtype(), x.device(), ffn2, null);
    }

    // =========================================================
    // RMSNorm
    // =========================================================

    private static void rmsnorm(MemorySegment in, MemorySegment out, int rows, int dim) {
        for (int r = 0; r < rows; r++) {
            long base = (long) r * dim;
            float sum = 0f;
            for (int d = 0; d < dim; d++) {
                float v = in.get(ValueLayout.JAVA_FLOAT, (base + d) * 4);
                sum += v * v;
            }
            float scale = (float) (1.0 / Math.sqrt(sum / dim + 1e-5));
            for (int d = 0; d < dim; d++) {
                float v = in.get(ValueLayout.JAVA_FLOAT, (base + d) * 4);
                out.set(ValueLayout.JAVA_FLOAT, (base + d) * 4, v * scale);
            }
        }
    }

    // =========================================================
    // MATMUL
    // =========================================================

    private static void matmul(
            MemorySegment A, MemorySegment B, MemorySegment C,
            int M, int K, int N) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0f;
                for (int k = 0; k < K; k++) {
                    float a = A.get(ValueLayout.JAVA_FLOAT, ((long) i * K + k) * 4);
                    float b = B.get(ValueLayout.JAVA_FLOAT, ((long) k * N + j) * 4);
                    sum += a * b;
                }
                C.set(ValueLayout.JAVA_FLOAT, ((long) i * N + j) * 4, sum);
            }
        }
    }

    // =========================================================
    // ADD
    // =========================================================

    private static void add(MemorySegment a, MemorySegment b, MemorySegment out, int n) {
        for (int i = 0; i < n; i++) {
            float va = a.get(ValueLayout.JAVA_FLOAT, i * 4);
            float vb = b.get(ValueLayout.JAVA_FLOAT, i * 4);
            out.set(ValueLayout.JAVA_FLOAT, i * 4, va + vb);
        }
    }

    // =========================================================
    // SiLU (Swish)
    // =========================================================

    private static void silu(MemorySegment seg, int n) {
        for (int i = 0; i < n; i++) {
            float x = seg.get(ValueLayout.JAVA_FLOAT, (long) i * 4);
            float y = (float) (x / (1.0 + Math.exp(-x)));
            seg.set(ValueLayout.JAVA_FLOAT, (long) i * 4, y);
        }
    }

    // =========================================================
    // SLICE (no copy)
    // =========================================================

    private static MemorySegment slice(MemorySegment seg, long offsetElements, long elements) {
        return seg.asSlice(offsetElements * 4, elements * 4);
    }
}
