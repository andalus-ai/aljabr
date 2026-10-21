package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Regression tests for the pure math primitives {@link GgufDecodeStep} calls.
 * These need no GGUF file, no mock objects, and no dependency on anything
 * unverified in this bundle — {@link GGUFVectorOps} takes plain
 * {@code float[]} in and out, so every expected value here was computed
 * independently (numpy, float64 intermediate precision, cast to float32 at
 * the same points the implementation does) rather than derived from reading
 * the implementation itself. That's what makes these a genuine check rather
 * than a tautology: a bug in {@code rmsNorm}'s eps placement or
 * {@code rope}'s rotation direction would fail one of these.
 *
 * <p>This is a small, honest slice of "Tier 1 — prove it's right" from the
 * production roadmap: it proves the math primitives are right. It does NOT
 * prove {@link GgufDecodeStep} assembles them correctly, and it does NOT
 * prove numerical correctness against llama.cpp on a real model — that
 * still requires the golden-output comparison from the roadmap, which
 * needs a real GGUF file and a running llama.cpp to compare against,
 * neither of which is available in this environment.
 *
 * <p>Delta tolerances are looser than float32 ULP because
 * {@link GGUFVectorOps}'s SIMD reductions (e.g. the sum-of-squares in
 * {@code rmsNorm}, the dot-product accumulation) sum lanes in a different
 * order than the scalar reference computation used to derive these expected
 * values — floating-point addition isn't associative, so a different
 * summation order gives a slightly different result even when both are
 * "correct." 1e-4 comfortably clears that noise while still catching a
 * real formula bug (those produce differences many orders of magnitude
 * larger).
 */
class GGUFVectorOpsTest {

    private static final float DELTA = 1e-4f;

    @Test
    void dot_computesInnerProduct() {
        float[] a = {1f, 2f, 3f};
        float[] b = {4f, 5f, 6f};
        assertEquals(32f, GGUFVectorOps.dot(a, 0, b, 0, 3), DELTA);
    }

    @Test
    void dot_respectsOffsets() {
        // Same vectors as above, embedded in larger arrays at nonzero offsets
        // -- this is exactly how GgufDecodeStep reads Q against a KV cache
        // row, so a bug in offset handling here would break attention
        // silently rather than throwing.
        float[] a = {99f, 99f, 1f, 2f, 3f};
        float[] b = {4f, 5f, 6f, 77f};
        assertEquals(32f, GGUFVectorOps.dot(a, 2, b, 0, 3), DELTA);
    }

    @Test
    void rmsNorm_uniformWeights() {
        float[] x = {1f, 2f, 3f, 4f};
        float[] weight = {1f, 1f, 1f, 1f};
        float[] out = new float[4];

        GGUFVectorOps.rmsNorm(x, weight, out, 4, 1e-5f);

        // scale = 1/sqrt(mean(x^2) + eps) = 1/sqrt(7.5 + 1e-5) ~= 0.365148
        float[] expected = {0.36514813f, 0.73029625f, 1.0954444f, 1.4605925f};
        assertArrayEquals(expected, out, DELTA);
    }

    @Test
    void rmsNorm_nonUniformWeightsAndNegativeValues() {
        // Negative x and non-1.0 weights exercise the weight multiply and
        // the squaring separately -- a bug that only shows up with negative
        // inputs (e.g. an accidental Math.abs somewhere) wouldn't be caught
        // by the all-positive case above.
        float[] x = {2f, -1f, 0.5f, 3f};
        float[] weight = {1.0f, 2.0f, 0.5f, 1.5f};
        float[] out = new float[4];

        GGUFVectorOps.rmsNorm(x, weight, out, 4, 1e-5f);

        float[] expected = {1.0596244f, -1.0596244f, 0.13245305f, 2.384155f};
        assertArrayEquals(expected, out, DELTA);
    }

    @Test
    void softmax_sumsToOneAndPreservesOrder() {
        float[] a = {1f, 2f, 3f};
        GGUFVectorOps.softmax(a, 3);

        float[] expected = {0.09003057f, 0.24472847f, 0.66524096f};
        assertArrayEquals(expected, a, DELTA);

        float sum = a[0] + a[1] + a[2];
        assertEquals(1.0f, sum, DELTA);
    }

    @Test
    void softmax_isShiftInvariant() {
        // softmax(x) == softmax(x + c) for any constant c -- this is the
        // property the max-subtraction trick in the implementation relies
        // on for numerical stability. If that subtraction were wired wrong
        // (e.g. subtracting the wrong lane, or only shifting some entries),
        // this is the test that would catch it.
        float[] a = {10_000f, 10_001f, 10_002f};
        float[] b = {1f, 2f, 3f};
        GGUFVectorOps.softmax(a, 3);
        GGUFVectorOps.softmax(b, 3);
        assertArrayEquals(b, a, DELTA);
    }

    @Test
    void swiGLU_matchesSiluTimesUp() {
        float[] gate = {0f, 1f, -1f};
        float[] up = {2f, 2f, 2f};
        float[] dst = new float[3];

        GGUFVectorOps.swiGLU(gate, up, dst, 3);

        // silu(0)=0, silu(1)=1/(1+e^-1)=0.7310586, silu(-1)=-1/(1+e^1)=-0.2689414
        float[] expected = {0f, 1.4621172f, -0.5378828f};
        assertArrayEquals(expected, dst, DELTA);
    }

    @Test
    void rope_rotatesFirstAndSecondHalfTogether() {
        // nHeads=1, headDim=4, half=2, pos=1, freqBase=10000 (the GGUF/llama
        // default). x = [1, 0, 1, 0] so the rotation's effect on each output
        // component is easy to attribute to a specific theta.
        float[] x = {1f, 0f, 1f, 0f};
        GGUFVectorOps.rope(x, 1, 4, 1, 10000f);

        // theta_0 = 1 / 10000^(0/4) = 1        -> cos=0.5403, sin=0.8415
        // theta_1 = 1 / 10000^(2/4) = 1/100     -> cos=0.99995, sin=0.01 (rotates x[1]/x[3], both 0 here)
        float[] expected = {-0.30116868f, 0f, 1.3817733f, 0f};
        assertArrayEquals(expected, x, DELTA);
    }

    @Test
    void rope_atPositionZeroIsIdentity() {
        // theta = 0/freqBase^k = 0 for every dimension at pos=0, so cos=1,
        // sin=0 everywhere -- RoPE at the first position must be a no-op.
        // This is the case GgufDecodeStep hits on every prompt's first
        // token, so it's worth pinning down explicitly.
        float[] x = {1f, 2f, 3f, 4f};
        float[] original = x.clone();

        GGUFVectorOps.rope(x, 1, 4, 0, 10000f);

        assertArrayEquals(original, x, DELTA);
    }
}
