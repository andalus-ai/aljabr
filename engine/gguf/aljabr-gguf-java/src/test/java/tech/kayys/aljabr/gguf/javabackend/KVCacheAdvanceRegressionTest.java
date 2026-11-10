package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.inference.KVCache;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * Regression test for the bug found while porting {@code LlamaForward}'s
 * algorithm into {@link GgufDecodeStep}: the ported-from code called
 * {@link KVCache#store} for every layer at every position but never called
 * {@link KVCache#advance()} — since {@code store} writes at the cache's own
 * internal {@code seqLen}, which only moves forward via {@code advance()},
 * every token was silently overwriting position 0 instead of accumulating
 * history. Wrong output, no exception — the kind of bug that survives a
 * "does it produce plausible-looking text" smoke test.
 *
 * <p>This test proves the fix: writing at two positions with
 * {@code advance()} called between them, both positions' data survive in
 * the cache afterward. If {@code advance()} were removed again (or a future
 * edit re-introduces the same mistake in a different call site), the second
 * assertion below fails immediately instead of the bug hiding until
 * multi-token generation looks subtly wrong.</p>
 */
class KVCacheAdvanceRegressionTest {

    @Test
    void storeThenAdvance_accumulatesAcrossPositions_doesNotOverwrite() {
        int layers = 1;
        int maxSeq = 4;
        int kvHeads = 1;
        int headDim = 2;
        KVCache cache = new KVCache(layers, maxSeq, kvHeads, headDim);

        float[] kPos0 = {1f, 1f};
        float[] vPos0 = {10f, 10f};
        cache.store(0, kPos0, vPos0);
        cache.advance(); // the call that was missing in the ported-from code

        float[] kPos1 = {2f, 2f};
        float[] vPos1 = {20f, 20f};
        cache.store(0, kPos1, vPos1);
        cache.advance();

        float[] kLayer0 = cache.kLayer(0);
        float[] vLayer0 = cache.vLayer(0);
        int stride = cache.headStride();

        // Position 0's data must still be there...
        assertArrayEquals(kPos0, slice(kLayer0, 0, stride));
        assertArrayEquals(vPos0, slice(vLayer0, 0, stride));

        // ...and position 1's data must be in the NEXT slot, not overwriting
        // position 0. Without cache.advance(), both stores would land at
        // offset 0 and this assertion (and the one above, since kPos1 would
        // have clobbered kPos0) would fail.
        assertArrayEquals(kPos1, slice(kLayer0, stride, stride));
        assertArrayEquals(vPos1, slice(vLayer0, stride, stride));
    }

    private static float[] slice(float[] array, int offset, int length) {
        float[] out = new float[length];
        System.arraycopy(array, offset, out, 0, length);
        return out;
    }
}
