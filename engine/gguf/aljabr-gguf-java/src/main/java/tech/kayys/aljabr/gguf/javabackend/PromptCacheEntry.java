package tech.kayys.aljabr.gguf.javabackend;

/**
 * Lightweight snapshot of prompt tokens, KV cache layers, and output logits.
 */
final class PromptCacheEntry {
    final int[] tokens;
    final float[][] k;
    final float[][] v;
    final float[] logits;

    PromptCacheEntry(int[] tokens, float[][] k, float[][] v, float[] logits) {
        this.tokens = tokens;
        this.k = k;
        this.v = v;
        this.logits = logits;
    }
}
