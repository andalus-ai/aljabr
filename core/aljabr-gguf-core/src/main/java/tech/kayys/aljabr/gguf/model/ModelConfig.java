package tech.kayys.aljabr.gguf.model;

import tech.kayys.aljabr.gguf.loader.gguf.GGUFFile;
/**
 * Configuration class for Model settings and properties.
 *
 * @author bhangun
 * @since 0.1.0
 */


public record ModelConfig(int contextLength, int embeddingDim, int nLayers, int ffnDim,
                         int nHeads, int nKVHeads, int headDim, int vocabSize,
                         float rmsNormEps, float ropeFreqBase, int bosTokenId, int eosTokenId) {
    
    public static ModelConfig fromGGUF(GGUFFile f) {
        String arch = f.getString("general.architecture", "llama");
        
        int ctxLen = (int) f.getLong(arch + ".context_length", 4096);
        int embDim = (int) f.getLong(arch + ".embedding_length", 4096);
        int nLayers = (int) f.getLong(arch + ".block_count", 32);
        int ffnDim = (int) f.getLong(arch + ".feed_forward_length", 11008);
        int nHeads = (int) f.getLong(arch + ".attention.head_count", 32);
        int nKVHeads = (int) f.getLong(arch + ".attention.head_count_kv", nHeads);
        float eps = f.getFloat(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
        float freq = f.getFloat(arch + ".rope.freq_base", 10000.0f);
        int vocab = (int) f.getLong(arch + ".vocab_size", f.getArray("tokenizer.ggml.tokens").size());
        int bos = (int) f.getLong("tokenizer.ggml.bos_token_id", 1);
        int eos = (int) f.getLong("tokenizer.ggml.eos_token_id", 2);
        
        int headDim = (int) f.getLong(arch + ".attention.key_length", embDim / nHeads);
        
        return new ModelConfig(ctxLen, embDim, nLayers, ffnDim, nHeads, nKVHeads,
                               headDim, vocab, eps, freq, bos, eos);
    }
    
    public int kvGroupSize() { return nHeads / nKVHeads; }
}
