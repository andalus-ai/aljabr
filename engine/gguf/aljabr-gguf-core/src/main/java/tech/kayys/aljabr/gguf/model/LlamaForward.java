package tech.kayys.aljabr.gguf.loader.model;

import tech.kayys.aljabr.gguf.loader.inference.KVCache;
import tech.kayys.aljabr.gguf.core.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.loader.tensor.GGUFVectorOps;
/**
 * 
 * Core class for kayys module.
 *
 * @author bhangun
 * @since 0.1.0
 */
public class LlamaForward {
    private final ModelConfig cfg;
    private final LlamaWeights w;
    private final float[] q, k, v, attnFull, attnOut, xb, gate, up, hb, scores;
    
    public LlamaForward(ModelConfig cfg, LlamaWeights w) {
        this.cfg = cfg;
        this.w = w;
        int dim = cfg.embeddingDim();
        int ffnDim = cfg.ffnDim();
        int nH = cfg.nHeads();
        int nKVH = cfg.nKVHeads();
        int hDim = cfg.headDim();
        
        this.q = new float[nH * hDim];
        this.k = new float[nKVH * hDim];
        this.v = new float[nKVH * hDim];
        this.attnFull = new float[nH * hDim];
        this.attnOut = new float[hDim];
        this.xb = new float[dim];
        this.gate = new float[ffnDim];
        this.up = new float[ffnDim];
        this.hb = new float[ffnDim];
        this.scores = new float[cfg.contextLength()];
    }
    
    public float[] forward(int tokenId, int pos, KVCache cache, float[] x) {
        int dim = cfg.embeddingDim();
        int nH = cfg.nHeads();
        int nKVH = cfg.nKVHeads();
        int hDim = cfg.headDim();
        int ffnDim = cfg.ffnDim();
        float eps = cfg.getRmsNormEps();
        
        // Embedding
        GGUFVectorOps.embedLookup(w.tokenEmbed, tokenId, dim, x);
        
        for (int layer = 0; layer < cfg.nLayers(); layer++) {
            // Attention RMSNorm
            GGUFVectorOps.rmsNorm(x, w.attnNorm[layer], xb, dim, eps);
            
            // Q, K, V projections
            GGUFVectorOps.matVecMulQuant(w.wQ[layer], xb, q, hb); // Reusing hb as scratch
            GGUFVectorOps.matVecMulQuant(w.wK[layer], xb, k, hb);
            GGUFVectorOps.matVecMulQuant(w.wV[layer], xb, v, hb);
            
            // RoPE
            GGUFVectorOps.rope(q, nH, hDim, pos, cfg.ropeFreqBase());
            GGUFVectorOps.rope(k, nKVH, hDim, pos, cfg.ropeFreqBase());
            
            // Store KV
            cache.store(layer, k, v);
            
            // Attention
            float scale = (float)(1.0 / Math.sqrt(hDim));
            int seqLen = pos + 1;
            
            GGUFVectorOps.zero(attnFull, nH * hDim);
            
            for (int h = 0; h < nH; h++) {
                int kvH = h / (nH / nKVH);
                int qOff = h * hDim;
                
                // Scores
                float[] kCache = cache.kLayer(layer);
                for (int t = 0; t < seqLen; t++) {
                    int kBase = t * cache.headStride() + kvH * hDim;
                    scores[t] = GGUFVectorOps.dot(q, qOff, kCache, kBase, hDim) * scale;
                }
                
                // Softmax
                GGUFVectorOps.softmax(scores, seqLen);
                
                // Weighted sum
                GGUFVectorOps.zero(attnOut, hDim);
                float[] vCache = cache.vLayer(layer);
                for (int t = 0; t < seqLen; t++) {
                    float wt = scores[t];
                    int vBase = t * cache.headStride() + kvH * hDim;
                    GGUFVectorOps.scaledAdd(attnOut, attnOut, wt, vCache, vBase, hDim);
                }
                
                System.arraycopy(attnOut, 0, attnFull, h * hDim, hDim);
            }
            
            // Output projection
            GGUFVectorOps.matVecMulQuant(w.wO[layer], attnFull, xb, hb);
            
            // Residual
            GGUFVectorOps.addInPlace(x, xb, dim);
            
            // FFN RMSNorm
            GGUFVectorOps.rmsNorm(x, w.ffnNorm[layer], xb, dim, eps);
            
            // FFN
            if (w.ffnGate[layer] != null) {
                GGUFVectorOps.matVecMulQuant(w.ffnGate[layer], xb, gate, hb);
                GGUFVectorOps.matVecMulQuant(w.ffnUp[layer], xb, up, hb);
                GGUFVectorOps.swiGLU(gate, up, hb, ffnDim);
            } else {
                GGUFVectorOps.matVecMulQuant(w.ffnUp[layer], xb, up, hb);
                GGUFVectorOps.siluInPlace(up, ffnDim); // Actually just silu
                System.arraycopy(up, 0, hb, 0, ffnDim);
            }
            
            GGUFVectorOps.matVecMulQuant(w.ffnDown[layer], hb, xb, gate); // Reusing gate as scratch
            GGUFVectorOps.addInPlace(x, xb, dim);
        }
        
        // Final RMSNorm
        GGUFVectorOps.rmsNorm(x, w.outputNorm, xb, dim, eps);
        
        // Output projection
        float[] logits = new float[cfg.getVocabSize()];
        GGUFVectorOps.matVecMulQuant(w.outputWeight, xb, logits, hb);
        return logits;
    }
}
