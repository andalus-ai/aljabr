package tech.kayys.aljabr.gguf.loader.model;

import tech.kayys.aljabr.gguf.loader.gguf.GGUFFile;
import tech.kayys.aljabr.gguf.core.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.loader.quant.Dequantizer;
/**
 * 
 * Core class for kayys module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public class LlamaWeights {
    public float[] tokenEmbed;
    public float[] outputNorm;
    public GGUFTensorInfo outputWeight;
    public float[][] attnNorm;
    public GGUFTensorInfo[] wQ, wK, wV, wO;
    public float[][] ffnNorm;
    public GGUFTensorInfo[] ffnGate, ffnUp, ffnDown;
    
    public static LlamaWeights load(GGUFFile f, ModelConfig cfg) {
        int L = cfg.nLayers();
        var w = new LlamaWeights();
        
        w.tokenEmbed = Dequantizer.dequantize(f.tensor("token_embd.weight"));
        w.outputNorm = Dequantizer.dequantize(f.tensor("output_norm.weight"));
        w.outputWeight = f.findTensor("output.weight").orElse(f.tensor("token_embd.weight"));
        
        w.attnNorm = new float[L][];
        w.wQ = new GGUFTensorInfo[L];
        w.wK = new GGUFTensorInfo[L];
        w.wV = new GGUFTensorInfo[L];
        w.wO = new GGUFTensorInfo[L];
        w.ffnNorm = new float[L][];
        w.ffnGate = new GGUFTensorInfo[L];
        w.ffnUp = new GGUFTensorInfo[L];
        w.ffnDown = new GGUFTensorInfo[L];
        
        for (int i = 0; i < L; i++) {
            String p = "blk." + i + ".";
            w.attnNorm[i] = Dequantizer.dequantize(f.tensor(p + "attn_norm.weight"));
            w.wQ[i] = f.tensor(p + "attn_q.weight");
            w.wK[i] = f.tensor(p + "attn_k.weight");
            w.wV[i] = f.tensor(p + "attn_v.weight");
            w.wO[i] = f.tensor(p + "attn_output.weight");
            w.ffnNorm[i] = Dequantizer.dequantize(f.tensor(p + "ffn_norm.weight"));
            w.ffnGate[i] = f.findTensor(p + "ffn_gate.weight").orElse(null);
            w.ffnUp[i] = f.tensor(p + "ffn_up.weight");
            w.ffnDown[i] = f.tensor(p + "ffn_down.weight");
        }
        return w;
    }
}
