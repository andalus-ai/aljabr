package tech.kayys.aljabr.gguf.loader.sampler;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
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
public class Sampler {
    private final float temperature;
    private final int topK;
    private final float topP;
    private final float repeatPenalty;
    
    public Sampler(float temperature, int topK, float topP, float repeatPenalty) {
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
        this.repeatPenalty = repeatPenalty;
    }
    
    public int sample(float[] logits, List<Integer> previousTokens) {
        int n = logits.length;
        if (n == 0) return 0;

        // Greedy fast-path when temperature <= 0
        if (temperature <= 0.0f) {
            if (repeatPenalty == 1.0f || previousTokens == null || previousTokens.isEmpty()) {
                float bestVal = logits[0];
                int bestIdx = 0;
                for (int i = 1; i < n; i++) {
                    if (logits[i] > bestVal) {
                        bestVal = logits[i];
                        bestIdx = i;
                    }
                }
                return bestIdx;
            }
            float[] probs = logits.clone();
            for (int token : previousTokens) {
                if (token >= 0 && token < n) {
                    if (probs[token] < 0) probs[token] *= repeatPenalty;
                    else probs[token] /= repeatPenalty;
                }
            }
            float bestVal = probs[0];
            int bestIdx = 0;
            for (int i = 1; i < n; i++) {
                if (probs[i] > bestVal) {
                    bestVal = probs[i];
                    bestIdx = i;
                }
            }
            return bestIdx;
        }

        float[] probs = logits.clone();
        
        // Apply repetition penalty only on tokens in previousTokens
        if (repeatPenalty != 1.0f && previousTokens != null && !previousTokens.isEmpty()) {
            for (int token : previousTokens) {
                if (token >= 0 && token < n) {
                    if (probs[token] < 0) probs[token] *= repeatPenalty;
                    else probs[token] /= repeatPenalty;
                }
            }
        }
        
        // Apply temperature
        float invTemp = 1.0f / temperature;
        for (int i = 0; i < n; i++) {
            probs[i] *= invTemp;
        }
        
        // Top-K selection using fixed-size min-heap: O(N log K) instead of O(N log N)
        if (topK > 0 && topK < n) {
            int k = topK;
            float[] heapVal = new float[k];
            int[] heapIdx = new int[k];
            
            // Fill initial k elements
            for (int i = 0; i < k; i++) {
                heapVal[i] = probs[i];
                heapIdx[i] = i;
            }
            // Build min-heap
            for (int i = (k / 2) - 1; i >= 0; i--) {
                siftDown(heapVal, heapIdx, i, k);
            }
            
            // Scan remaining elements
            for (int i = k; i < n; i++) {
                float v = probs[i];
                if (v > heapVal[0]) {
                    heapVal[0] = v;
                    heapIdx[0] = i;
                    siftDown(heapVal, heapIdx, 0, k);
                }
            }
            
            // Softmax over top-K elements ONLY
            float max = heapVal[0];
            for (int j = 1; j < k; j++) {
                if (heapVal[j] > max) max = heapVal[j];
            }
            float sum = 0;
            for (int j = 0; j < k; j++) {
                heapVal[j] = (float) Math.exp(heapVal[j] - max);
                sum += heapVal[j];
            }
            float invSum = 1.0f / sum;
            for (int j = 0; j < k; j++) {
                heapVal[j] *= invSum;
            }
            
            // Sample from top-K
            double r = ThreadLocalRandom.current().nextDouble();
            double cum = 0;
            for (int j = 0; j < k; j++) {
                cum += heapVal[j];
                if (r < cum) return heapIdx[j];
            }
            return heapIdx[k - 1];
        }
        
        // Full vocabulary softmax fallback when topK is not set
        float max = probs[0];
        for (float p : probs) if (p > max) max = p;
        float sum = 0;
        for (int i = 0; i < n; i++) {
            probs[i] = (float) Math.exp(probs[i] - max);
            sum += probs[i];
        }
        float invSum = 1.0f / sum;
        for (int i = 0; i < n; i++) probs[i] *= invSum;
        
        double r = ThreadLocalRandom.current().nextDouble();
        double cum = 0;
        for (int i = 0; i < n; i++) {
            cum += probs[i];
            if (r < cum) return i;
        }
        return 0;
    }

    private static void siftDown(float[] val, int[] idx, int i, int n) {
        int half = n >>> 1;
        while (i < half) {
            int child = (i << 1) + 1;
            int right = child + 1;
            if (right < n && val[right] < val[child]) {
                child = right;
            }
            if (val[i] <= val[child]) {
                break;
            }
            float tempV = val[i];
            val[i] = val[child];
            val[child] = tempV;
            int tempI = idx[i];
            idx[i] = idx[child];
            idx[child] = tempI;
            i = child;
        }
    }
}
