package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFParser;
import tech.kayys.aljabr.gguf.loader.GGUFReader;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

class GemmaInspectTest {

    @Test
    void inspectGemma4() throws Exception {
        Path path = Path.of("/Users/bhangun/.aqli/models/blobs/unsloth/gemma-4-12b-it-GGUF/unsloth__gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf");
        if (!path.toFile().exists()) {
            return;
        }

        Arena arena = Arena.ofShared();
        try (GGUFReader reader = new GGUFReader(path, arena)) {
            GGUFModel model = new GGUFParser().parse(reader.segment(), arena);
            for (Map.Entry<String, Object> entry : model.metadata().entrySet()) {
                if (entry.getKey().contains("chat_template") || !entry.getKey().startsWith("tokenizer.")) {
                    System.out.println("Meta: " + entry.getKey() + " = " + entry.getValue());
                }
            }
            for (GGUFTensorInfo t : model.tensors()) {
                String n = t.name();
                if (n.startsWith("blk.0.") || n.startsWith("blk.1.") || n.startsWith("blk.5.")) {
                    System.out.println("Layer Tensor: " + n + ", shape=" + Arrays.toString(t.shape()));
                }
                if (!n.startsWith("blk.")) {
                    System.out.println("Global Tensor: " + n + ", typeId=" + t.typeId() + ", shape=" + Arrays.toString(t.shape()));
                }
            }
            for (int i = 0; i < 48; i++) {
                int layer = i;
                GGUFTensorInfo sc = model.tensors().stream().filter(t -> ("blk." + layer + ".layer_output_scale.weight").equals(t.name())).findFirst().orElse(null);
                if (sc != null) {
                    float[] v = new float[1];
                    tech.kayys.aljabr.gguf.runtime.GgufTensorOps.dequantizeRow(model, sc, 0L, v);
                    System.out.printf("Layer %d scale: typeId=%d, shape=%s, val=%.6f%n", layer, sc.typeId(), Arrays.toString(sc.shape()), v[0]);
                }
            }
            GGUFTensorInfo q0 = model.tensors().stream().filter(t -> "blk.0.attn_q.weight".equals(t.name())).findFirst().orElse(null);
            if (q0 != null) {
                int K = (int) tech.kayys.aljabr.gguf.runtime.GgufTensorOps.matrixColumns(q0);
                int N = (int) tech.kayys.aljabr.gguf.runtime.GgufTensorOps.matrixRows(q0);
                System.out.println("Testing blk.0.attn_q.weight K=" + K + ", N=" + N);
                float[] in = new float[K];
                for (int i = 0; i < K; i++) in[i] = 1.0f / (1.0f + i);
                float[] outCpu = new float[N];
                float[] outMetal = new float[N];
                long t0 = System.nanoTime();
                tech.kayys.aljabr.gguf.runtime.GgufTensorOps.matVecParallel(model, q0, in, outCpu);
                long tCpu = System.nanoTime() - t0;
                for (int iter = 0; iter < 5; iter++) {
                    long s0 = System.nanoTime();
                    GgufMetalBridge.matVec(model, q0, in, outMetal);
                    long sMetal = System.nanoTime() - s0;
                    System.out.printf("Metal call %d: %.2f ms%n", iter, sMetal / 1e6);
                }
                float maxDiff = 0.0f;
                for (int i = 0; i < N; i++) {
                    maxDiff = Math.max(maxDiff, Math.abs(outCpu[i] - outMetal[i]));
                }
                System.out.println("Max diff CPU vs Metal: " + maxDiff);
                System.out.println("CPU first 5: " + Arrays.toString(Arrays.copyOf(outCpu, 5)));
                System.out.println("Metal first 5: " + Arrays.toString(Arrays.copyOf(outMetal, 5)));

                GGUFTensorInfo ffnGate = model.tensors().stream().filter(t -> "blk.0.ffn_gate.weight".equals(t.name())).findFirst().orElse(null);
                if (ffnGate != null) {
                    int kG = (int) tech.kayys.aljabr.gguf.runtime.GgufTensorOps.matrixColumns(ffnGate);
                    int nG = (int) tech.kayys.aljabr.gguf.runtime.GgufTensorOps.matrixRows(ffnGate);
                    float[] inG = new float[kG];
                    float[] outG = new float[nG];
                    for (int iter = 0; iter < 5; iter++) {
                        long s0 = System.nanoTime();
                        GgufMetalBridge.matVec(model, ffnGate, inG, outG);
                        System.out.printf("ffn_gate (K=%d, N=%d, type=%d) iter %d: %.2f ms%n", kG, nG, ffnGate.typeId(), iter, (System.nanoTime() - s0) / 1e6);
                    }
                }
                GGUFTensorInfo ffnDown = model.tensors().stream().filter(t -> "blk.0.ffn_down.weight".equals(t.name())).findFirst().orElse(null);
                if (ffnDown != null) {
                    int kD = (int) tech.kayys.aljabr.gguf.runtime.GgufTensorOps.matrixColumns(ffnDown);
                    int nD = (int) tech.kayys.aljabr.gguf.runtime.GgufTensorOps.matrixRows(ffnDown);
                    float[] inD = new float[kD];
                    for (int i = 0; i < kD; i++) inD[i] = 1.0f / (1.0f + i);
                    float[] outCpuD = new float[nD];
                    float[] outD = new float[nD];
                    tech.kayys.aljabr.gguf.runtime.GgufTensorOps.matVecParallel(model, ffnDown, inD, outCpuD);
                    for (int iter = 0; iter < 5; iter++) {
                        long s0 = System.nanoTime();
                        GgufMetalBridge.matVec(model, ffnDown, inD, outD);
                        System.out.printf("ffn_down (K=%d, N=%d, type=%d) iter %d: %.2f ms%n", kD, nD, ffnDown.typeId(), iter, (System.nanoTime() - s0) / 1e6);
                    }
                    float maxDiffD = 0.0f;
                    for (int i = 0; i < nD; i++) maxDiffD = Math.max(maxDiffD, Math.abs(outCpuD[i] - outD[i]));
                    System.out.println("Max diff CPU vs Metal (Q6_K ffn_down): " + maxDiffD);
                }
                GGUFTensorInfo ffnUp = model.tensors().stream().filter(t -> "blk.0.ffn_up.weight".equals(t.name())).findFirst().orElse(null);
                if (ffnGate != null && ffnUp != null && ffnDown != null) {
                    float[] inA = new float[3840];
                    for (int i = 0; i < 3840; i++) inA[i] = 1.0f / (1.0f + i);
                    float[] outDown = new float[3840];
                    for (int l : new int[]{0, 1, 2, 3, 4, 5}) {
                        GGUFTensorInfo g = model.tensors().stream().filter(t -> ("blk." + l + ".ffn_gate.weight").equals(t.name())).findFirst().orElse(null);
                        GGUFTensorInfo u = model.tensors().stream().filter(t -> ("blk." + l + ".ffn_up.weight").equals(t.name())).findFirst().orElse(null);
                        GGUFTensorInfo d = model.tensors().stream().filter(t -> ("blk." + l + ".ffn_down.weight").equals(t.name())).findFirst().orElse(null);
                        for (int iter = 0; iter < 3; iter++) {
                            long s0 = System.nanoTime();
                            boolean ok = GgufMetalBridge.ffnGatedFused(model, g, u, d, inA, outDown, true);
                            System.out.printf("Layer %d ffnGatedFused iter %d: %.2f ms (ok=%b)%n", l, iter, (System.nanoTime() - s0) / 1e6, ok);
                        }
                    }
                }
                tech.kayys.aljabr.gguf.tokenizer.GGUFTokenizer tok = new tech.kayys.aljabr.gguf.tokenizer.GGUFTokenizer(model);
                String promptText = "<bos><|turn>user\nwhat is 2*9<turn|>\n<|turn>model\n";
                System.out.println("Formatted prompt: " + promptText);
                long[] enc = tok.encode(promptText, tech.kayys.aljabr.spi.tokenizer.EncodeOptions.builder().addBos(tok.shouldAddBos()).build());
                System.out.println("Encoded tokens (" + enc.length + "): " + java.util.Arrays.toString(enc));
                for (long id : enc) {
                    System.out.println("  token " + id + " -> '" + tok.decode(new long[]{id}, tech.kayys.aljabr.spi.tokenizer.DecodeOptions.builder().build()) + "'");
                }

                GgufModelConfig cfg = GgufModelConfig.from(model);

                // ── 1-token sanity check (no bidirectionality; should equal llama.cpp for single token) ──
                System.out.println("\n=== 1-token forward() sanity check (token=2 = <bos>) ===");
                {
                    GgufDecodeStep step1 = new GgufDecodeStep(model, cfg);
                    tech.kayys.aljabr.gguf.loader.inference.KVCache cache1 =
                            new tech.kayys.aljabr.gguf.loader.inference.KVCache(
                                    cfg.nLayers(), cfg.contextLength(), cfg.nKVHeadsPerLayer(), cfg.headDimPerLayer());
                    long tFwd0 = System.currentTimeMillis();
                    float[] logits1 = step1.forward(2, 0, cache1, true);  // bos token at pos=0
                    long tFwdDuration = System.currentTimeMillis() - tFwd0;
                    System.out.printf(">>> Single token forward took: %d ms <<<%n", tFwdDuration);
                    int best1 = 0; float bval1 = Float.NEGATIVE_INFINITY;
                    for (int i = 0; i < logits1.length; i++) {
                        if (logits1[i] > bval1) { bval1 = logits1[i]; best1 = i; }
                    }
                    System.out.printf("  top-1 id=%d ('%s') val=%.3f%n", best1,
                            tok.decode(new long[]{best1}, tech.kayys.aljabr.spi.tokenizer.DecodeOptions.builder().build()), bval1);
                    // top-5
                    int[] t5 = new int[5]; float[] v5 = new float[5]; java.util.Arrays.fill(v5, Float.NEGATIVE_INFINITY);
                    for (int i = 0; i < logits1.length; i++) {
                        if (logits1[i] > v5[4]) {
                            t5[4]=i; v5[4]=logits1[i];
                            for (int j=4; j>0 && v5[j]>v5[j-1]; j--) { int ti=t5[j]; float tv=v5[j]; t5[j]=t5[j-1]; v5[j]=v5[j-1]; t5[j-1]=ti; v5[j-1]=tv; }
                        }
                    }
                    for (int i=0; i<5; i++) System.out.printf("  top%d: id=%d val=%.3f '%s'%n", i, t5[i], v5[i],
                            tok.decode(new long[]{t5[i]}, tech.kayys.aljabr.spi.tokenizer.DecodeOptions.builder().build()));
                    System.out.println("=======================================================\n");
                }


                // ── Diagnostic: dump key config params ────────────────────────────
                System.out.println("\n=== GgufModelConfig ===");
                System.out.printf("  arch=%s nLayers=%d embDim=%d nHeads=%d nKVHeads=%d%n",
                        cfg.architecture(), cfg.nLayers(), cfg.embeddingDim(), cfg.nHeads(), cfg.nKVHeads());
                System.out.printf("  headDim=%d headDimKV=%d ffnDim=%d%n",
                        cfg.headDim(), cfg.headDimKV(), cfg.ffnDim());
                System.out.printf("  ropeFreqBase=%.0f ropeFreqBaseSwa=%.0f%n",
                        cfg.ropeFreqBase(), cfg.ropeFreqBaseSwa());
                System.out.printf("  attentionScale=%.4f embeddingScale=%.4f finalSoftcap=%.1f%n",
                        cfg.attentionScale(), cfg.embeddingScale(), cfg.finalLogitSoftcap());
                System.out.printf("  hasUnitVNorm=%b isGeGLU=%b isGemma4=%b%n",
                        cfg.hasUnitVNorm(), cfg.isGeGLU(), cfg.isGemma4());
                System.out.printf("  slidingWindow=%d nEmbdPerLayer=%d%n",
                        cfg.slidingWindow(), cfg.nEmbdPerLayer());
                if (cfg.swaPattern() != null) {
                    System.out.println("  swaPattern[0..11]: " + java.util.Arrays.toString(java.util.Arrays.copyOf(cfg.swaPattern(), 12)));
                }
                if (cfg.nKVHeadsPerLayer() != null) {
                    System.out.println("  nKVHeadsPerLayer[0..11]: " + java.util.Arrays.toString(java.util.Arrays.copyOf(cfg.nKVHeadsPerLayer(), 12)));
                }
                if (cfg.headDimPerLayer() != null) {
                    System.out.println("  headDimPerLayer[0..11]: " + java.util.Arrays.toString(java.util.Arrays.copyOf(cfg.headDimPerLayer(), 12)));
                }
                System.out.println("======================\n");
                // ──────────────────────────────────────────────────────────────────

                GgufDecodeStep step = new GgufDecodeStep(model, cfg);
                tech.kayys.aljabr.gguf.loader.inference.KVCache cache;
                if (cfg.nKVHeadsPerLayer() != null && cfg.headDimPerLayer() != null) {
                    cache = new tech.kayys.aljabr.gguf.loader.inference.KVCache(cfg.nLayers(), cfg.contextLength(), cfg.nKVHeadsPerLayer(), cfg.headDimPerLayer());
                } else {
                    cache = new tech.kayys.aljabr.gguf.loader.inference.KVCache(cfg.nLayers(), cfg.contextLength(), cfg.nKVHeads(), cfg.headDim());
                }

                int[] encInt = new int[enc.length];
                for (int i = 0; i < enc.length; i++) encInt[i] = (int) enc[i];

                System.out.println("Running batch prefill (" + enc.length + " tokens)...");
                long prefillStart = System.currentTimeMillis();
                float[] logits = step.prefill(encInt, cache);
                System.out.printf("Prefill done in %.2f s%n", (System.currentTimeMillis() - prefillStart) / 1000.0);

                if (logits != null) {
                    int[] topIdx = new int[10];
                    float[] topVal = new float[10];
                    java.util.Arrays.fill(topVal, Float.NEGATIVE_INFINITY);
                    for (int i = 0; i < logits.length; i++) {
                        if (logits[i] > topVal[9]) {
                            topIdx[9] = i; topVal[9] = logits[i];
                            for (int j = 9; j > 0 && topVal[j] > topVal[j-1]; j--) {
                                int ti = topIdx[j]; float tv = topVal[j];
                                topIdx[j] = topIdx[j-1]; topVal[j] = topVal[j-1];
                                topIdx[j-1] = ti; topVal[j-1] = tv;
                            }
                        }
                    }
                    System.out.println("AFTER PROMPT Top-10 logits: " + java.util.Arrays.toString(topIdx));
                    for (int i = 0; i < 10; i++) {
                        System.out.printf("  #%d: id=%d ('%s') val=%.3f%n", i, topIdx[i],
                                tok.decode(new long[]{topIdx[i]}, tech.kayys.aljabr.spi.tokenizer.DecodeOptions.builder().build()),
                                topVal[i]);
                    }

                    // Greedy generation of 10 tokens
                    System.out.println("\n--- Greedy generation ---");
                    int nextToken = topIdx[0];
                    for (int gen = 0; gen < 10; gen++) {
                        int pos = enc.length + gen;
                        String piece = tok.decode(new long[]{nextToken}, tech.kayys.aljabr.spi.tokenizer.DecodeOptions.builder().build());
                        System.out.printf("Gen %d (id=%d): '%s'%n", gen, nextToken, piece);
                        long tGen0 = System.currentTimeMillis();
                        logits = step.forward(nextToken, pos, cache, true);
                        long tGen = System.currentTimeMillis() - tGen0;
                        System.out.printf(">>> Gen %d forward took: %d ms (%.2f tok/s) <<<%n", gen, tGen, 1000.0 / Math.max(1, tGen));
                        int[] genTop = new int[5];
                        float[] genVal = new float[5];
                        java.util.Arrays.fill(genVal, Float.NEGATIVE_INFINITY);
                        for (int i = 0; i < logits.length; i++) {
                            if (logits[i] > genVal[4]) {
                                genVal[4] = logits[i];
                                genTop[4] = i;
                                for (int j = 4; j > 0 && genVal[j] > genVal[j - 1]; j--) {
                                    int ti = genTop[j]; float tv = genVal[j];
                                    genTop[j] = genTop[j - 1]; genVal[j] = genVal[j - 1];
                                    genTop[j - 1] = ti; genVal[j - 1] = tv;
                                }
                            }
                        }
                        for (int k = 0; k < 5; k++) {
                            System.out.printf("   #%d: id=%d val=%.3f '%s'%n", k, genTop[k], genVal[k],
                                    tok.decode(new long[]{genTop[k]}, tech.kayys.aljabr.spi.tokenizer.DecodeOptions.builder().build()));
                        }
                        nextToken = genTop[0];
                    }
                }
            }
        } finally {
            arena.close();
        }
    }
}
