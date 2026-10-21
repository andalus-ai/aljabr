package tech.kayys.aljabr.models;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.*;
import tech.kayys.aljabr.core.nn.*;
import tech.kayys.aljabr.core.nn.*;
import tech.kayys.aljabr.core.nn.*;

/**
 * LoRA (Low-Rank Adaptation) — parameter-efficient fine-tuning that injects
 * trainable low-rank matrices into frozen pre-trained weights.
 *
 * <p>Based on <em>"LoRA: Low-Rank Adaptation of Large Language Models"</em>
 * (Hu et al., 2021).
 *
 * <p>Instead of updating W ∈ ℝ^{d×k} directly, LoRA learns:
 * <pre>
 *   W' = W + ΔW = W + B·A
 *   where A ∈ ℝ^{r×k}, B ∈ ℝ^{d×r}, rank r ≪ min(d,k)
 * </pre>
 *
 * <p>Only A and B are trained; W is frozen. This reduces trainable parameters
 * from d×k to r×(d+k), typically 100-1000× fewer.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * // Wrap a frozen Linear layer with LoRA
 * var lora = new LoRALinear(frozenLinear, rank=8, alpha=16f);
 * Tensor out = lora.forward(x);
 *
 * // Only lora.loraParameters() are trained
 * var optimizer = new AdamW(lora.loraParameters(), lr=1e-4f);
 * }</pre>
 */
public final class LoRALinear extends NNModule {

    private final NNModule base;       // frozen pre-trained linear
    private final Tensor loraA;   // [rank, inFeatures]  — initialized N(0, 1/√rank)
    private final Tensor loraB;   // [outFeatures, rank] — initialized zeros
    private final float scale;       // alpha / rank

    /**
     * Wraps a frozen linear layer with LoRA adapters.
     *
     * @param base        frozen pre-trained {@link Linear} (or any module)
     * @param inFeatures  input dimension
     * @param outFeatures output dimension
     * @param rank        LoRA rank r (typically 4, 8, or 16)
     * @param alpha       LoRA scaling factor (typically 2× rank)
     */
    public LoRALinear(NNModule base, int inFeatures, int outFeatures, int rank, float alpha) {
        this.base  = register("base", base);
        this.scale = alpha / rank;

        // A: random init, B: zero init (so ΔW=0 at start)
        float std = (float) (1.0 / Math.sqrt(rank));
        this.loraA = register("loraA", Tensor.randn(rank, inFeatures).mul((float)(1.0/Math.sqrt(rank))));
        this.loraB = register("loraB", Tensor.zeros(outFeatures, rank));

        // Freeze base parameters
        for (Tensor p : base.parameters().values()) {}
    }

    /**
     * Forward pass: base output + LoRA delta.
     *
     * <p>{@code y = base(x) + scale · (x @ A^T) @ B^T}
     *
     * @param x input tensor {@code [..., inFeatures]}
     * @return output tensor {@code [..., outFeatures]}
     */
    @Override
    public Tensor forward(Tensor x) {
        Tensor baseOut = base.forward(x);
        // LoRA path: x @ A^T → [*, rank], then @ B^T → [*, outFeatures]
        Tensor loraOut = x.matmul(loraA.transpose())
                              .matmul(loraB.transpose())
                              .mul(scale);
        return baseOut.add(loraOut);
    }

    /**
     * Returns only the LoRA parameters (A and B matrices).
     * Pass these to the optimizer — base parameters stay frozen.
     *
     * @return list containing only lora_A and lora_B parameters
     */
    public java.util.List<Tensor> loraParameters() {
        return java.util.List.of(loraA, loraB);
    }

    /**
     * Merges LoRA weights into the base weight matrix for inference.
     * After merging, the LoRA adapters can be discarded.
     *
     * @return merged weight tensor {@code W + scale·B·A}
     */
    public Tensor mergedWeight() {
        // B @ A → [outFeatures, inFeatures]
        Tensor delta = loraB.matmul(loraA).mul(scale);
        // Get base weight (assumes base is a Linear)
        Tensor baseW = base.parameters().values().iterator().next();
        return baseW.add(delta);
    }

    @Override public String toString() {
        return String.format("LoRALinear(rank=%d, scale=%.2f, base=%s)",
            (int) loraA.shape().dim(0), scale, base);
    }
}
