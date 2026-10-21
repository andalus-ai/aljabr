package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Low-Rank Adaptation (LoRA) layer.
 * 
 * <p>
 * Wraps a base Linear layer and injects trainable low-rank matrices (A and B)
 * while freezing the base layer's parameters.
 * @author bhangun
 */
public class LoRALayer extends NNModule {

    private final Linear baseLayer;
    private final int rank;
    private final double alpha;
    private final double scaling;

    public LoRALayer(Linear baseLayer, int rank, double alpha) {
        this.baseLayer = baseLayer;
        this.rank = rank;
        this.alpha = alpha;
        this.scaling = alpha / rank;

        long inFeatures = baseLayer.inFeatures();
        long outFeatures = baseLayer.outFeatures();

        // Initialize LoRA A with random normal (scaled) and LoRA B with zeros
        Tensor loraA = Tensor.randn(inFeatures, rank).mul((float) Math.sqrt(2.0 / inFeatures));
        Tensor loraB = Tensor.zeros(rank, outFeatures);

        registerParameter("lora_a", loraA);
        registerParameter("lora_b", loraB);

        // Freeze base layer parameters
        for (Tensor param : baseLayer.getParameters()) {
            param.setRequiresGrad(false);
        }

        // Register the base layer so its frozen parameters are still tracked (if
        // needed)
        registerModule("base", baseLayer);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor baseOutput = baseLayer.forward(input);

        Tensor loraA = parameters.get("lora_a");
        Tensor loraB = parameters.get("lora_b");

        // loraOutput = input * A * B * scaling
        Tensor loraOutput = input.matmul(loraA).matmul(loraB).mul((float) scaling);

        return baseOutput.add(loraOutput);
    }

    public Linear getBaseLayer() {
        return baseLayer;
    }

    public int getRank() {
        return rank;
    }

    public double getAlpha() {
        return alpha;
    }

    public double getScaling() {
        return scaling;
    }

    @Override
    public String toString() {
        return String.format("LoRALayer(base=%s, rank=%d, alpha=%.1f)", baseLayer, rank, alpha);
    }
}
