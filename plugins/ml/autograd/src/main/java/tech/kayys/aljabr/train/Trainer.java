package tech.kayys.aljabr.train;

import tech.kayys.aljabr.nn.Module;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.data.Batch;
import tech.kayys.aljabr.data.DataLoader;
import tech.kayys.aljabr.ir.GGraph;
import tech.kayys.aljabr.ir.GValueId;

import java.util.*;
import java.util.function.Consumer;

/**
 * @deprecated Legacy training entrypoint retained for compatibility.
 *             Prefer {@link tech.kayys.aljabr.trainer.Trainers} and the
 *             richer {@link tech.kayys.aljabr.ml.train.Trainer} runtime.
 * @author bhangun
 */
@Deprecated(since = "0.1.0", forRemoval = false)
public final class Trainer {
    private final Module model;
    private final Optimizer optimizer;
    private final LossFunction loss;
    private final Device device;
    private final Metrics metrics;
    private final List<Consumer<Integer>> callbacks = new ArrayList<>();

    private int currentEpoch = 0;
    private boolean training = true;

    public Trainer(Module model, Optimizer optimizer, LossFunction loss, Device device) {
        this.model = model;
        this.optimizer = optimizer;
        this.loss = loss;
        this.device = device;
        this.metrics = new Metrics();

        // Move model to device
        model.to(device);
    }

    public void train(DataLoader trainLoader, DataLoader valLoader, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            currentEpoch = epoch;

            // Training phase
            model.zeroGrad();
            float trainLoss = trainEpoch(trainLoader);

            // Validation phase
            float valLoss = evaluate(valLoader);

            // Log metrics
            metrics.log(epoch, trainLoss, valLoss);

            // Call callbacks
            for (var cb : callbacks) {
                cb.accept(epoch);
            }

            // Early stopping
            if (metrics.shouldStop()) {
                System.out.println("Early stopping at epoch " + epoch);
                break;
            }
        }
    }

    private float trainEpoch(DataLoader loader) {
        float totalLoss = 0;
        int batches = 0;

        for (Batch batch : loader.batches()) {
            if (!training)
                break;

            // Move data to device
            Tensor input = batch.tokens.to(device);
            Tensor target = batch.targets.to(device);

            // Forward pass
            Tensor output = model.forward(input);
            Tensor lossTensor = this.loss.forward(output, target);

            // Backward pass
            model.zeroGrad();
            lossTensor.backward();

            // Optimizer step
            optimizer.step(model.parameters());

            totalLoss += lossTensor.item();
            batches++;

            // Periodic logging
            if (batches % 100 == 0) {
                System.out.printf("  Batch %d, Loss: %.4f%n", batches, lossTensor.item());
            }
        }

        return totalLoss / batches;
    }

    private float evaluate(DataLoader loader) {
        model.zeroGrad();
        float totalLoss = 0;
        int batches = 0;

        // Disable gradients for evaluation
        boolean oldTraining = training;
        training = false;

        for (Batch batch : loader.batches()) {
            Tensor input = batch.tokens.to(device);
            Tensor target = batch.targets.to(device);

            Tensor output = model.forward(input);
            Tensor lossTensor = this.loss.forward(output, target);

            totalLoss += lossTensor.item();
            batches++;
        }

        training = oldTraining;
        return totalLoss / batches;
    }

    public void trainStep(GGraph model, GValueId loss, Map<String, Tensor> params, Map<String, Tensor> inputs) {
        // Placeholder for IR-based training step
        // In a real implementation, this would:
        // 1. Run forward pass on GGraph
        // 2. Use AutogradEngine to get gradients for params
        // 3. Update params using optimizer
    }

    public void addCallback(Consumer<Integer> callback) {
        callbacks.add(callback);
    }

    public void stop() {
        training = false;
    }

    public Metrics metrics() {
        return metrics;
    }

    // Loss functions
    public enum LossFunction {
        CROSS_ENTROPY,
        MSE,
        MAE,
        BCE;

        public Tensor forward(Tensor pred, Tensor target) {
            switch (this) {
                case CROSS_ENTROPY:
                    return pred.crossEntropy(target);
                case MSE:
                    return pred.sub(target).pow(2).mean();
                case MAE:
                    return pred.sub(target).abs().mean();
                case BCE:
                    return pred.binaryCrossEntropy(target);
                default:
                    throw new UnsupportedOperationException();
            }
        }
    }

    // Metrics tracking
    public static class Metrics {
        private final List<Float> trainLosses = new ArrayList<>();
        private final List<Float> valLosses = new ArrayList<>();
        private int patience = 5;
        private float bestLoss = Float.MAX_VALUE;
        private int noImprovement = 0;

        public void log(int epoch, float trainLoss, float valLoss) {
            trainLosses.add(trainLoss);
            valLosses.add(valLoss);

            System.out.printf("Epoch %d - Train Loss: %.4f, Val Loss: %.4f%n",
                    epoch, trainLoss, valLoss);

            if (valLoss < bestLoss - 1e-4) {
                bestLoss = valLoss;
                noImprovement = 0;
            } else {
                noImprovement++;
            }
        }

        public boolean shouldStop() {
            return noImprovement >= patience;
        }

        public float bestLoss() {
            return bestLoss;
        }

        public List<Float> trainLosses() {
            return trainLosses;
        }

        public List<Float> valLosses() {
            return valLosses;
        }
    }
}
