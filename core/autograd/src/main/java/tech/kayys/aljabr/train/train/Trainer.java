package tech.kayys.aljabr.ml.train;

import tech.kayys.aljabr.ml.autograd.GradTensor;
import tech.kayys.aljabr.ml.optim.GradScaler;
import tech.kayys.aljabr.ml.optim.Optimizer;
import tech.kayys.aljabr.ml.nn.Parameter;
import tech.kayys.aljabr.train.data.DataLoader.Batch;
import tech.kayys.aljabr.trainer.api.TrainerConfig;
import tech.kayys.aljabr.trainer.api.TrainerSession;
import tech.kayys.aljabr.trainer.api.TrainingListener;
import tech.kayys.aljabr.trainer.api.TrainingSummary;

import java.io.Closeable;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * Complete training pipeline with callbacks, schedulers, and monitoring.
 *
 * <p>
 * Trainer that handles the training loop, validation,
 * checkpointing, early stopping, and learning rate scheduling.
 * </p>
 *
 * <h3>Example Usage</h3>
 * 
 * <pre>{@code
 * Trainer trainer = Trainer.builder()
 *         .model(inputs -> model.forward(inputs))
 *         .optimizer(optimizer)
 *         .loss((preds, targets) -> preds.subtract(targets).pow(2).mean())
 *         .callbacks(List.of(
 *                 EarlyStopping.patience(5),
 *                 ModelCheckpoint.at("best.pt"),
 *                 ConsoleLogger.create()))
 *         .epochs(100)
 *         .gradientClip(1.0)
 *         .build();
 *
 * trainer.fit(trainLoader, valLoader);
 * }</pre>
 *
 * @author Aljabr Team
 * @version 0.1.0
 */
public class Trainer implements Closeable, TrainerSession {

    private final Model model;
    private final Optimizer optimizer;
    private final LossFunction lossFn;
    private final TrainerConfig config;
    private final List<TrainingListener> listeners;
    private final List<LRScheduler> schedulers;

    private final TrainingMetrics metrics;
    private final AtomicBoolean stopped = new AtomicBoolean(false);
    private final GradScaler gradScaler;

    private int currentEpoch = 0;
    private int globalStep = 0;

    /**
     * Model interface for training.
     */
    @FunctionalInterface
    public interface Model {
        GradTensor forward(GradTensor inputs);

        /** Set the model to training mode. */
        default void train() {
        }

        /** Set the model to evaluation mode. */
        default void eval() {
        }
    }

    /**
     * Loss function interface.
     */
    @FunctionalInterface
    public interface LossFunction {
        GradTensor compute(GradTensor predictions, GradTensor targets);
    }

    /**
     * DataLoader type alias — any Iterable of Batch.
     */
    public interface DataLoader extends Iterable<Batch> {
    }

    private Trainer(Builder builder) {
        this.model = Objects.requireNonNull(builder.model, "model must not be null");
        this.optimizer = Objects.requireNonNull(builder.optimizer, "optimizer must not be null");
        this.lossFn = Objects.requireNonNull(builder.lossFn, "loss function must not be null");
        this.config = builder.buildConfig();
        this.listeners = new ArrayList<>(builder.listeners);
        this.schedulers = new ArrayList<>(builder.schedulers);
        this.metrics = new TrainingMetrics();

        // Initialize GradScaler for mixed precision training
        if (config.mixedPrecision()) {
            this.gradScaler = GradScaler.builder()
                    .initScale(65536.0)
                    .growthInterval(2000)
                    .build();
        } else {
            this.gradScaler = null;
        }
    }

    /**
     * Create a new builder for constructing Trainer instances.
     *
     * @return new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Run the complete training loop.
     *
     * @param trainLoader training data loader
     * @param valLoader   validation data loader (may be null)
     */
    public void fit(Iterable<Batch> trainLoader, Iterable<Batch> valLoader) {
        onTrainingStart();

        try {
            for (int epoch = 0; epoch < config.epochs() && !stopped.get(); epoch++) {
                currentEpoch = epoch;
                onEpochStart(epoch);

                // Training phase
                double trainLoss = trainOneEpoch(trainLoader);
                metrics.updateTrainLoss(epoch, trainLoss);

                onEpochEnd(epoch, trainLoss);

                // Validation phase
                if (valLoader != null) {
                    double valLoss = validate(valLoader);
                    metrics.updateValLoss(epoch, valLoss);
                    onValidationEnd(epoch, valLoss);
                }

                // Scheduler step
                for (LRScheduler scheduler : schedulers) {
                    scheduler.step(metrics);
                }

                // Check early stopping
                if (shouldStopEarly()) {
                    onEarlyStopping(epoch);
                    break;
                }
            }
        } catch (Exception e) {
            onTrainingError(e);
            throw new TrainingException("Training failed at epoch " + currentEpoch, e);
        } finally {
            metrics.markEnded();
            onTrainingEnd();
        }
    }

    /**
     * Run training without validation.
     *
     * @param trainLoader training data loader
     */
    public void fit(Iterable<Batch> trainLoader) {
        fit(trainLoader, null);
    }

    /**
     * Run a single training epoch.
     *
     * @param loader data loader
     * @return average loss for the epoch
     */
    private double trainOneEpoch(Iterable<Batch> loader) {
        model.train(); // set training mode
        double totalLoss = 0.0;
        int batchCount = 0;

        for (Batch batch : loader) {
            if (stopped.get())
                break;

            onBatchStart(globalStep);

            GradTensor inputs = batch.inputs();
            GradTensor targets = batch.labels();

            // Forward pass
            GradTensor predictions = model.forward(inputs);
            GradTensor loss = lossFn.compute(predictions, targets);

            // Mixed precision: scale loss before backward
            if (gradScaler != null) {
                gradScaler.scale(loss);
            }

            // Backward pass
            loss.backward();

            // Mixed precision: unscale and check for overflow
            if (gradScaler != null) {
                boolean overflow = gradScaler.unscaleAndCheck(
                        optimizer.parameters().stream().map(Parameter::data).toList());
                if (!overflow && config.gradientClip() > 0) {
                    clipGradients(config.gradientClip());
                }
            } else if (config.gradientClip() > 0) {
                // Standard gradient clipping
                clipGradients(config.gradientClip());
            }

            // Optimizer step
            if (gradScaler != null) {
                gradScaler.step(optimizer);
                gradScaler.update();
            } else {
                optimizer.step();
            }
            optimizer.zeroGrad();

            double lossValue = loss.item();
            // Unscale loss value for logging if using mixed precision
            if (gradScaler != null) {
                lossValue /= gradScaler.getScale();
            }
            totalLoss += lossValue;
            batchCount++;

            metrics.updateBatchLoss(globalStep, lossValue);
            onBatchEnd(globalStep, lossValue);

            globalStep++;
        }

        return batchCount > 0 ? totalLoss / batchCount : 0.0;
    }

    /**
     * Run validation.
     *
     * @param loader validation data loader
     * @return average validation loss
     */
    private double validate(Iterable<Batch> loader) {
        model.eval(); // set evaluation mode
        double totalLoss = 0.0;
        int batchCount = 0;

        for (Batch batch : loader) {
            GradTensor inputs = batch.inputs();
            GradTensor targets = batch.labels();

            GradTensor predictions = model.forward(inputs);
            GradTensor loss = lossFn.compute(predictions, targets);

            totalLoss += loss.item();
            batchCount++;
        }

        return batchCount > 0 ? totalLoss / batchCount : 0.0;
    }

    /**
     * Clip gradients to max norm.
     *
     * @param maxNorm maximum gradient norm
     */
    private void clipGradients(double maxNorm) {
        double totalNorm = 0.0;

        // First pass: compute total norm
        for (Parameter param : optimizer.parameters()) {
            GradTensor data = param.data();
            if (data.grad() != null) {
                float[] grad = data.grad().data();
                for (float g : grad) {
                    totalNorm += g * g;
                }
            }
        }
        totalNorm = Math.sqrt(totalNorm);

        // Second pass: clip if needed
        if (totalNorm > maxNorm) {
            double scale = maxNorm / (totalNorm + 1e-6);
            for (Parameter param : optimizer.parameters()) {
                GradTensor data = param.data();
                if (data.grad() != null) {
                    float[] grad = data.grad().data();
                    for (int i = 0; i < grad.length; i++) {
                        grad[i] *= (float) scale;
                    }
                }
            }
        }
    }

    /**
     * Check if early stopping criteria is met.
     *
     * @return true if should stop early
     */
    private boolean shouldStopEarly() {
        return listeners.stream()
                .filter(EarlyStopping.class::isInstance)
                .map(EarlyStopping.class::cast)
                .anyMatch(EarlyStopping::shouldStop);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Callback Hooks
    // ─────────────────────────────────────────────────────────────────────────

    private void onTrainingStart() {
        listeners.forEach(listener -> listener.onTrainingStart(this));
    }

    private void onEpochStart(int epoch) {
        listeners.forEach(listener -> listener.onEpochStart(this, epoch));
    }

    private void onEpochEnd(int epoch, double trainLoss) {
        listeners.forEach(listener -> listener.onEpochEnd(this, epoch, trainLoss));
    }

    private void onValidationEnd(int epoch, double valLoss) {
        listeners.forEach(listener -> listener.onValidationEnd(this, epoch, valLoss));
    }

    private void onBatchStart(int step) {
        listeners.forEach(listener -> listener.onBatchStart(this, step));
    }

    private void onBatchEnd(int step, double loss) {
        listeners.forEach(listener -> listener.onBatchEnd(this, step, loss));
    }

    private void onEarlyStopping(int epoch) {
        listeners.forEach(listener -> listener.onEarlyStopping(this, epoch));
    }

    private void onTrainingError(Exception e) {
        listeners.forEach(listener -> listener.onTrainingError(this, e));
    }

    private void onTrainingEnd() {
        TrainingSummary summary = summary();
        listeners.forEach(listener -> listener.onTrainingEnd(this, summary));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Stop training gracefully.
     */
    public void stop() {
        stopped.set(true);
    }

    @Override
    public boolean isStopped() {
        return stopped.get();
    }

    /**
     * Get current epoch number.
     *
     * @return current epoch
     */
    public int getCurrentEpoch() {
        return currentEpoch;
    }

    @Override
    public int currentEpoch() {
        return getCurrentEpoch();
    }

    /**
     * Get global step number.
     *
     * @return global step
     */
    public int getGlobalStep() {
        return globalStep;
    }

    @Override
    public int globalStep() {
        return getGlobalStep();
    }

    /**
     * Get training metrics.
     *
     * @return metrics instance
     */
    public TrainingMetrics getMetrics() {
        return metrics;
    }

    /**
     * Get the model being trained.
     *
     * @return model
     */
    public Model getModel() {
        return model;
    }

    /**
     * Get the optimizer.
     *
     * @return optimizer
     */
    public Optimizer getOptimizer() {
        return optimizer;
    }

    /**
     * Get training configuration.
     *
     * @return config
     */
    public TrainerConfig getConfig() {
        return config;
    }

    @Override
    public TrainerConfig config() {
        return config;
    }

    @Override
    public TrainingSummary summary() {
        return metrics.toTrainingSummary();
    }

    /**
     * Get the gradient scaler (for mixed precision training).
     *
     * @return GradScaler instance, or null if mixed precision is disabled
     */
    public GradScaler getGradScaler() {
        return gradScaler;
    }

    @Override
    public void close() {
        stopped.set(true);
        listeners.forEach(TrainingListener::close);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Builder
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Builder for constructing Trainer instances.
     */
    public static class Builder {
        private Model model;
        private Optimizer optimizer;
        private LossFunction lossFn;
        private int epochs = 100;
        private double gradientClip = 0.0;
        private boolean mixedPrecision = false;
        private Path checkpointDir = null;
        private final List<TrainingListener> listeners = new ArrayList<>();
        private final List<LRScheduler> schedulers = new ArrayList<>();

        private Builder() {
        }

        /**
         * Set the model to train.
         *
         * @param model model interface
         * @return this builder
         */
        public Builder model(Model model) {
            this.model = model;
            return this;
        }

        /**
         * Set the optimizer.
         *
         * @param optimizer optimizer instance
         * @return this builder
         */
        public Builder optimizer(Optimizer optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        /**
         * Set the loss function.
         *
         * @param lossFn loss function
         * @return this builder
         */
        public Builder loss(LossFunction lossFn) {
            this.lossFn = lossFn;
            return this;
        }

        /**
         * Set number of training epochs.
         *
         * @param epochs number of epochs
         * @return this builder
         */
        public Builder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        /**
         * Set gradient clipping norm (0.0 = disabled).
         *
         * @param maxNorm maximum gradient norm
         * @return this builder
         */
        public Builder gradientClip(double maxNorm) {
            this.gradientClip = maxNorm;
            return this;
        }

        /**
         * Enable mixed precision training.
         *
         * @param enabled true to enable
         * @return this builder
         */
        public Builder mixedPrecision(boolean enabled) {
            this.mixedPrecision = enabled;
            return this;
        }

        /**
         * Set checkpoint directory.
         *
         * @param dir checkpoint directory path
         * @return this builder
         */
        public Builder checkpointDir(Path dir) {
            this.checkpointDir = dir;
            return this;
        }

        /**
         * Add a callback.
         *
         * @param callback callback instance
         * @return this builder
         */
        public Builder callback(Callback callback) {
            this.listeners.add(callback);
            return this;
        }

        /**
         * Add multiple callbacks.
         *
         * @param callbacks list of callbacks
         * @return this builder
         */
        public Builder callbacks(List<Callback> callbacks) {
            this.listeners.addAll(callbacks);
            return this;
        }

        public Builder listener(TrainingListener listener) {
            this.listeners.add(listener);
            return this;
        }

        public Builder listeners(List<? extends TrainingListener> listeners) {
            this.listeners.addAll(listeners);
            return this;
        }

        /**
         * Add a learning rate scheduler.
         *
         * @param scheduler scheduler instance
         * @return this builder
         */
        public Builder scheduler(LRScheduler scheduler) {
            this.schedulers.add(scheduler);
            return this;
        }

        /**
         * Build the Trainer instance.
         *
         * @return configured trainer
         */
        public Trainer build() {
            return new Trainer(this);
        }

        private TrainerConfig buildConfig() {
            return new TrainerConfig(epochs, gradientClip, mixedPrecision, checkpointDir);
        }
    }

    /**
     * Training exception.
     */
    public static class TrainingException extends RuntimeException {
        public TrainingException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
