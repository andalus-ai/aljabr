package tech.kayys.aljabr.ml.train;

import java.io.Closeable;

import tech.kayys.aljabr.trainer.api.TrainerSession;
import tech.kayys.aljabr.trainer.api.TrainingListener;
import tech.kayys.aljabr.trainer.api.TrainingSummary;

/**
 * Callback interface for training events.
 *
 * <p>
 * Implement this interface to hook into various points of the training loop
 * for logging, checkpointing, early stopping, or custom behavior.
 * </p>
 *
 * <h3>Example</h3>
 * 
 * <pre>{@code
 * public class MyCallback implements Callback {
 *     @Override
 *     public void onEpochEnd(Trainer trainer, int epoch, double trainLoss) {
 *         System.out.printf("Epoch %d: loss=%.4f%n", epoch, trainLoss);
 *     }
 * }
 * }</pre>
 *
 * @author Aljabr Team
 * @version 0.1.0
 */
public interface Callback extends Closeable, TrainingListener {

    /**
     * Called when training starts.
     *
     * @param trainer trainer instance
     */
    default void onTrainingStart(Trainer trainer) {
    }

    /**
     * Called at the start of each epoch.
     *
     * @param trainer trainer instance
     * @param epoch   epoch number (0-based)
     */
    default void onEpochStart(Trainer trainer, int epoch) {
    }

    /**
     * Called at the end of each epoch (after training, before validation).
     *
     * @param trainer   trainer instance
     * @param epoch     epoch number (0-based)
     * @param trainLoss average training loss for the epoch
     */
    default void onEpochEnd(Trainer trainer, int epoch, double trainLoss) {
    }

    /**
     * Called after validation completes.
     *
     * @param trainer trainer instance
     * @param epoch   epoch number (0-based)
     * @param valLoss average validation loss for the epoch
     */
    default void onValidationEnd(Trainer trainer, int epoch, double valLoss) {
    }

    /**
     * Called at the start of each batch.
     *
     * @param trainer trainer instance
     * @param step    global step number
     */
    default void onBatchStart(Trainer trainer, int step) {
    }

    /**
     * Called at the end of each batch.
     *
     * @param trainer trainer instance
     * @param step    global step number
     * @param loss    loss value for this batch
     */
    default void onBatchEnd(Trainer trainer, int step, double loss) {
    }

    /**
     * Called when early stopping is triggered.
     *
     * @param trainer trainer instance
     * @param epoch   epoch number when stopped
     */
    default void onEarlyStopping(Trainer trainer, int epoch) {
    }

    /**
     * Called when a training error occurs.
     *
     * @param trainer trainer instance
     * @param error   the exception that occurred
     */
    default void onTrainingError(Trainer trainer, Exception error) {
    }

    /**
     * Called when training ends (successfully or via early stopping).
     *
     * @param trainer trainer instance
     */
    default void onTrainingEnd(Trainer trainer) {
    }

    @Override
    default void onTrainingStart(TrainerSession session) {
        if (session instanceof Trainer trainer) {
            onTrainingStart(trainer);
        }
    }

    @Override
    default void onEpochStart(TrainerSession session, int epoch) {
        if (session instanceof Trainer trainer) {
            onEpochStart(trainer, epoch);
        }
    }

    @Override
    default void onEpochEnd(TrainerSession session, int epoch, double trainLoss) {
        if (session instanceof Trainer trainer) {
            onEpochEnd(trainer, epoch, trainLoss);
        }
    }

    @Override
    default void onValidationEnd(TrainerSession session, int epoch, double valLoss) {
        if (session instanceof Trainer trainer) {
            onValidationEnd(trainer, epoch, valLoss);
        }
    }

    @Override
    default void onBatchStart(TrainerSession session, int step) {
        if (session instanceof Trainer trainer) {
            onBatchStart(trainer, step);
        }
    }

    @Override
    default void onBatchEnd(TrainerSession session, int step, double loss) {
        if (session instanceof Trainer trainer) {
            onBatchEnd(trainer, step, loss);
        }
    }

    @Override
    default void onEarlyStopping(TrainerSession session, int epoch) {
        if (session instanceof Trainer trainer) {
            onEarlyStopping(trainer, epoch);
        }
    }

    @Override
    default void onTrainingError(TrainerSession session, Exception error) {
        if (session instanceof Trainer trainer) {
            onTrainingError(trainer, error);
        }
    }

    @Override
    default void onTrainingEnd(TrainerSession session, TrainingSummary summary) {
        if (session instanceof Trainer trainer) {
            onTrainingEnd(trainer);
        }
    }

    @Override
    default void close() {
        // Default: no-op
    }
}
