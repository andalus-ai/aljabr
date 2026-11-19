package tech.kayys.aljabr.optim;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import tech.kayys.aljabr.core.nn.Parameter;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;
import tech.kayys.aljabr.core.tensor.autograd.AutogradContext;

/**
 * Fixed-learning-rate Adafactor with factored second-moment estimates for
 * tensors of rank two or greater and an unfactored estimate for vectors.
 *
 * <p>At step {@code t}, {@code beta2 = 1 - t^-0.8}. For a matrix-like
 * gradient, row and column means of its squared values are maintained
 * independently and combined as
 * {@code variance = rowMean * columnMean / mean(rowMean)}. Lower-rank
 * gradients use a full element-wise second-moment estimate. The update is
 * {@code parameter -= learningRate * gradient / (sqrt(variance) + epsilon)}.</p>
 *
 * <p>The two-argument constructor retains fixed-rate behavior. The four-argument
 * constructor enables update-RMS clipping and parameter-scale adaptation.
 * Relative-step scheduling and learning-rate warmup are not included.</p>
 */
public final class Adafactor implements Optimizer, AutoCloseable {
    private final float learningRate;
    private final float epsilon;
    private final float updateRmsClipThreshold;
    private final float parameterScaleEpsilon;

    private final Map<String, Tensor> rowSecondMoments = new LinkedHashMap<>();
    private final Map<String, Tensor> columnSecondMoments = new LinkedHashMap<>();
    private final Map<String, Tensor> fullSecondMoments = new LinkedHashMap<>();
    private final Map<String, Tensor> parameterSnapshots = new LinkedHashMap<>();
    private final Map<String, Long> parameterSteps = new LinkedHashMap<>();

    public Adafactor(float learningRate) {
        this(learningRate, 1.0e-8f);
    }

    public Adafactor(float learningRate, float epsilon) {
        this(learningRate, epsilon, 0.0f, 0.0f);
    }

    /**
     * Creates Adafactor with update-RMS clipping and parameter-scale adaptation.
     *
     * @param learningRate fixed base learning rate
     * @param epsilon denominator stability epsilon
     * @param updateRmsClipThreshold maximum RMS of the normalized update; must be positive
     * @param parameterScaleEpsilon lower bound for RMS(parameter); must be positive
     */
    public Adafactor(float learningRate, float epsilon,
            float updateRmsClipThreshold, float parameterScaleEpsilon) {
        if (!Float.isFinite(learningRate) || learningRate < 0.0f) {
            throw new IllegalArgumentException("Learning rate must be finite and non-negative");
        }
        if (!Float.isFinite(epsilon) || epsilon <= 0.0f) {
            throw new IllegalArgumentException("epsilon must be finite and positive");
        }
        if (!Float.isFinite(updateRmsClipThreshold) || updateRmsClipThreshold <= 0.0f) {
            throw new IllegalArgumentException("Update RMS clip threshold must be finite and positive");
        }
        if (!Float.isFinite(parameterScaleEpsilon) || parameterScaleEpsilon <= 0.0f) {
            throw new IllegalArgumentException("Parameter scale epsilon must be finite and positive");
        }
        this.learningRate = learningRate;
        this.epsilon = epsilon;
        this.updateRmsClipThreshold = updateRmsClipThreshold;
        this.parameterScaleEpsilon = parameterScaleEpsilon;
    }

    public float learningRate() { return learningRate; }
    public float epsilon() { return epsilon; }
    public boolean updateRmsClippingEnabled() { return updateRmsClipThreshold > 0.0f; }
    public boolean parameterScaleAdaptationEnabled() { return parameterScaleEpsilon > 0.0f; }

    @Override
    public void step(Map<String, Parameter> parameters, Map<String, Tensor> gradients) {
        Objects.requireNonNull(parameters, "parameters must not be null");
        Objects.requireNonNull(gradients, "gradients must not be null");

        boolean hasGradient = false;
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            String name = Objects.requireNonNull(entry.getKey(), "parameter name must not be null");
            Parameter parameter = Objects.requireNonNull(entry.getValue(), "parameter " + name);
            Tensor gradient = gradients.get(name);
            if (gradient != null && parameter.isTrainable()) {
                parameter.validateGradient(gradient);
                hasGradient = true;
            }
        }
        if (!hasGradient) return;

        try (AutogradContext.Scope ignored = AutogradContext.pauseRecording()) {
            Map<String, Tensor> nextRows = new LinkedHashMap<>();
            Map<String, Tensor> nextColumns = new LinkedHashMap<>();
            Map<String, Tensor> nextFull = new LinkedHashMap<>();
            Map<String, Tensor> updates = new LinkedHashMap<>();
            Map<String, Long> nextSteps = new LinkedHashMap<>();
            try {
                for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
                    String name = entry.getKey();
                    Parameter parameter = entry.getValue();
                    Tensor gradient = gradients.get(name);
                    if (gradient == null || !parameter.isTrainable()) continue;

                    Tensor current = parameter.tensor();
                    boolean replaced = parameterSnapshots.containsKey(name)
                            && parameterSnapshots.get(name) != current;
                    long nextStep = replaced ? 1L : parameterSteps.getOrDefault(name, 0L) + 1L;
                    float beta2 = 1.0f - (float) Math.pow(nextStep, -0.8);

                    Tensor variance = null;
                    try {
                        if (gradient.shape().rank() >= 2) {
                            Tensor squared = null;
                            Tensor rowObservation = null;
                            Tensor columnObservation = null;
                            Tensor rowAverage = null;
                            Tensor columnAverage = null;
                            Tensor meanRowAverage = null;
                            Tensor normalizedRows = null;
                            try {
                                squared = gradient.mul(gradient);
                                rowObservation = squared.mean(-1, true);
                                columnObservation = squared.mean(-2, true);
                                rowAverage = updateAverage(
                                        rowObservation, replaced ? null : rowSecondMoments.get(name), beta2);
                                nextRows.put(name, rowAverage);
                                rowAverage = null;
                                columnAverage = updateAverage(
                                        columnObservation, replaced ? null : columnSecondMoments.get(name), beta2);
                                nextColumns.put(name, columnAverage);
                                columnAverage = null;

                                meanRowAverage = nextRows.get(name).mean(-2, true);
                                normalizedRows = nextRows.get(name).div(meanRowAverage);
                                variance = normalizedRows.mul(nextColumns.get(name));
                            } finally {
                                if (squared != null) squared.release();
                                if (rowObservation != null) rowObservation.release();
                                if (columnObservation != null) columnObservation.release();
                                if (rowAverage != null) rowAverage.release();
                                if (columnAverage != null) columnAverage.release();
                                if (meanRowAverage != null) meanRowAverage.release();
                                if (normalizedRows != null) normalizedRows.release();
                            }
                        } else {
                            Tensor squared = null;
                            Tensor average = null;
                            try {
                                squared = gradient.mul(gradient);
                                average = updateAverage(
                                        squared, replaced ? null : fullSecondMoments.get(name), beta2);
                                nextFull.put(name, average);
                                average = null;
                                variance = nextFull.get(name);
                            } finally {
                                if (squared != null) squared.release();
                                if (average != null) average.release();
                            }
                        }

                        Tensor root = null;
                        Tensor denominator = null;
                        Tensor normalized = null;
                        Tensor clipped = null;
                        Tensor clipScale = null;
                        Tensor parameterScale = null;
                        Tensor scaled = null;
                        Tensor learningRateScaled = null;
                        try {
                            root = variance.sqrt();
                            denominator = root.add(epsilon);
                            normalized = gradient.div(denominator);
                            clipped = normalized;
                            if (updateRmsClippingEnabled()) {
                                clipScale = clippingScale(normalized, updateRmsClipThreshold, epsilon);
                                clipped = normalized.mul(clipScale);
                            }
                            if (parameterScaleAdaptationEnabled()) {
                                parameterScale = parameterScale(current, parameterScaleEpsilon);
                                learningRateScaled = clipped.mul(parameterScale);
                            } else {
                                learningRateScaled = clipped;
                            }
                            scaled = learningRateScaled.mul(learningRate);
                            updates.put(name, current.sub(scaled));
                        } finally {
                            if (root != null) root.release();
                            if (denominator != null) denominator.release();
                            if (normalized != null) normalized.release();
                            if (clipped != null && clipped != normalized) clipped.release();
                            if (clipScale != null) clipScale.release();
                            if (parameterScale != null) parameterScale.release();
                            if (learningRateScaled != null && learningRateScaled != clipped) {
                                learningRateScaled.release();
                            }
                            if (scaled != null) scaled.release();
                        }
                    } finally {
                        if (variance != null && !nextFull.containsValue(variance)) variance.release();
                    }
                    nextSteps.put(name, nextStep);
                }

                // Publish parameter values before committing the corresponding
                // factored/unfactored state. Snapshot identity detects any
                // partially published replacement on the next call.
                updates.forEach((name, value) -> parameters.get(name).updateTensor(value));

                nextRows.forEach((name, value) -> {
                    Tensor old = rowSecondMoments.put(name, value);
                    if (old != null) old.release();
                });
                nextColumns.forEach((name, value) -> {
                    Tensor old = columnSecondMoments.put(name, value);
                    if (old != null) old.release();
                });
                nextFull.forEach((name, value) -> {
                    Tensor old = fullSecondMoments.put(name, value);
                    if (old != null) old.release();
                });
                updates.forEach((name, value) -> {
                    parameterSnapshots.put(name, value);
                    parameterSteps.put(name, nextSteps.get(name));
                });
            } catch (RuntimeException | Error failure) {
                nextRows.forEach((name, value) -> {
                    if (rowSecondMoments.get(name) != value) value.release();
                });
                nextColumns.forEach((name, value) -> {
                    if (columnSecondMoments.get(name) != value) value.release();
                });
                nextFull.forEach((name, value) -> {
                    if (fullSecondMoments.get(name) != value) value.release();
                });
                updates.forEach((name, value) -> {
                    if (parameters.get(name).tensor() != value) value.release();
                });
                throw failure;
            }
        }
    }

    /**
     * Returns min(1, threshold / RMS(update)) as a rank-zero tensor.
     * The min is expressed with tensor arithmetic, without host scalar extraction.
     */
    private static Tensor clippingScale(Tensor update, float threshold, float epsilon) {
        Tensor squared = null;
        Tensor meanSquare = null;
        Tensor rms = null;
        Tensor stabilizedRms = null;
        Tensor inverseRms = null;
        Tensor ratio = null;
        Tensor one = null;
        Tensor sum = null;
        Tensor difference = null;
        Tensor absoluteDifference = null;
        Tensor result = null;
        try {
            squared = update.mul(update);
            meanSquare = squared.mean();
            rms = meanSquare.sqrt();
            stabilizedRms = rms.add(epsilon);
            inverseRms = stabilizedRms.pow(-1.0f);
            ratio = inverseRms.mul(threshold);
            one = TensorFactory.ones(update.backend());
            sum = one.add(ratio);
            difference = one.sub(ratio);
            absoluteDifference = difference.abs();
            result = sum.sub(absoluteDifference).mul(0.5f);
            return result;
        } catch (RuntimeException | Error failure) {
            if (result != null) result.release();
            throw failure;
        } finally {
            if (squared != null) squared.release();
            if (meanSquare != null) meanSquare.release();
            if (rms != null) rms.release();
            if (stabilizedRms != null) stabilizedRms.release();
            if (inverseRms != null) inverseRms.release();
            if (ratio != null) ratio.release();
            if (one != null) one.release();
            if (sum != null) sum.release();
            if (difference != null) difference.release();
            if (absoluteDifference != null) absoluteDifference.release();
        }
    }

    /** Returns max(RMS(parameter), floor) as a rank-zero tensor. */
    private static Tensor parameterScale(Tensor parameter, float floor) {
        Tensor squared = null;
        Tensor meanSquare = null;
        Tensor rms = null;
        Tensor floorAdjusted = null;
        Tensor difference = null;
        Tensor absoluteDifference = null;
        Tensor sum = null;
        Tensor result = null;
        try {
            squared = parameter.mul(parameter);
            meanSquare = squared.mean();
            rms = meanSquare.sqrt();
            floorAdjusted = rms.add(floor);
            difference = rms.add(-floor);
            absoluteDifference = difference.abs();
            sum = floorAdjusted.add(absoluteDifference);
            result = sum.mul(0.5f);
            return result;
        } catch (RuntimeException | Error failure) {
            if (result != null) result.release();
            throw failure;
        } finally {
            if (squared != null) squared.release();
            if (meanSquare != null) meanSquare.release();
            if (rms != null) rms.release();
            if (floorAdjusted != null) floorAdjusted.release();
            if (difference != null) difference.release();
            if (absoluteDifference != null) absoluteDifference.release();
            if (sum != null) sum.release();
        }
    }

    private static Tensor updateAverage(Tensor observation, Tensor previous, float beta2) {
        Tensor base = previous == null ? observation.zerosLike() : previous.mul(beta2);
        Tensor contribution = null;
        try {
            contribution = observation.mul(1.0f - beta2);
            return base.add(contribution);
        } finally {
            base.release();
            if (contribution != null) contribution.release();
        }
    }

    @Override
    public void reset() {
        rowSecondMoments.values().forEach(Tensor::release);
        columnSecondMoments.values().forEach(Tensor::release);
        fullSecondMoments.values().forEach(Tensor::release);
        rowSecondMoments.clear();
        columnSecondMoments.clear();
        fullSecondMoments.clear();
        parameterSnapshots.clear();
        parameterSteps.clear();
    }

    @Override
    public void close() {
        reset();
    }
}
