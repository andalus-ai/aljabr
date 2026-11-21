package tech.kayys.aljabr.models.diffusion;

import java.util.Arrays;

/**
 * Immutable inference-time view over a training noise schedule.
 * Timesteps are ordered from the noisiest training step to the clean sample.
 */
public final class DiffusionSchedule {

    private final float[] alphasCumprod;
    private final int[] timesteps;

    DiffusionSchedule(float[] alphasCumprod, int[] timesteps) {
        this.alphasCumprod = Arrays.copyOf(alphasCumprod, alphasCumprod.length);
        this.timesteps = Arrays.copyOf(timesteps, timesteps.length);
    }

    public int trainingTimesteps() {
        return alphasCumprod.length;
    }

    public int inferenceSteps() {
        return timesteps.length;
    }

    public int timestepAt(int stepIndex) {
        return timesteps[stepIndex];
    }

    public int previousTimestepAt(int stepIndex) {
        return stepIndex + 1 < timesteps.length ? timesteps[stepIndex + 1] : -1;
    }

    public float alphaCumprodAtTrainingTimestep(int timestep) {
        return alphasCumprod[timestep];
    }

    float alphaCumprodAtStep(int stepIndex) {
        return alphasCumprod[timestepAt(stepIndex)];
    }

    float previousAlphaCumprodAtStep(int stepIndex) {
        int previousTimestep = previousTimestepAt(stepIndex);
        return previousTimestep < 0 ? 1.0f : alphasCumprod[previousTimestep];
    }

    public int[] timesteps() {
        return Arrays.copyOf(timesteps, timesteps.length);
    }
}
