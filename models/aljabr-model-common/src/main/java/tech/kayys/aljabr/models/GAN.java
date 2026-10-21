package tech.kayys.aljabr.models;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.*;
import tech.kayys.aljabr.core.nn.*;
import tech.kayys.aljabr.core.nn.*;
import tech.kayys.aljabr.core.nn.*;

/**
 * Generative Adversarial Network (GAN) — trains a generator and discriminator
 * in a minimax game to produce realistic samples.
 *
 * <p>Based on <em>"Generative Adversarial Nets"</em> (Goodfellow et al., 2014).
 *
 * <p>Training objective:
 * <pre>
 *   D loss = -[log D(x) + log(1 - D(G(z)))]   (maximize)
 *   G loss = -log D(G(z))                       (non-saturating)
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var gan = new GAN(latentDim=100, hiddenDim=256, outputDim=784);
 * Tensor fake = gan.generate(batchSize=32);
 * Tensor dLoss = gan.discriminatorLoss(real, fake.detach());
 * Tensor gLoss = gan.generatorLoss(fake);
 * }</pre>
 */
public final class GAN extends NNModule {

    private final int latentDim;
    private final NNModule generator;
    private final NNModule discriminator;

    /**
     * Creates a GAN with MLP generator and discriminator.
     *
     * @param latentDim  noise vector dimension
     * @param hiddenDim  hidden layer size for both networks
     * @param outputDim  generator output / discriminator input dimension
     */
    public GAN(int latentDim, int hiddenDim, int outputDim) {
        this.latentDim     = latentDim;
        this.generator     = register("generator",     buildGenerator(latentDim, hiddenDim, outputDim));
        this.discriminator = register("discriminator", buildDiscriminator(outputDim, hiddenDim));
    }

    /**
     * Generates fake samples from random noise.
     *
     * @param batchSize number of samples to generate
     * @return generated samples {@code [batchSize, outputDim]}
     */
    public Tensor generate(int batchSize) {
        Tensor z = Tensor.randn(batchSize, latentDim);
        return generator.forward(z);
    }

    /**
     * Computes the discriminator loss (binary cross-entropy on real=1, fake=0).
     *
     * @param real real samples {@code [N, outputDim]}
     * @param fake fake samples from generator (detached) {@code [N, outputDim]}
     * @return scalar discriminator loss
     */
    public Tensor discriminatorLoss(Tensor real, Tensor fake) {
        Tensor dReal = discriminator.forward(real);
        Tensor dFake = discriminator.forward(fake);
        // -[log(D(x)) + log(1 - D(G(z)))]
        float loss = 0f;
        float[] dr = dReal.toFloatArray(), df = dFake.toFloatArray();
        for (int i = 0; i < dr.length; i++) {
            float r = Math.max(1e-7f, Math.min(1-1e-7f, dr[i]));
            float f = Math.max(1e-7f, Math.min(1-1e-7f, df[i]));
            loss -= (float)(Math.log(r) + Math.log(1 - f));
        }
        return Tensor.of(new float[]{(loss / dr.length)}, 1);
    }

    /**
     * Computes the generator loss (non-saturating: -log D(G(z))).
     *
     * @param fake fake samples from generator {@code [N, outputDim]}
     * @return scalar generator loss
     */
    public Tensor generatorLoss(Tensor fake) {
        Tensor dFake = discriminator.forward(fake);
        float loss = 0f;
        float[] df = dFake.toFloatArray();
        for (int i = 0; i < df.length; i++) {
            float f = Math.max(1e-7f, Math.min(1-1e-7f, df[i]));
            loss -= (float) Math.log(f);
        }
        return Tensor.of(new float[]{(loss / df.length)}, 1);
    }

    @Override
    public Tensor forward(Tensor z) { return generator.forward(z); }

    /** @return the generator module */
    public NNModule getGenerator()     { return generator; }

    /** @return the discriminator module */
    public NNModule getDiscriminator() { return discriminator; }

    // ── Network builders ──────────────────────────────────────────────────

    private static NNModule buildGenerator(int latentDim, int hiddenDim, int outputDim) {
        return new Sequential(
            new Linear(latentDim, hiddenDim), new ReLU(),
            new Linear(hiddenDim, hiddenDim), new ReLU(),
            new Linear(hiddenDim, outputDim)  // tanh applied externally
        );
    }

    private static NNModule buildDiscriminator(int inputDim, int hiddenDim) {
        return new Sequential(
            new Linear(inputDim,  hiddenDim), new LeakyReLU(0.2f),
            new Linear(hiddenDim, hiddenDim), new LeakyReLU(0.2f),
            new Linear(hiddenDim, 1)          // sigmoid applied in loss
        );
    }
}
