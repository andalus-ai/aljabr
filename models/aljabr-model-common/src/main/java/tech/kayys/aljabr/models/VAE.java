package tech.kayys.aljabr.models;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.nn.*;
import tech.kayys.aljabr.core.nn.*;
import tech.kayys.aljabr.core.nn.*;
import tech.kayys.aljabr.core.nn.*;

/**
 * Variational Autoencoder (VAE) — generative model that learns a latent
 * distribution over the input data.
 *
 * <p>Based on <em>"Auto-Encoding Variational Bayes"</em> (Kingma & Welling, 2013).
 *
 * <p>Architecture:
 * <pre>
 *   Encoder: x → μ, log σ²
 *   Reparameterize: z = μ + ε·σ  (ε ~ N(0,1))
 *   Decoder: z → x̂
 *   Loss: reconstruction + β·KL(q(z|x) || p(z))
 * </pre>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var vae = new VAE(inputDim=784, latentDim=32, hiddenDim=256);
 * VAE.Output out = vae.forward(x);
 * Tensor loss = vae.loss(x, out);
 * }</pre>
 */
public final class VAE extends NNModule {

    private final Linear encHidden, encMu, encLogVar;
    private final Linear decHidden, decOut;
    private final float beta;

    /**
     * Default constructor for placeholder usage.
     */
    public VAE() {
        this(784, 32, 256, 1.0f);
    }

    /**
     * Creates a VAE with a single hidden layer encoder/decoder.
     *
     * @param inputDim  input/output dimension
     * @param latentDim latent space dimension
     * @param hiddenDim hidden layer dimension
     */
    public VAE(int inputDim, int latentDim, int hiddenDim) {
        this(inputDim, latentDim, hiddenDim, 1.0f);
    }

    /**
     * Creates a β-VAE with controllable KL weight.
     *
     * @param inputDim  input/output dimension
     * @param latentDim latent space dimension
     * @param hiddenDim hidden layer dimension
     * @param beta      KL divergence weight (β=1 → standard VAE, β>1 → disentangled)
     */
    public VAE(int inputDim, int latentDim, int hiddenDim, float beta) {
        this.beta      = beta;
        this.encHidden = register("enc_h",      new Linear(inputDim,  hiddenDim));
        this.encMu     = register("enc_mu",     new Linear(hiddenDim, latentDim));
        this.encLogVar = register("enc_logvar", new Linear(hiddenDim, latentDim));
        this.decHidden = register("dec_h",      new Linear(latentDim, hiddenDim));
        this.decOut    = register("dec_out",    new Linear(hiddenDim, inputDim));
    }

    /**
     * VAE forward pass output.
     *
     * @param recon  reconstructed input {@code [N, inputDim]}
     * @param mu     latent mean {@code [N, latentDim]}
     * @param logVar latent log-variance {@code [N, latentDim]}
     * @param z      sampled latent vector {@code [N, latentDim]}
     */
    public record Output(Tensor recon, Tensor mu, Tensor logVar, Tensor z) {}

    /**
     * Forward pass: encode → reparameterize → decode.
     *
     * @param x input tensor {@code [N, inputDim]}
     * @return {@link Output} with reconstruction, μ, log σ², and z
     */
    @Override
    public Tensor forward(Tensor x) {
        return forwardFull(x).recon();
    }

    /**
     * Decodes a latent vector z back into the original data space.
     *
     * @param z latent vector [N, latentDim]
     * @return  reconstructed data [N, inputDim]
     */
    public Tensor decode(Tensor z) {
        return decOut.forward(decHidden.forward(z).relu()).sigmoid();
    }

    /**
     * Full forward pass returning all VAE outputs.
     *
     * @param x input tensor {@code [N, inputDim]}
     * @return {@link Output}
     */
    public Output forwardFull(Tensor x) {
        // Encode
        Tensor h      = encHidden.forward(x).relu();
        Tensor mu     = encMu.forward(h);
        Tensor logVar = encLogVar.forward(h);

        // Reparameterize: z = μ + ε·exp(0.5·logVar)
        Tensor std = logVar.mul(0.5f).exp();
        Tensor eps = Tensor.randn(mu.shape().dims());
        Tensor z   = mu.add(eps.mul(std));

        // Decode
        Tensor recon = decOut.forward(decHidden.forward(z).relu()).sigmoid();
        return new Output(recon, mu, logVar, z);
    }

    /**
     * Computes the ELBO loss: reconstruction (BCE) + β·KL divergence.
     *
     * <p>KL divergence for Gaussian: {@code -0.5 · Σ(1 + logVar - μ² - exp(logVar))}
     *
     * @param x   original input
     * @param out VAE forward output
     * @return scalar ELBO loss
     */
    public Tensor loss(Tensor x, Output out) {
        // Reconstruction loss: binary cross-entropy
        float[] xd = x.toFloatArray(), rd = out.recon().toFloatArray();
        float bce = 0f;
        for (int i = 0; i < xd.length; i++) {
            float r = Math.max(1e-7f, Math.min(1 - 1e-7f, rd[i]));
            bce -= xd[i] * (float) Math.log(r) + (1 - xd[i]) * (float) Math.log(1 - r);
        }
        bce /= xd.length;

        // KL divergence: -0.5 * mean(1 + logVar - mu^2 - exp(logVar))
        float[] mu = out.mu().toFloatArray(), lv = out.logVar().toFloatArray();
        float kl = 0f;
        for (int i = 0; i < mu.length; i++)
            kl += 1f + lv[i] - mu[i] * mu[i] - (float) Math.exp(lv[i]);
        kl = -0.5f * kl / mu.length;

        return Tensor.of(new float[]{(bce + beta * kl)}, 1);
    }
}
