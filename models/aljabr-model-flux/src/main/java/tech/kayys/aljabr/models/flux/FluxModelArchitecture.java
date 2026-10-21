package tech.kayys.aljabr.models.flux;

/**
 * Model architecture mapping and weight key conventions for FLUX MM-DiT and Autoencoders.
 */
public final class FluxModelArchitecture {

    private FluxModelArchitecture() {
    }

    // --- Dual Text Encoders ---
    public static final String CLIP_TEXT_EMBED = "text_encoder.text_model.embeddings.token_embedding.weight";
    public static final String CLIP_FINAL_LAYER_NORM = "text_encoder.text_model.final_layer_norm.weight";
    public static final String T5_SHARED_EMBED = "text_encoder_2.shared.weight";
    public static final String T5_FINAL_LAYER_NORM = "text_encoder_2.encoder.final_layer_norm.weight";

    // --- Flux DiT Transformer ---
    public static final String IMG_IN_PROJ = "transformer.img_in.weight";
    public static final String TXT_IN_PROJ = "transformer.txt_in.weight";
    public static final String TIME_IN_PROJ = "transformer.time_in.in_layer.weight";
    public static final String GUIDANCE_IN_PROJ = "transformer.guidance_in.in_layer.weight";
    public static final String VECTOR_IN_PROJ = "transformer.vector_in.in_layer.weight";
    public static final String FINAL_LAYER_LINEAR = "transformer.final_layer.linear.weight";
    public static final String FINAL_LAYER_ADA_LN = "transformer.final_layer.adaLN_modulation.1.weight";

    // --- Double Stream MM-DiT Blocks ---
    public static String doubleStreamImgQkv(int blockIdx) {
        return "transformer.double_blocks.%d.img_attn.qkv.weight".formatted(blockIdx);
    }

    public static String doubleStreamTxtQkv(int blockIdx) {
        return "transformer.double_blocks.%d.txt_attn.qkv.weight".formatted(blockIdx);
    }

    public static String doubleStreamImgProj(int blockIdx) {
        return "transformer.double_blocks.%d.img_attn.proj.weight".formatted(blockIdx);
    }

    public static String doubleStreamTxtProj(int blockIdx) {
        return "transformer.double_blocks.%d.txt_attn.proj.weight".formatted(blockIdx);
    }

    public static String doubleStreamImgMlp(int blockIdx) {
        return "transformer.double_blocks.%d.img_mlp.0.weight".formatted(blockIdx);
    }

    public static String doubleStreamTxtMlp(int blockIdx) {
        return "transformer.double_blocks.%d.txt_mlp.0.weight".formatted(blockIdx);
    }

    // --- Single Stream Blocks ---
    public static String singleStreamLinear1(int blockIdx) {
        return "transformer.single_blocks.%d.linear1.weight".formatted(blockIdx);
    }

    public static String singleStreamLinear2(int blockIdx) {
        return "transformer.single_blocks.%d.linear2.weight".formatted(blockIdx);
    }

    public static String singleStreamModulation(int blockIdx) {
        return "transformer.single_blocks.%d.modulation.lin.weight".formatted(blockIdx);
    }

    // --- VAE Autoencoder ---
    public static final String VAE_POST_QUANT_CONV = "vae.post_quant_conv.weight";
    public static final String VAE_DECODER_CONV_IN = "vae.decoder.conv_in.weight";
    public static final String VAE_DECODER_CONV_OUT = "vae.decoder.conv_out.weight";

    // --- Hyperparameters / Constants ---
    public static final int LATENT_CHANNELS = 16;
    public static final float VAE_SCALE_FACTOR = 0.3611f;
    public static final float VAE_SHIFT_FACTOR = 0.1159f;
    public static final int CLIP_SEQUENCE_LENGTH = 77;
    public static final int T5_SEQUENCE_LENGTH = 512;
}
