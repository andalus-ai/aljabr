package tech.kayys.aljabr.spi.model.ir;

/** Well-known architecture families. Callers may define additional families. */
public final class ArchitectureFamilies {

    public static final ArchitectureFamily DECODER_ONLY_TRANSFORMER =
            new ArchitectureFamily("transformer.decoder-only", ArchitectureKind.DECODER_ONLY_TRANSFORMER);
    public static final ArchitectureFamily ENCODER_ONLY_TRANSFORMER =
            new ArchitectureFamily("transformer.encoder-only", ArchitectureKind.ENCODER_ONLY_TRANSFORMER);
    public static final ArchitectureFamily ENCODER_DECODER_TRANSFORMER =
            new ArchitectureFamily("transformer.encoder-decoder", ArchitectureKind.ENCODER_DECODER_TRANSFORMER);
    public static final ArchitectureFamily VISION_TRANSFORMER =
            new ArchitectureFamily("vision.transformer", ArchitectureKind.VISION);
    public static final ArchitectureFamily CONVOLUTIONAL =
            new ArchitectureFamily("convolutional", ArchitectureKind.CONVOLUTIONAL);
    public static final ArchitectureFamily DIFFUSION =
            new ArchitectureFamily("diffusion", ArchitectureKind.DIFFUSION);
    public static final ArchitectureFamily SPEECH =
            new ArchitectureFamily("speech", ArchitectureKind.SPEECH);
    public static final ArchitectureFamily MULTIMODAL =
            new ArchitectureFamily("multimodal", ArchitectureKind.MULTIMODAL);
    public static final ArchitectureFamily EMBEDDING =
            new ArchitectureFamily("embedding", ArchitectureKind.EMBEDDING);
    public static final ArchitectureFamily RERANKER =
            new ArchitectureFamily("reranker", ArchitectureKind.RERANKER);

    private ArchitectureFamilies() {
    }
}
