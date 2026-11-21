package tech.kayys.aljabr.diffusion.core;

/** Raised when no installed family provider supports a requested model descriptor. */
public final class UnsupportedDiffusionFamilyException extends IllegalArgumentException {

    public UnsupportedDiffusionFamilyException(String message) {
        super(message);
    }
}
