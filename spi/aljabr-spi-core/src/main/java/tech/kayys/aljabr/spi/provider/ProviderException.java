package tech.kayys.aljabr.spi.provider;
/** Aljabr-owned failure raised while initializing or closing a provider. */
public class ProviderException extends Exception {
    public ProviderException(String message) { super(message); }
    public ProviderException(String message, Throwable cause) { super(message, cause); }
}
