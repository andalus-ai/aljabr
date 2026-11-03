package tech.kayys.aljabr.spi.provider;
/**
 * Base contract for built-in and independently published providers.
 * Discoverable through ServiceLoader using this interface's fully qualified name.
 * Deliberately independent of inference, CDI, reactive libraries, logging, and accelerator SDKs.
 */
public interface AljabrProvider extends AutoCloseable {
    ProviderDescriptor descriptor();
    /** Initializes this provider for one Aljabr runtime. */
    default void initialize(ProviderContext context) throws ProviderException { }
    /** Releases provider-owned resources; tolerate calls after partial initialization. */
    @Override
    default void close() throws ProviderException { }
}
