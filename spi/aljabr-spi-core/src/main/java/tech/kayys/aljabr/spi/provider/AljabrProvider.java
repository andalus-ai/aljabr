package tech.kayys.aljabr.spi.provider;

/**
 * Base contract for built-in and independently published providers.
 * Discoverable through ServiceLoader using this interface's fully qualified name.
 * Deliberately independent of inference, CDI, reactive libraries, logging, and accelerator SDKs.
 *
 * <p>A provider instance belongs to one Aljabr runtime. The runtime discovers providers
 * first, then explicitly initializes the instances it elects to use. Initialization
 * must be performed at most once per instance. If initialization fails, the instance
 * must be treated as unusable and closed rather than initialized again.</p>
 *
 * <p>Closing a provider is terminal and idempotent. It must release resources acquired
 * during successful or partial initialization and close any sessions it owns before
 * releasing resources shared by those sessions. A closed provider must reject further
 * initialization and resource acquisition.</p>
 */
public interface AljabrProvider extends AutoCloseable {
    ProviderDescriptor descriptor();

    /** Initializes this provider for one Aljabr runtime; call at most once per instance. */
    default void initialize(ProviderContext context) throws ProviderException { }

    /**
     * Releases provider-owned resources, including outstanding provider-owned sessions.
     * Must be idempotent and safe after partial initialization.
     */
    @Override
    default void close() throws ProviderException { }
}
