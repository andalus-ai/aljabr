package tech.kayys.aljabr.spi.plugin;

import java.util.List;
import java.util.Optional;

/**
 * Registry for managing plugin lifecycle and discovery.
 */
public interface PluginRegistry {
    void initialize();

    void registerPlugin(AljabrPlugin plugin);

    void unregisterPlugin(String pluginId);

    List<AljabrPlugin> all();

    <T extends AljabrPlugin> List<T> byType(Class<T> type);

    Optional<AljabrPlugin> byId(String pluginId);

    void reload(String pluginId);

    List<AljabrPlugin.PluginMetadata> listMetadata();

    boolean isHealthy();

    List<String> unhealthyPlugins();

    void shutdownAll();
}
