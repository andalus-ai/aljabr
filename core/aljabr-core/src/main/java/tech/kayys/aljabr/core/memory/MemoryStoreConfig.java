package tech.kayys.aljabr.core.memory;
/**
 * 
 * Configuration class for memorystore settings.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public class MemoryStoreConfig {
    private String engineType;
    private String dbPath;

    public MemoryStoreConfig(String engineType, String dbPath) {
        this.engineType = engineType;
        this.dbPath = dbPath;
    }

    public String getEngineType() {
        return engineType;
    }

    public String getDbPath() {
        return dbPath;
    }
}
