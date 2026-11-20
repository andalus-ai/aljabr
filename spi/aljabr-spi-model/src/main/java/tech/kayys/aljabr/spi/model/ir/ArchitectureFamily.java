package tech.kayys.aljabr.spi.model.ir;

/**
 * Extensible architecture family identifier with an optional broad category.
 * The identifier is the stable identity; the kind is descriptive metadata.
 */
public record ArchitectureFamily(String id, ArchitectureKind kind) {

    public ArchitectureFamily {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("id must not be blank");
        }
        if (kind == null) {
            throw new IllegalArgumentException("kind must not be null");
        }
    }

    public static ArchitectureFamily custom(String id) {
        return new ArchitectureFamily(id, ArchitectureKind.CUSTOM);
    }
}
